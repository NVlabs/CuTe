# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Optimized COPY (Whitepaper §2.6.1).

Reshapes the iteration space of `pycute.alg.ref.copy` before running it, in
three stages, each establishing one invariant:

* **Common domain** — `greatest_common_domain` finds a tiler admissible for
  both layouts, since equal-size shapes need not share one (`(5,7)` vs
  `(7,5)`). Dividing by it splits each tensor into `(Compat, InCompat)`; only
  `Compat` can be optimized, and `InCompat` is iterated as-is.
* **Nullspace** — `nullspace(dst)` collects the coordinates that share one
  destination address. A source that varies across them is a write-after-write
  error and is rejected; a source that is constant across them is writing the
  same value repeatedly, so all but one write is dropped.
* **Alignment** — `right_inverse(dst)` reorders the domain into destination
  memory order, leaving the innermost run contiguous in `dst` (Whitepaper
  §3.4.2 AutoVectorization). The `src` layout over that run gives the width
  and stride a vectorized transfer would use.

The surviving runs are handed to the reference loop, so the result is always
that of `pycute.alg.ref.copy`, down to element-at-a-time in the worst case.
"""

from __future__ import annotations

from pycute import (Tensor, size, stride, coalesce, greatest_common_domain,
                    logical_divide, nullspace, right_inverse)

from .ref.copy import copy as _memcpy


def copy(src: Tensor, dst: Tensor) -> None:
  """
  COPY with Layout Analysis
  """
  if size(src) != size(dst):
    raise ValueError(f"copy: size mismatch {size(src)} != {size(dst)}")

  # Extract the domain-compatible portions

  common_domain = greatest_common_domain(src, dst)
  src_c = logical_divide(src, common_domain)  # (Compat, InCompat)
  dst_c = logical_divide(dst, common_domain)  # (Compat, InCompat)

  # Account for nullspaces of dst and src within compatible portions

  null_dst = nullspace(dst_c.layout[0])
  src_n = logical_divide(src_c, (null_dst, None))  # ((Null, NonNull), InCompat)
  dst_n = logical_divide(dst_c, (null_dst, None))  # ((Null, NonNull), InCompat)

  if dst_n.layout[0][0](size(null_dst) - 1) != 0:
    raise ValueError(f"Sanity: nullspace definition error")
  if src_n.layout[0][0](size(null_dst) - 1) != 0:
    raise ValueError(f"Write-after-write error detected in dst: {dst}")

  # Slice the common stride-0 modes out
  src_n = src_n[(0,None),None]  # (NonNull, Incompat)
  dst_n = dst_n[(0,None),None]  # (NonNull, Incompat)

  # Vectorization of the compatible portions

  inv_dst = right_inverse(dst_n.layout[0])
  src_v = logical_divide(src_n, (inv_dst, None))  # ((Id, NonId), Incompat)
  dst_v = logical_divide(dst_n, (inv_dst, None))  # ((Id, NonId), Incompat)

  # This section is technically instruction-aware,
  # we are looking for a common contiguous run (vector) in src and dst
  # but other instructions may look for other patterns completely
  src_dst_v = coalesce(logical_divide(src_n.layout[0], inv_dst)[0])[0]
  try:
    vec_size = 1
    if stride(src_dst_v) == 1:
      vec_size = size(src_dst_v)
  except TypeError:
    pass

  src_v = logical_divide(src_v, (vec_size, None))  # ((V, Rest), Incompat)
  dst_v = logical_divide(dst_v, (vec_size, None))  # ((V, Rest), Incompat)

  # Slice and dispatch to an optimized/vectorized array_copy

  for j in range(size[1](dst_v)):
    for i in range(size[0][1](dst_v)):
      _memcpy(src_v[(None,i),j], dst_v[(None,i),j])
