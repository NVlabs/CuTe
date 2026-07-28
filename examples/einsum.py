# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pycute import *


def batch_gemm(A : Tensor,   # (M,K,L)
               B : Tensor,   # (N,K,L)
               C : Tensor):  # (M,N,L)
  """
  A generic batch GEMM operation.
  """
  if rank(A) != 3 or rank(B) != 3 or rank(C) != 3:
    raise ValueError("batch_gemm: operands must be rank-3")
  if size[0](A) != size[0](C):
    raise ValueError("Row sizes M do not match")
  if size[0](B) != size[1](C):
    raise ValueError("Column sizes N do not match")
  if size[1](A) != size[1](B):
    raise ValueError("Reduction sizes K do not match")
  if size[2](A) != size[2](B) != size[2](C):
    raise ValueError("Batch sizes L do not match")

  for l in range(size[2](C)):
    for k in range(size[1](B)):
      for n in range(size[1](C)):
        for m in range(size[0](C)):
          C[m,n,l] += A[m,k,l] * B[n,k,l]


def _fold(tensor : Tensor, labels, groups) -> Tensor:
  """
  Fold `tensor` into a rank-3 view by concatenating its modes per `groups`
  (a triple of label lists). Each group becomes one (multi-)mode and an empty
  group becomes a size-1 mode. Shares the accessor -- no data is copied.
  """
  pos = {ch: i for i, ch in enumerate(labels)}
  def group(g):
    if not g:
      return Layout(1, 0)
    if len(g) == 1:
      return tensor.layout[pos[g[0]]]
    return make_layout([tensor.layout[pos[ch]] for ch in g])
  return Tensor(tensor.accessor, make_layout([group(g) for g in groups]))


def einsum(subscripts : str, A : Tensor, B : Tensor, C : Tensor) -> Tensor:
  """
  Evaluate an explicit binary einsum by folding into `batch_gemm`.

  `subscripts` has the form "a_modes,b_modes->c_modes" with one label per mode
  (e.g. "stupr,qtru->stqp"). Each label is classified as a row (M, in A and C),
  column (N, in B and C), reduction (K, in A and B), or batch (L, in A, B and C)
  mode, and the operands are folded -- a zero-copy change of view -- into the
  canonical  A:(M,K,L)  B:(N,K,L)  C:(M,N,L)  layouts that `batch_gemm` consumes.

  The result accumulates into the caller-provided `C` (`C += A * B`); einsum
  never allocates or zeroes `C`, so zero `C` beforehand for assignment semantics.
  """
  # Parse the explicit equation "a_modes,b_modes->c_modes" into per-operand labels.
  parts = subscripts.replace(" ", "").split("->")
  if len(parts) != 2:
    raise ValueError(f"einsum: expected exactly one '->' in {subscripts!r}")
  inputs = parts[0].split(",")
  if len(inputs) != 2:
    raise ValueError(f"einsum: expected exactly two ','-separated inputs in {subscripts!r}")
  a_mode, b_mode, c_mode = list(inputs[0]), list(inputs[1]), list(parts[1])

  # Validate each operand and record every label's extent (must agree everywhere).
  extent = {}
  for name, tensor, mode in (("A", A, a_mode), ("B", B, b_mode), ("C", C, c_mode)):
    if rank(tensor) != len(mode):
      raise ValueError(f"einsum: operand {name} has rank {rank(tensor)} "
                       f"but {len(mode)} labels {''.join(mode)!r}")
    if len(set(mode)) != len(mode):
      raise ValueError(f"einsum: repeated label within operand {name}: {''.join(mode)!r}")
    for i, ch in enumerate(mode):
      e = size[i](tensor)
      if extent.setdefault(ch, e) != e:
        raise ValueError(f"einsum: label {ch!r} has inconsistent extents {extent[ch]} and {e}")

  # Classify each label by where it appears (Whitepaper, "Tensors and Folding"),
  # in first-appearance order:
  #   row M -- A,C    col N -- B,C    red K -- A,B    bat L -- A,B,C
  SA, SB, SC = set(a_mode), set(b_mode), set(c_mode)
  order = list(dict.fromkeys(a_mode + b_mode + c_mode))
  row = [x for x in order if x in SA and x in SC and x not in SB]
  col = [x for x in order if x in SB and x in SC and x not in SA]
  red = [x for x in order if x in SA and x in SB and x not in SC]
  bat = [x for x in order if x in SA and x in SB and x in SC]
  unsupported = (SA | SB | SC) - set(row + col + red + bat)
  if unsupported:
    raise ValueError(f"einsum: label(s) {sorted(unsupported)} appear in only one operand; "
                     f"each label must appear in at least two of A, B, C")

  A3 = _fold(A, a_mode, (row, red, bat))   # (M,K,L)
  B3 = _fold(B, b_mode, (col, red, bat))   # (N,K,L)
  C3 = _fold(C, c_mode, (row, col, bat))   # (M,N,L)

  batch_gemm(A3, B3, C3)                   # C += A * B
  return C