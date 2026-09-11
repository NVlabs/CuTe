# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GEMM — batched matrix multiplication (Whitepaper §2.6.2).

A rank-3 algorithm: iterate the row, column, reduction and batch index spaces
and remap through each tensor's layout. Different `(A, B, C)` layout triples
realize gemm, gemv, ger, dot, gett, grouped gemm, conv, etc.
"""

from __future__ import annotations

from pycute import Layout, Tensor, make_layout, rank, size


def gemm(A: Tensor, B: Tensor, C: Tensor) -> None:
  """
  Reference batched GEMM: `C[m,n,l] += A[m,k,l] * B[n,k,l]`.

  Each operand is rank-3, but a mode may be a multi-mode, so an operand of any
  rank participates once its modes are folded into these four roles -- row `M`,
  column `N`, reduction `K` and batch `L`.

  Rank-2 operands are the unbatched case.

  Pre-conditions:
    rank(A) == rank(B) == rank(C), and either 3 or 2
    size[0](A) == size[0](C)                  the row extent M
    size[0](B) == size[1](C)                  the column extent N
    size[1](A) == size[1](B)                  the reduction extent K
    size[2](A) == size[2](B) == size[2](C)    the batch extent L
  """
  # In the non-batched case, append a size-1 batch mode for convenience.
  if rank(A) == 2 and rank(B) == 2 and rank(C) == 2:
    A, B, C = (Tensor(T.accessor, make_layout([T.layout[0], T.layout[1], Layout(1, 0)])) for T in (A, B, C))
  if rank(A) != 3 or rank(B) != 3 or rank(C) != 3:
    raise ValueError(f"gemm: operands must be all rank-3 or all rank-2, "
                     f"got {rank(A)}, {rank(B)}, {rank(C)}")
  if size[0](A) != size[0](C):
    raise ValueError(f"gemm: row size mismatch {size[0](A)} != {size[0](C)}")
  if size[0](B) != size[1](C):
    raise ValueError(f"gemm: column size mismatch {size[0](B)} != {size[1](C)}")
  if size[1](A) != size[1](B):
    raise ValueError(f"gemm: reduction size mismatch {size[1](A)} != {size[1](B)}")
  if size[2](A) != size[2](C) or size[2](B) != size[2](C):
    raise ValueError(f"gemm: batch size mismatch {size[2](A)}, {size[2](B)}, {size[2](C)}")

  for l in range(size[2](C)):
    for k in range(size[1](A)):
      for n in range(size[1](C)):
        for m in range(size[0](C)):
          C[m, n, l] += A[m, k, l] * B[n, k, l]
