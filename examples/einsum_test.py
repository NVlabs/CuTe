# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for examples/einsum.py.

Run from the repository root:
    pytest examples/einsum_test.py

The oracle is a dependency-free brute force: it enumerates every combination of
label values and accumulates A * B into the output coordinate, so it validates
einsum's folding against the contraction's definition with no numpy dependency.
"""

import ctypes
import itertools
import logging

import pytest

from pycute import *
from examples.einsum import einsum, _fold

logger = logging.getLogger()


def _make(mode, extents, stride=None, value=lambda i: 0.0):
  """A flat Tensor whose modes are `mode`, filled in 1-D layout order by `value`."""
  shape  = tuple(extents[ch] for ch in mode)
  layout = Layout(shape, stride) if stride is not None else Layout(shape)
  T = make_tensor(layout, dtype=ctypes.c_double)
  for i in range(size(T)):
    T[i] = value(i)
  return T


def _reference(a_mode, b_mode, c_mode, extents, A, B):
  """Brute-force {c_coord: value} for C[c] = sum over the contracted labels of A*B."""
  labels = list(dict.fromkeys(a_mode + b_mode + c_mode))
  out = {}
  for combo in itertools.product(*[range(extents[l]) for l in labels]):
    assign = dict(zip(labels, combo))
    a_idx = tuple(assign[l] for l in a_mode)
    b_idx = tuple(assign[l] for l in b_mode)
    c_idx = tuple(assign[l] for l in c_mode)
    out[c_idx] = out.get(c_idx, 0.0) + A[a_idx] * B[b_idx]
  return out


class TestEinsum:

  def _check(self, subscripts, extents,
             a_stride=None, b_stride=None, c_stride=None, c_init=0.0):
    print("=" * 20)
    print(subscripts, extents)

    a_mode, b_mode, c_mode = (list(s) for s in
                              subscripts.replace(" ", "").replace("->", ",").split(","))
    A = _make(a_mode, extents, a_stride, value=lambda i: i + 1)
    B = _make(b_mode, extents, b_stride, value=lambda i: 2 * i + 3)
    C = _make(c_mode, extents, c_stride, value=lambda i: c_init)

    logger.info(f"\n{subscripts}:\n    {A}\n  * {B}\n -> {C}")

    assert einsum(subscripts, A, B, C) is C   # in-place, returns C

    # Validate the result
    expected = _reference(a_mode, b_mode, c_mode, extents, A, B)
    for c_idx, val in expected.items():
      assert C[c_idx] == pytest.approx(c_init + val, abs=1e-7), f"{subscripts} at {c_idx}"

  # -- correctness over a range of contractions --------------------------------

  def test_gemm(self):
    self._check("mk,nk->mn", dict(m=3, n=4, k=5))

  def test_gemm_transposed_output(self):
    self._check("mk,nk->nm", dict(m=3, n=4, k=5))

  def test_gemm_strided_operands(self):
    # Non-compact (row-major-ish) strides exercise the layout fold, not just shape.
    self._check("mk,nk->mn", dict(m=3, n=4, k=5),
                a_stride=(5, 1), b_stride=(5, 1), c_stride=(4, 1))

  def test_batched_leading(self):
    self._check("bmk,bnk->bmn", dict(b=2, m=3, n=4, k=5))

  def test_batched_trailing(self):
    self._check("mkb,nkb->mnb", dict(b=2, m=3, n=4, k=5))

  def test_multi_row_and_reduction(self):
    # Two row modes and two reduction modes -> hierarchical M and K.
    self._check("ijp,np->ijn", dict(i=2, j=3, p=4, n=5))

  def test_whitepaper_contraction(self):
    # C_stqp = A_stupr * B_qtru  (Whitepaper eq. for tensor folding).
    self._check("stupr,qtru->stqp", dict(s=2, t=3, u=2, p=2, r=3, q=2))

  def test_accumulates_into_nonzero_C(self):
    self._check("mk,nk->mn", dict(m=2, n=2, k=3), c_init=10.0)

  # -- classic vector / matrix products ----------------------------------------

  def test_inner_product(self):
    # Inner product: the single reduction mode collapses to a rank-0 scalar C.
    self._check("i,i->", dict(i=6))

  def test_outer_product(self):
    # Outer product: M=i, N=j, C[i,j] = a[i] * b[j].
    self._check("i,j->ij", dict(i=3, j=4))

  def test_elementwise_product(self):
    # Hadamard product: the shared label is a batch mode (in A, B and C).
    self._check("i,i->i", dict(i=6))

  def test_elementwise_product_2d(self):
    self._check("ij,ij->ij", dict(i=3, j=4))

  def test_scalar_scaling(self):
    # Scalar product: every label is a batch mode of one side.
    self._check("i,->i", dict(i=5))

  def test_scalar_matrix(self):
    # Scalar product: every label is a batch mode of one side.
    self._check("mn,->mn", dict(m=5, n=6))

  def test_scalar_transpose(self):
    # Scalar product: every label is a batch mode of one side.
    self._check("mn,->nm", dict(m=5, n=6))

  def test_matrix_vector(self):
    self._check("ij,j->i", dict(i=3, j=4))

  def test_vector_matrix(self):
    self._check("i,ij->j", dict(i=3, j=4))

  def test_batched_matrix_vector(self):
    self._check("bij,bj->bi", dict(b=2, i=3, j=4))

  # -- folded view structure ---------------------------------------------------

  def test_fold_shapes(self):
    # Fold A_stupr into (M=s,p ; K=u,r ; L=t) and check the rank-3 view's sizes.
    A = _make("stupr", dict(s=2, t=3, u=2, p=4, r=5))
    A3 = _fold(A, "stupr", (["s", "p"], ["u", "r"], ["t"]))   # (M,K,L)
    assert rank(A3) == 3
    assert size[0](A3) == 2 * 4   # M = s,p
    assert size[1](A3) == 2 * 5   # K = u,r
    assert size[2](A3) == 3       # L = t

  def test_fold_empty_batch_is_size_one(self):
    A = _make("mk", dict(m=3, k=5))
    A3 = _fold(A, "mk", (["m"], ["k"], []))      # empty batch group
    assert size[2](A3) == 1

  # -- validation errors -------------------------------------------------------

  def _abc(self, extents=dict(m=3, n=4, k=5)):
    return (_make("mk", extents),
            _make("nk", extents),
            _make("mn", extents))

  def test_error_missing_arrow(self):
    A, B, C = self._abc()
    with pytest.raises(ValueError):
      einsum("mk,nk", A, B, C)

  def test_error_three_inputs(self):
    A, B, C = self._abc()
    with pytest.raises(ValueError):
      einsum("mk,nk,xy->mn", A, B, C)

  def test_error_single_operand_label(self):
    A = _make("mkx", dict(m=3, n=4, k=5, x=2))   # 'x' only in A
    B = _make("nk",  dict(m=3, n=4, k=5, x=2))
    C = _make("mn",  dict(m=3, n=4, k=5, x=2))
    with pytest.raises(ValueError):
      einsum("mkx,nk->mn", A, B, C)

  def test_error_repeated_label(self):
    A = _make("mm", dict(m=3, n=4))
    B = _make("nm", dict(m=3, n=4))
    C = _make("mn", dict(m=3, n=4))
    with pytest.raises(ValueError):
      einsum("mm,nm->mn", A, B, C)

  def test_error_rank_mismatch(self):
    A, B, C = self._abc()
    with pytest.raises(ValueError):
      einsum("mkj,nk->mn", A, B, C)   # 3 labels for a rank-2 A

  def test_error_extent_mismatch(self):
    A = _make("mk", dict(m=3, k=5))
    B = _make("nk", dict(n=4, k=6))   # k disagrees: 5 vs 6
    C = _make("mn", dict(m=3, n=4))
    with pytest.raises(ValueError):
      einsum("mk,nk->mn", A, B, C)
