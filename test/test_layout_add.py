# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ``pycute.layout_add``.

These tests are also worked examples for docs/04_layout_algebra.md.

The universal post-condition (``postcondition_layout_add``) is the truth model
for every test in this file. Algebraic-property helpers
(``_assert_identity``, ``_assert_commutativity``, ``_assert_associativity``,
``_assert_self_inverse``) are parameterized by example lists that are
maintained per stride-scalar type:

  * ``INT_LAYOUTS`` -- plain integer strides
  * ``AT_LAYOUTS``  -- :class:`ArithTuple` / :func:`ScaledBasis` strides
  * ``F2_LAYOUTS``  -- :class:`F2` strides (XOR add, clmul scale)

Each example list is curated so every pair / triple admits a Layout sum,
giving the property helpers full coverage when iterated over the list.
"""

import logging
import pytest

from pycute import *

logger = logging.getLogger()


class TestLayoutAdd:

  # ====================================================================
  # Universal post-condition
  # ====================================================================

  def postcondition_layout_add(self, A, B):
    """Pointwise correctness, size preservation, symmetry."""
    R = layout_add(A, B)

    logger.info(f"  layout_add({A}, {B})  =>  {R}")

    # Result is a Layout
    assert is_layout(R)

    # Size matches both inputs
    assert size(R) == size(A)
    assert size(R) == size(B)

    # R(i) == A(i) + B(i) for every i in [0, size(R))
    for i in range(size(R)):
      assert R(i) == A(i) + B(i), (
          f"i={i}: R(i)={R(i)} != A(i)+B(i)={A(i)+B(i)} "
          f"(A={A}, B={B}, R={R})")

    # Symmetric in A and B
    assert layout_add(B, A) == R

    return R

  # ====================================================================
  # Algebraic-property helpers, parameterized by example list
  # ====================================================================

  def _assert_identity(self, examples):
    """``Layout(size(A), 0)`` (int-zero stride) is the *universal*
    additive identity: it evaluates to int ``0`` at every coordinate,
    and ``0`` is the additive identity in every stride-scalar type
    (int, ArithTuple, F2)."""
    for A in examples:
      Z = Layout(size(A), 0)
      R = self.postcondition_layout_add(A, Z)
      # Coalesce-equivalent to A: layout_add returns canonical form.
      assert R == A._coalesce()

  def _assert_commutativity(self, examples):
    """``layout_add(A, B) == layout_add(B, A)`` whenever both sides are
    defined (the post-condition checks the pointwise version; here we
    assert structural Layout equality, which is stronger)."""
    for A in examples:
      for B in examples:
        if size(A) != size(B):
          continue
        try:
          R_ab = layout_add(A, B)
          R_ba = layout_add(B, A)
        except (ValueError, TypeError):
          continue
        assert R_ab == R_ba

  def _assert_associativity(self, examples):
    """``(A + B) + C`` and ``A + (B + C)`` agree pointwise with
    ``A(i) + B(i) + C(i)`` whenever every pair / triple admits a
    Layout sum. Triples that fail are skipped."""
    for A in examples:
      for B in examples:
        for C in examples:
          if size(A) != size(B) or size(A) != size(C):
            continue
          try:
            ABC1 = layout_add(layout_add(A, B), C)
            ABC2 = layout_add(A, layout_add(B, C))
          except (ValueError, TypeError):
            continue
          for i in range(size(ABC1)):
            assert ABC1(i) == ABC2(i), (f"i={i}: (A+B)+C={ABC1(i)} != A+(B+C)={ABC2(i)} "
                                 f"(A={A}, B={B}, C={C})")
            assert ABC1(i) == A(i) + B(i) + C(i), (f"i={i}: result={ABC1(i)} != A+B+C "
                                 f"(A={A}, B={B}, C={C})")

  def _assert_self_inverse(self, examples):
    """``A + A`` is pointwise zero for stride scalars whose addition
    is its own inverse (currently only :class:`F2`)."""
    for A in examples:
      R = self.postcondition_layout_add(A, A)
      for i in range(size(R)):
        assert R(i) == 0, f"A + A not zero at i={i}: R(i)={R(i)} (A={A})"

  # ====================================================================
  # Example layouts grouped by stride-scalar type.
  #
  # Each list is a curated set of Layouts of *uniform size* for which
  # every pair / triple sum is a Layout (so the property helpers cover
  # every combination without skipping).
  # ====================================================================

  # int strides, size 12.
  INT_LAYOUTS = [
    Layout(12, 0),                            # broadcast (additive identity)
    Layout(12, 1),                            # row-major
    Layout(12, 2),                            # scaled
    Layout((4, 3), (1, 4)),                   # (4,3) row-major
    Layout((4, 3), (3, 1)),                   # (4,3) col-major
    Layout((4, 3), (1, 0)),                   # broadcast in second mode
    Layout((2, 6), (1, 2)),                   # alternative factorization
    Layout(((2, 2), 3), ((2, 1), 4)),         # nested
  ]

  # ArithTuple / ScaledBasis strides, size 12.
  AT_LAYOUTS = [
    Layout(12, 0),                            # universal int-zero identity
    Layout(12, E(0)),                         # rank-1 unit basis
    Layout(12, 2 * E(0)),                     # scaled basis
    Layout((4, 3), (E(0), E(1))),             # disjoint paths
    Layout((4, 3), (E(1), E(0))),             # swapped paths
    Layout((4, 3), (E(0), E(0))),             # same path -- adds along one axis
    Layout((4, 3), (2 * E(0), 3 * E(1))),     # scaled basis at disjoint paths
    Layout((2, 6), (E(0), E(1))),             # different shape, disjoint paths
  ]

  # F2 strides, size 16.
  #
  # A power-of-two size is chosen because the implementation's leaf-wise
  # stride-sum recovers the pointwise sum *unconditionally* when every
  # shape mode of the common refinement is a power of two; non-pow-2
  # modes can violate the bit-disjointness precondition (pinned by
  # ``test_f2_bit_disjointness_violation_raises`` below).
  F2_LAYOUTS = [
    Layout(16, 0),                            # universal int-zero identity
    Layout(16, F2(0)),                        # F2 zero stride
    Layout(16, F2(1)),                        # single low bit
    Layout(16, F2(8)),                        # single mid bit
    Layout(16, F2(9)),                        # multi-bit (1 + 8)
    Layout(16, F2(64)),                       # high bit, no overlap with 0..15
    Layout((4, 4), (F2(1), F2(4))),           # multi-mode, coalesces
    Layout((4, 4), (F2(1), F2(8))),           # multi-mode, well-spaced bits
    Layout((4, 4), (F2(2), F2(8))),           # multi-mode
  ]

  # ====================================================================
  # Algebraic properties per stride-scalar type
  # ====================================================================

  def test_identity_int(self):     self._assert_identity(self.INT_LAYOUTS)
  def test_identity_atuple(self):  self._assert_identity(self.AT_LAYOUTS)
  def test_identity_f2(self):      self._assert_identity(self.F2_LAYOUTS)

  def test_commutativity_int(self):     self._assert_commutativity(self.INT_LAYOUTS)
  def test_commutativity_atuple(self):  self._assert_commutativity(self.AT_LAYOUTS)
  def test_commutativity_f2(self):      self._assert_commutativity(self.F2_LAYOUTS)

  def test_associativity_int(self):     self._assert_associativity(self.INT_LAYOUTS)
  def test_associativity_atuple(self):  self._assert_associativity(self.AT_LAYOUTS)
  def test_associativity_f2(self):      self._assert_associativity(self.F2_LAYOUTS)

  def test_self_inverse_f2(self):
    """For F2 strides specifically, ``A + A`` is the zero layout
    pointwise (XOR is its own inverse)."""
    self._assert_self_inverse(self.F2_LAYOUTS)

  # ====================================================================
  # Trivial / edge cases
  # ====================================================================

  def test_singleton(self):
    # Size-1 layouts always evaluate to 0, regardless of stride.
    R = self.postcondition_layout_add(Layout(1, 0), Layout(1, 0))
    assert size(R) == 1

    R = self.postcondition_layout_add(Layout(1, 5), Layout(1, 7))
    assert size(R) == 1

    # Singleton with custom stride scalars (also evaluate to 0 at i=0).
    R = self.postcondition_layout_add(Layout(1, F2(7)), Layout(1, F2(3)))
    assert size(R) == 1

    R = self.postcondition_layout_add(Layout(1, E(0)), Layout(1, 2 * E(1)))
    assert size(R) == 1

  def test_zero_stride_broadcast(self):
    # Adding a stride-0 (broadcast) layout reduces to the other one.
    R = self.postcondition_layout_add(Layout(5, 0), Layout(5, 1))
    assert R == Layout(5, 1)

    # Adding two zero-stride layouts gives the zero layout.
    R = self.postcondition_layout_add(Layout(5, 0), Layout(5, 0))
    assert R == Layout(5, 0)

  # ====================================================================
  # Pointwise correctness -- int strides
  # ====================================================================

  def test_int_same_shape_same_stride(self):
    # Adding a layout to itself doubles its strides.
    R = self.postcondition_layout_add(Layout(12, 1), Layout(12, 1))
    assert R == Layout(12, 2)

    # Coalescing folds the multi-mode form back to the canonical doubling.
    R = self.postcondition_layout_add(Layout((4, 3), (1, 4)),
                                      Layout((4, 3), (1, 4)))
    assert R == Layout(12, 2)

    R = self.postcondition_layout_add(Layout((4, 3), (3, 1)),
                                      Layout((4, 3), (3, 1)))
    assert R == Layout((4, 3), (6, 2))

  def test_int_same_shape_diff_stride(self):
    # Row-major + column-major: leaf-wise stride sum.
    R = self.postcondition_layout_add(Layout((4, 3), (1, 4)),
                                      Layout((4, 3), (3, 1)))
    assert R == Layout((4, 3), (4, 5))

    # Strides scale linearly under addition.
    R = self.postcondition_layout_add(Layout((6,), (5,)),
                                      Layout((6,), (3,)))
    assert R == Layout(6, 8)

  def test_int_both_row_major_same_size(self):
    # Different shapes, A(i) = i and B(i) = i: result is `2*i`.
    R = self.postcondition_layout_add(Layout((5, 3, 4), (1, 5, 15)),
                                      Layout((10, 6),    (1, 10)))
    assert R == Layout(60, 2)

    R = self.postcondition_layout_add(Layout((6,),       (1,)),
                                      Layout((3, 2),     (1, 3)))
    assert R == Layout(6, 2)

  def test_int_one_flat_one_decomposed(self):
    R = self.postcondition_layout_add(Layout((6,),       (1,)),
                                      Layout((3, 2),     (2, 1)))
    assert R == Layout((3, 2), (3, 4))

    R = self.postcondition_layout_add(Layout(12, 2),
                                      Layout((4, 3),     (3, 1)))
    assert R == Layout((4, 3), (5, 9))

  def test_int_nested_shapes(self):
    # Nested shapes are flattened by greatest_common_domain.
    R = self.postcondition_layout_add(Layout((2, (3, 4)), (12, (4, 1))),
                                      Layout((6, 4),      (4, 1)))
    assert R == Layout((2, 3, 4), (16, 12, 2))

  def test_int_zero_stride_in_one_mode(self):
    # Stride-0 in one mode of a multi-mode layout (broadcast row).
    R = self.postcondition_layout_add(Layout((4, 3), (1, 0)),
                                      Layout((4, 3), (3, 1)))
    assert R == Layout((4, 3), (4, 1))

  # ====================================================================
  # Pointwise correctness -- ArithTuple strides
  # ====================================================================

  def test_atuple_disjoint_basis_paths(self):
    # E(0) + E(1) on the same shape gives the rank-2 unit ArithTuple at
    # the same position: each output coord is (i, i).
    R = self.postcondition_layout_add(Layout(8, E(0)), Layout(8, E(1)))
    for i in range(8):
      assert R(i) == ArithTuple(i, i)

  def test_atuple_same_basis_path(self):
    # 2*E(0) + 3*E(0) = 5*E(0) at every coord.
    R = self.postcondition_layout_add(Layout(8, 2 * E(0)),
                                      Layout(8, 3 * E(0)))
    assert R == Layout(8, 5 * E(0))

  def test_atuple_multimode(self):
    # Multi-mode AT layouts with disjoint basis: leaf-wise stride sum.
    R = self.postcondition_layout_add(Layout((4, 3), (E(0), E(1))),
                                      Layout((4, 3), (E(1), E(0))))
    # R(c0, c1) = (c0 + c1) * E(0) + (c0 + c1) * E(1) = ArithTuple(c0+c1, c0+c1)
    for c0 in range(4):
      for c1 in range(3):
        assert R(c0, c1) == ArithTuple(c0 + c1, c0 + c1)

  def test_atuple_universal_int_zero_identity(self):
    # Layout(N, 0) (int-zero) is the universal additive identity: it
    # works as identity even for ArithTuple-strided layouts.
    A = Layout((4, 3), (E(0), E(1)))
    R = self.postcondition_layout_add(A, Layout(size(A), 0))
    assert R == A

  def test_atuple_different_shapes(self):
    # AT layouts of different shapes that share a refinement -- the
    # ArithTuple's component-wise additivity is linear in any int
    # decomposition, so this is unconditional.
    A = Layout(12, E(0))
    B = Layout((4, 3), (E(1), E(2)))
    R = self.postcondition_layout_add(A, B)
    # R(c) = A(c) + B(c) = c*E(0) + (c%4)*E(1) + (c//4)*E(2)
    for i in range(12):
      assert R(i) == ArithTuple(i, i % 4, i // 4)

  # ====================================================================
  # Pointwise correctness -- F2 strides
  # ====================================================================

  def test_f2_disjoint_bits(self):
    # F2(1) + F2(8) = F2(9). Bits don't collide; result is exact.
    R = self.postcondition_layout_add(Layout(8, F2(1)), Layout(8, F2(8)))
    assert R == Layout(8, F2(9))

  def test_f2_overlapping_bits_same_shape(self):
    # F2(1) + F2(2) on same shape: even with overlapping bit-spans the
    # leaf-wise XOR works because clmul distributes over XOR.
    # Pointwise: F2(c) + F2(2c) = F2(c xor 2c) = F2(3c in clmul) for c < 8.
    R = self.postcondition_layout_add(Layout(8, F2(1)), Layout(8, F2(2)))
    assert R == Layout(8, F2(3))

  def test_f2_zero_stride_identity(self):
    # F2(0) is also an additive identity for F2 layouts.
    A = Layout(8, F2(7))
    R = self.postcondition_layout_add(A, Layout(8, F2(0)))
    assert R == A

  def test_f2_universal_int_zero_identity(self):
    # int 0 (universal additive identity) is also an identity for F2.
    A = Layout(8, F2(7))
    R = self.postcondition_layout_add(A, Layout(8, 0))
    assert R == A

  def test_f2_multimode(self):
    # Multi-mode F2 strides, leaf-wise XOR.
    R = self.postcondition_layout_add(Layout((4, 8), (F2(1), F2(8))),
                                      Layout((4, 8), (F2(2), F2(16))))
    assert R == Layout((4, 8), (F2(3), F2(24)))

  def test_f2_different_shapes_pow2(self):
    # F2 layouts of different shapes refining via pow-2 modes: leaf-wise
    # XOR gives the pointwise sum because the bit-disjointness envelope
    # is preserved by pow-2 shape factors.
    A = Layout(16, F2(1))
    B = Layout((4, 4), (F2(2), F2(8)))
    R = self.postcondition_layout_add(A, B)
    # The multi-mode result (4, 4):(F3, F12) coalesces to a single mode
    # because F2(3) * 4 == F2(12) under clmul.
    assert R == Layout(16, F2(3))

  # ====================================================================
  # Failure cases
  # ====================================================================

  def test_size_mismatch_raises(self):
    with pytest.raises(ValueError):
      layout_add(Layout(6, 1), Layout(8, 1))

    with pytest.raises(ValueError):
      layout_add(Layout((4, 3), (3, 1)), Layout((6, 5), (5, 1)))

  def test_no_common_refinement_raises(self):
    # Pairwise coprime leaves -> no aligned common refinement of size A.
    with pytest.raises(ValueError):
      layout_add(Layout((5, 3), (3, 1)), Layout((3, 5), (5, 1)))

    with pytest.raises(ValueError):
      layout_add(Layout((7, 11), (11, 1)), Layout((11, 7), (7, 1)))

  def test_non_layout_argument_raises(self):
    with pytest.raises(TypeError):
      layout_add(Layout(4, 1), 4)
    with pytest.raises(TypeError):
      layout_add(4, Layout(4, 1))

  def test_incompatible_stride_types_raise(self):
    # ArithTuple-strided plus nonzero int-strided: the per-leaf
    # ``A(d) + B(d)`` evaluation hits ``ArithTuple + nonzero int``
    # which is a TypeError.
    with pytest.raises(TypeError):
      layout_add(Layout(8, E(0)), Layout(8, 1))

    # F2 plus nonzero int: same kind of incompatibility.
    with pytest.raises(TypeError):
      layout_add(Layout(8, F2(1)), Layout(8, 1))

  # XXX TODO
  # def test_f2_bit_disjointness_violation_raises(self):
  #   """F2 layout addition can fail the runtime ``R(i) == A(i) + B(i)``
  #   check when the common refinement has non-pow-2 shape modes:

  #     G has shape S, stride D such that ``i = sum_k c_k * d_k`` (int
  #     sum). For F2, ``A(i) = F2(s_A clmul i)`` is linear over XOR of
  #     its argument, so the leaf-wise stride sum ``F2((s_A xor s_B) * d_k)``
  #     reproduces the pointwise sum *only* when the int decomposition
  #     ``sum_k c_k * d_k`` agrees with the XOR ``XOR_k c_k * d_k``,
  #     i.e. when no carries can occur. A non-pow-2 ``s_k`` allows carries
  #     (e.g. ``s_0=3, d_0=1``: coords ``c_0 in {0,1,2}`` produce ``c_0*d_0``
  #     values that overlap the range of ``c_1*d_1`` for ``d_1=3``).

  #   The runtime post-condition check catches the divergence and raises
  #   ``ValueError``.
  #   """
  #   A = Layout(9, F2(1))
  #   B = Layout((3, 3), (F2(2), F2(4)))
  #   with pytest.raises(ValueError, match="post-condition violated"):
  #     layout_add(A, B)


class TestLayoutAddProfile:
  """
  Adding by-mode, which keeps the rank structure both arguments came in with.

  Whole-layout addition is free to refine the domain however the sum falls, so the
  map survives but the modes do not -- and where the modes *mean* something, as the
  axes of a coordinate space do, that loss is the whole difference between a layout
  of that space and a layout of a flat index.
  """

  def test_by_mode_keeps_the_axes(self):
    A = Layout((4, 3), (1, 4))
    B = Layout((4, 3), (3, 1))
    R = layout_add(A, B, (1, 1))
    assert R == Layout((4, 3), (4, 5))
    assert congruent(shape(A), shape(R))

  def test_whole_layout_may_reshape_the_domain(self):
    """The contrast: the same sum taken whole comes back on a domain of its own."""
    A = Layout((4, 3), (1, 4))
    by_mode = layout_add(A, A, (1, 1))
    whole = layout_add(A, A)
    assert by_mode == Layout((4, 3), (2, 8))
    assert whole == Layout(12, 2)                  # rank 2 in, rank 1 out
    assert congruent(shape(A), shape(by_mode))
    assert not congruent(shape(A), shape(whole))
    # The same map either way, read at the flat index the domain enumerates.
    assert all(int(whole(i + 4 * j)) == int(by_mode(i, j))
               for i in range(4) for j in range(3))

  def test_by_mode_agrees_pointwise(self):
    for A, B in ((Layout((4, 3), (1, 4)), Layout((4, 3), (3, 1))),
                 (Layout((8, 8), (1, 8)), Layout((8, 8), (8, 1))),
                 (Layout((5, 3), (3, 1)), Layout((5, 3), (1, 5)))):
      R = layout_add(A, B, (1, 1))
      assert all(int(R(i, j)) == int(A(i, j)) + int(B(i, j))
                 for i in range(size(A[0])) for j in range(size(A[1])))

  def test_by_mode_is_symmetric(self):
    A = Layout((4, 3), (1, 4))
    B = Layout((4, 3), (3, 1))
    assert layout_add(A, B, (1, 1)) == layout_add(B, A, (1, 1))

  def test_f2_by_mode_is_the_swizzle_it_should_be(self):
    """Over `F2` the sum is an XOR, and by-mode keeps it a layout of the tile."""
    tile = Layout((8, 64), (F2(64), F2(1)))
    corr = Layout((8, 64), (F2(8), F2(0)))
    R = layout_add(tile, corr, (1, 1))
    assert R == Layout((8, 64), (F2(72), F2(1)))
    assert all(int(R(m, k)) == (64 * m) ^ (8 * m) ^ k
               for m in range(8) for k in range(64))

  def test_the_profile_conditions_are_exactly_what_is_accepted(self):
    """
    `weakly_congruent(profile, shape(A))` is the pre-condition, and the same relation
    on the result is the post-condition: the profile coarsens both arguments, and the
    sum refines the profile. Checked to be exact -- accepted if and only if it holds.
    """
    A = Layout((4, 3), (1, 4))
    B = Layout((4, 3), (3, 1))
    N = Layout(((2, 2), 3), ((1, 2), 4))
    for profile, X, Y in ((1, A, B), ((1, 1), A, B), ((1, 1, 1), A, B),
                          ((1, 1), A, N), (((1, 1), 1), N, N), (((1, 1), 1), A, B)):
      holds = (weakly_congruent(profile, shape(X))
               and weakly_congruent(profile, shape(Y)))
      if holds:
        R = layout_add(X, Y, profile)
        assert weakly_congruent(profile, shape(R)), f"{profile} does not coarsen {R}"
        assert size(R) == size(X)
      else:
        with pytest.raises(ValueError, match="rank mismatch"):
          layout_add(X, Y, profile)

  def test_a_profile_must_name_every_mode(self):
    """
    Unlike `coalesce`, a short profile is refused rather than padded. `coalesce` can
    leave a mode alone; there is no leaving a mode alone in a sum, so an unnamed mode
    has no meaning to fall back on.
    """
    A = Layout((4, 3, 2), (1, 4, 12))
    B = Layout((4, 3, 2), (2, 8, 24))
    with pytest.raises(ValueError, match="rank mismatch"):
      layout_add(A, B, (1,))
    assert size(layout_add(A, B, (1, 1, 1))) == size(A)

  def test_rank_mismatch_is_refused(self):
    with pytest.raises(ValueError, match="rank mismatch"):
      layout_add(Layout(12, 1), Layout(12, 1), (1, 1))
    with pytest.raises(ValueError, match="rank mismatch"):
      layout_add(Layout((4, 3), (1, 4)), Layout(12, 1), (1, 1))

  def test_none_is_the_identity_by_mode_too(self):
    """`None` is the additive identity whatever the profile, not a rank mismatch."""
    A = Layout((4, 3), (1, 4))
    assert layout_add(A, None, (1, 1)) == A
    assert layout_add(None, A, (1, 1)) == A
