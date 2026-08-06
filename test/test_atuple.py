# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.atuple
"""

import pytest

from pycute import *


class TestArithTuple:

  def test_atuple(self):

    assert ArithTuple((1,2,3)) + ArithTuple((7,8,9)) == ArithTuple((8,10,12))
    assert ArithTuple((1,2,3)) + (7,8,9) == (8,10,12)
    assert (7,8,9) + ArithTuple((1,2,3)) == (8,10,12)

    assert ArithTuple((1,2,(3,4))) + ArithTuple((7,8,(9,10))) == ArithTuple((8,10,(12,14)))
    assert ArithTuple((1,2,(3,4))) + (7,8,(9,10)) == (8,10,(12,14))
    assert (7,8,(9,10)) + ArithTuple((1,2,(3,4))) == (8,10,(12,14))

    assert ArithTuple(1,2,3) + ArithTuple(7,8,9) == ArithTuple(8,10,12)
    assert ArithTuple(1,2,3) + (7,8,9) == (8,10,12)
    assert (7,8,9) + ArithTuple(1,2,3) == (8,10,12)

    assert ArithTuple(1,2,(3,4)) + ArithTuple(7,8,(9,10)) == ArithTuple(8,10,(12,14))
    assert ArithTuple(1,2,(3,4)) + (7,8,(9,10)) == (8,10,(12,14))
    assert (7,8,(9,10)) + ArithTuple(1,2,(3,4)) == (8,10,(12,14))

    assert ArithTuple(1,2,3) + (7,) == (8,2,3)
    assert ArithTuple(1,2,3) * 4 == (4,8,12)
    assert 4 * ArithTuple(1,2,3) == (4,8,12)

    #assert is_tuple(ArithTuple(1,2,3))
    assert not is_int(ArithTuple(1,2,3))
    assert is_stride_scalar(ArithTuple(1,2,3))

    # Violates ATuple compatibility
    with pytest.raises(TypeError):
      ArithTuple((1,2,3)) + 7   # Not iter

  def test_atuple_sub(self):
    """``Z^S`` is an abelian group under elementwise addition, so subtraction is
    defined. Unlike ``+`` it does not commute, so ``0 - x`` negates ``x``."""
    assert ArithTuple(1,2,3) - ArithTuple(7,8,9) == (-6,-6,-6)
    assert ArithTuple(1,2,3) - (7,8,9) == (-6,-6,-6)
    assert (7,8,9) - ArithTuple(1,2,3) == (6,6,6)
    assert ArithTuple(1,2,(3,4)) - ArithTuple(7,8,(9,10)) == (-6,-6,(-6,-6))

    assert 4*E(0) - E(0) == 3*E(0)
    assert E(0) - E(0) == 0                  # the unique additive identity
    assert E(0) - 0 == E(0)
    assert 0 - E(0) == -1*E(0)               # ``__rsub__`` negates rather than delegating

    with pytest.raises(TypeError):
      ArithTuple(1,2,3) - 7                  # rank-incompatible, as for ``+``

  def test_atuple_sub_of_different_axes_cannot_cancel(self):
    """A cross-axis difference has two nonzero terms, so it is never its own
    additive inverse. `right_inverse` relies on this to reject a mode belonging to
    a different codomain axis without testing the axis explicitly."""
    for a, b in [(4*E(0), E(1)), (E(1), E(0)), (E(0,1), E(1,0))]:
      residue = a - b
      assert residue != 0
      assert residue + residue != 0          # only 0 is self-inverse in Z^S
    assert (E(0) - E(0)) + (E(0) - E(0)) == 0

  def test_atuple_lt(self):

    shape_trg = (4,(5,6),2)
    for i in range(size(shape_trg)):
      assert 0 < ArithTuple(idx2crd(i+1, shape_trg))
      for j in range(i+1, size(shape_trg)):
        assert ArithTuple(idx2crd(i, shape_trg)) < ArithTuple(idx2crd(j, shape_trg))
        assert ArithTuple(idx2crd(j, shape_trg)) > ArithTuple(idx2crd(i, shape_trg))

  def test_sbasis(self):

    assert ScaledBasis(42,[]) == 42
    assert ScaledBasis(42,[0]) == (42,0,0,0,0)
    assert ScaledBasis(42,[1]) == (0,42,0,0,0)
    assert ScaledBasis(42,[0,0]) == ((42,0,0,0,0),0,0,0,0)
    assert ScaledBasis(42,[0,1]) == ((0,42,0,0,0),0,0,0,0)
    assert ScaledBasis(42,[1,0]) == (0,(42,0,0,0,0),0,0,0)
    assert ScaledBasis(42,[1,1]) == (0,(0,42,0,0,0),0,0,0)

    assert 2 * ScaledBasis(42,[]) == 84
    assert 2 * ScaledBasis(42,[0]) == (84,0,0,0,0)
    assert 2 * ScaledBasis(42,[1]) == (0,84,0,0,0)
    assert 2 * ScaledBasis(42,[0,0]) == ((84,0,0,0,0),0,0,0,0)
    assert 2 * ScaledBasis(42,[0,1]) == ((0,84,0,0,0),0,0,0,0)
    assert 2 * ScaledBasis(42,[1,0]) == (0,(84,0,0,0,0),0,0,0)
    assert 2 * ScaledBasis(42,[1,1]) == (0,(0,84,0,0,0),0,0,0)

    test_seq = [[], [0], [1], [2],
                [0,0], [0,1], [0,2],
                [1,0], [1,1], [1,2],
                [2,0], [2,1], [2,2]]

    for seq0 in test_seq:
      for seq1 in test_seq:
        if seq0 == seq1:
          assert E(*seq0) == E(*seq1)
          assert 3 * E(*seq0) == 3 * E(*seq1)
          assert 0 * E(*seq0) == 0 * E(*seq1)
          assert 3 * E(*seq0) != 4 * E(*seq1)
        else:
          assert E(*seq0) != E(*seq1)
          assert 0 * E(*seq0) != 3 * E(*seq1)
          assert 0 * E(*seq0) == 0 * E(*seq1)

    #assert is_tuple(E(1))
    assert not is_int(E(1))
    assert is_stride_scalar(E(1))

  def test_sbasis_lt(self):

    test_seq = [[], [0], [1], [2],
                [0,0], [0,1], [0,2],
                [1,0], [1,1], [1,2],
                [2,0], [2,1], [2,2]]

    for seq in test_seq:
      assert 0 < E(*seq)

  def test_coord_layout(self):

    A = Layout((5,4), (E(0), E(1)))
    for i in range(size[0](A)):
      for j in range(size[1](A)):
        assert A(i,j) == ArithTuple(i,j)

    A = Layout((5,4), (E(0), E(2)))
    for i in range(size[0](A)):
      for j in range(size[1](A)):
        assert A(i,j) == ArithTuple(i,0,j)

    A = Layout((5,4), (E(2), E(1)))
    for i in range(size[0](A)):
      for j in range(size[1](A)):
        assert A(i,j) == ArithTuple(0,j,i)

    A = Layout((5,4), (E(2), 0))
    for i in range(size[0](A)):
      for j in range(size[1](A)):
        assert A(i,j) == ArithTuple(0,0,i)

    A = Layout((5,4), (E(2), E(1,3)))
    for i in range(size[0](A)):
      for j in range(size[1](A)):
        assert A(i,j) == ArithTuple(0,(0,0,0,j),i)

    A = Layout((4,(4,2)), (E(1),(E(0),4*E(1))))
    for i in range(size[0](A)):
      for j in range(size[1](A)):
        assert A(i,j) == ArithTuple(j%4,i+4*(j//4))


class TestImplicitZeroEquality:
  """Trailing positions extend by implicit zero in ``__eq__`` (see
  ``_atuple_eq``), so representations that agree on every explicit
  position compare equal regardless of length."""

  def test_three_constructions_match_under_eq(self):
    # Same rank-1 element, three different in-memory representations.
    a = E(0)                          # data = (1,)
    b = ArithTuple(1, 0)              # data = (1, 0)
    c = ArithTuple((1,))              # data = (1,)
    assert a == b
    assert b == c

  def test_structural_zeros_equal_int_zero(self):
    assert ArithTuple() == 0
    assert ArithTuple(0, 0, 0) == 0
    assert ArithTuple((0, 0)) == 0
    assert 0 * E(2) == 0
    assert ScaledBasis(0, [3]) == 0

  def test_int_and_atuple_have_different_ranks(self):
    # int v is rank-0; ArithTuple((v,)) is rank-1. They are not equal
    # for v != 0 (a nonzero int has no rank-1+ representation).
    assert ArithTuple((5,)) != 5
    assert 5 != ArithTuple((5,))

  def test_eq_returns_notimplemented_for_foreign_types(self):
    # Comparisons against types outside {int, ArithTuple, tuple/list}
    # return NotImplemented from __eq__ and fall through to Python's
    # default of False.
    x = ArithTuple(1, 2)
    assert not x == None
    assert not x == "foo"
    assert not x == {1: 2}
    assert not x == frozenset([1, 2])

  def test_idx2crd_is_consistent_across_constructions(self):
    assert idx2crd(E(0),             (8, 8)) == (1, 0)
    assert idx2crd(ArithTuple(1, 0), (8, 8)) == (1, 0)
    assert idx2crd(ArithTuple((1,)), (8, 8)) == (1, 0)


class TestWeaklyCongruentImplicitZero:
  """Trailing implicit zeros are admissible at any sub-shape."""

  def test_int_coarsens_anything(self):
    assert weakly_congruent(1, (8, 8))

  def test_unit_atuple_coarsens_anything(self):
    assert weakly_congruent(ArithTuple(1), (8, 8))

  def test_implicitly_extended_atuple_coarsens_compatible_profile(self):
    assert weakly_congruent(ArithTuple((1,)), (8, 8))

  def test_explicit_atuple_coarsens_matching_profile(self):
    assert weakly_congruent(ArithTuple((1, 1)), (8, 8))

  def test_atuple_rank_exceeding_profile_is_not_weakly_congruent(self):
    assert not weakly_congruent(ArithTuple((1, 1, 1)), (8, 8))

  def test_basis_at_path_coarsens_compatible_subprofile(self):
    assert weakly_congruent(E(0, 0), ((4, 5), 8))
    assert not weakly_congruent(E(0, 0), (8, 8))


class TestBasisRepr:
  """``basis_repr`` is the algebraic decomposition of an
  ``int | ArithTuple`` into ``(value, path)`` pairs;
  ``is_basis`` is a thin predicate over its length."""

  def test_basis_repr_of_int_is_rank_zero(self):
    # Every Python int v -- including 0 -- is the single rank-zero
    # scaled basis term v * E().
    assert basis_repr(0) == [(0, ())]
    assert basis_repr(7) == [(7, ())]

  def test_basis_repr_of_all_zero_atuple_matches_int_zero(self):
    # An ArithTuple whose every leaf is zero decomposes to the same
    # rank-zero term as int 0, matching ``ArithTuple(...) == 0``.
    assert basis_repr(0 * E(2)) == [(0, ())]
    assert basis_repr(ArithTuple(0)) == [(0, ())]
    assert basis_repr(ArithTuple(0, ArithTuple(0, 0))) == [(0, ())]

  def test_basis_repr_of_unit_basis(self):
    assert basis_repr(E(0)) == [(1, (0,))]
    assert basis_repr(E(1, 2)) == [(1, (1, 2))]

  def test_basis_repr_of_multi_term_sum(self):
    assert basis_repr(3*E(0) + 5*E(1)) == [(3, (0,)), (5, (1,))]

  def test_basis_repr_round_trip(self):
    """The defining identity: ``ArithTuple(x) == sum(v * E(*s) for v, s in basis_repr(x))``."""
    for x in [0, 7, E(0), 5 * E(1, 2),
              3 * E(0) + 5 * E(1),
              ArithTuple(1, 2, (3, 4))]:
      rep = basis_repr(x)
      rebuilt = sum((v * E(*s) for v, s in rep), 0)
      assert ArithTuple(x) == rebuilt


class TestIsBasis:
  """``is_basis(x)`` is true iff ``x`` is a single scaled basis vector
  ``v * E(*s)``. Every Python ``int`` counts as the rank-zero basis
  element ``v * E()`` (including ``0``); only multi-term sums fail."""

  def test_is_basis_recognizes_unit_basis_elements(self):
    assert is_basis(E(0))
    assert is_basis(E(1, 2))
    assert is_basis(5 * E(1, 2))

  def test_is_basis_recognizes_int_as_rank_zero(self):
    assert is_basis(7)
    assert basis_repr(7) == [(7, ())]
    # Zero is also a rank-zero basis element: 0 == 0 * E(). This makes
    # `proj(x, 0) == get[()](x) == x` well-defined.
    assert is_basis(0)
    assert basis_repr(0) == [(0, ())]

  def test_is_basis_recognizes_all_zero_atuple(self):
    # An all-zero ArithTuple equals int 0 and is a rank-zero basis
    # element, decomposing to the same single term as int 0.
    assert is_basis(0 * E(2))
    assert basis_repr(0 * E(2)) == [(0, ())]

  def test_is_basis_does_not_recognize_multi_term_sum(self):
    assert not is_basis(3 * E(0) + 5 * E(1))


class TestProjAndUnit:
  """``proj(x, profile)`` extracts the leaf of ``x`` at ``profile``'s
  path; ``unit(profile)`` returns the multiplicative unit of
  ``profile``'s algebra at that path, which for an ``int`` or an
  ``ArithTuple`` is the unit basis element. Both require ``profile``
  to be a single basis element and reject multi-term sums. A scalar
  whose algebra has a different identity supplies a ``_unit`` hook
  instead -- see ``test_swizzle.py`` for ``F2``."""

  def test_proj_extracts_leaf_at_basis_path(self):
    x = ArithTuple(7, (8, 9))
    assert proj(x, E()) == x               # int profile -> identity
    assert proj(x, E(0)) == 7
    assert proj(x, E(1)) == (8, 9)
    assert proj(x, E(1,0)) == 8
    assert proj(x, E(1,1)) == 9

  def test_unit_returns_unit_basis_at_profile_path(self):
    assert unit(E()) == 1                  # int profile -> int 1
    assert unit(E(0)) == E(0)
    assert unit(E(1,2)) == E(1,2)
    # ``unit`` ignores the coefficient; it only uses the path.
    assert unit(5 * E(1, 2)) == E(1, 2)

  def test_proj_and_unit_raise_on_multi_term_sum(self):
    multi = 3 * E(0) + 5 * E(1)
    with pytest.raises(TypeError):
      proj(ArithTuple(7, 8), multi)
    with pytest.raises(TypeError):
      unit(multi)


class TestArithTupleF2Leaves:
  """A leaf may be any ``StrideScalar``, not only an ``int``, so one coordinate
  axis can carry a swizzled (``F2``) offset while another stays an ordinary
  index. Every operation defers to the leaf's own algebra."""

  F0 = ScaledBasis(F2(1), (0,))           # F2-valued basis element at axis 0
  F4 = ScaledBasis(F2(4), (0,))

  def test_construction_accepts_f2_leaves(self):
    assert ArithTuple(F2(1), 0) == self.F0
    assert ArithTuple(F2(5)) == F2(5)      # a lone stride scalar passes through
    assert ArithTuple(F2(1), 3) == (F2(1), 3)

  def test_basis_repr_sees_an_f2_leaf(self):
    """``basis_repr`` yields any stride-scalar leaf. Restricting it to ``is_int``
    made an F2-valued axis collapse to the rank-zero zero term, so it silently
    reported as ``0`` through ``__str__``, ``proj``, ``unit`` and ``is_basis``."""
    assert basis_repr(self.F0) == [(F2(1), (0,))]
    assert is_basis(self.F0)
    assert str(self.F0) == "F1@0"
    assert proj(self.F0, self.F0) == F2(1)

  def test_unit_keeps_both_the_axis_and_the_leaf_algebra(self):
    assert unit(self.F4) == self.F0
    assert isinstance(proj(unit(self.F4), self.F0), F2)

  def test_addition_is_xor_per_leaf(self):
    assert self.F0 + self.F0 == 0                    # XOR is self-inverse
    assert self.F4 + self.F0 == ScaledBasis(F2(5), (0,))
    assert self.F0 + E(1) == (F2(1), 1)

  def test_negation_and_subtraction_defer_to_the_leaf(self):
    """``F2`` negation is identity, so ``-`` cannot be implemented as ``(-1) *``.
    That form would also hang: ``F2`` scales by an operand's bits, and a negative
    integer has no finite bit expansion."""
    assert -self.F0 == self.F0
    assert 0 - self.F0 == self.F0
    assert self.F4 - self.F0 == ScaledBasis(F2(5), (0,))
    assert self.F0 - self.F0 == 0

  def test_ordering_defers_to_the_leaf(self):
    assert self.F0 < self.F4
    assert self.F4 > self.F0

  def test_a_mixed_codomain_layout_evaluates(self):
    """Axis 0 is a swizzled offset, axis 1 an ordinary index."""
    A = Layout((8, 8), (self.F0, E(1)))
    assert A(2, 3) == (F2(2), 3)
    assert A(5, 7) == (F2(5), 7)

  def test_reflected_operators_bridge_the_two_types(self):
    """``F2`` yields ``NotImplemented`` for an operand it cannot handle, so Python
    falls back on ``ArithTuple``'s reflected operator instead of raising. Without
    that, an ``F2`` coordinate could not be scaled by an ``F2``-valued stride."""
    assert F2(3) * self.F0 == ScaledBasis(F2(3), (0,))
    assert self.F0 * F2(3) == ScaledBasis(F2(3), (0,))
    assert F2(0) + self.F0 == self.F0        # 0 is every algebra's additive identity

  def test_unknown_operands_still_raise(self):
    """``NotImplemented`` is deferral, not silence: Python raises once neither side
    knows what to do. A *nonzero scalar* is refused directly instead, since it is
    an operand type both understand -- the mismatch is one of rank."""
    with pytest.raises(TypeError):
      self.F0 * 'x'
    with pytest.raises(TypeError):
      self.F0 + None
    with pytest.raises(TypeError):
      E(0) * E(1)                           # scalars only
    with pytest.raises(TypeError):
      self.F0 + 7                           # rank mismatch, not an unknown type


class TestArithTupleSet:
  """``_set`` wraps a sequence of already-lifted children directly as
  ``self.data``. The assertions below check both the stored ``.data``
  (verbatim) and ``==`` (which extends trailing positions by implicit
  zero)."""

  def test_set_stores_data_verbatim_and_eq_extends_with_zeros(self):
    x = ArithTuple._set([1, 0, 0])
    assert x.data == (1, 0, 0)     # stored as supplied
    assert x == E(0)               # == extends trailing zeros

  def test_set_all_zero_data_equals_int_zero(self):
    assert ArithTuple._set([]).data == ()
    assert ArithTuple._set([0, 0]).data == (0, 0)
    assert ArithTuple._set([]) == 0
    assert ArithTuple._set([0, 0]) == 0

  def test_set_round_trips_through_constructor(self):
    assert ArithTuple._set((3, 5)).data == (3, 5)
    assert ArithTuple._set((3, 5)) == ArithTuple(3, 5)
