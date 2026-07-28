# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.shape: congruent, weakly_congruent, compatible.

These tests are also worked examples for docs/02_shape_stride.md.

References:
* Whitepaper, §2.1 (congruent, weakly_congruent on HTuple)
* Whitepaper, §2.2.1 (compatible on Shape)
"""

import pytest

from pycute import *


class TestCongruent:
  """`congruent(a, b)` is an equivalence relation: same hierarchical profile."""

  def test_flat_shapes_with_different_values(self):
    assert congruent((4, 8), (5, 7))
    assert congruent(31, 42)
    assert (congruent(((4, 6), (3, (2, 2), 8)),
                              ((1, 1), (1, (1, 1), 1))))

  def test_different_profiles(self):
    assert not congruent((4, 8), (4, (2, 4)))
    assert not congruent(31, (4, 8))
    assert not congruent((1, 1, 1), (1, 1))

  def test_congruent_is_reflexive(self):
    for x in [7, (3, 4), ((1, 2), 3), ((1, (2, 3)), 4)]:
      assert congruent(x, x)


class TestWeaklyCongruent:
  """`weakly_congruent(a, b)` is a partial order: `a` coarsens `b`."""

  def test_int_coarsens_anything(self):
    assert weakly_congruent(30, (3, 4))
    assert weakly_congruent(30, ((3, 4), 5))
    assert weakly_congruent(30, (((3, 4), 5), (6, 7)))

  def test_tuple_does_not_coarsen_int(self):
    assert not weakly_congruent((3, 4), 30)

  def test_tuples_must_have_same_top_level_rank(self):
    assert not weakly_congruent((1, 2, 3), (1, 2))

  def test_recursive_coarsening(self):
    assert weakly_congruent((3, 4), (5, (6, 7)))
    assert not weakly_congruent((3, (4, 5)), (5, 6))

  def test_a_shape_is_weakly_congruent_with_itself(self):
    """Reflexivity: every shape is weakly congruent with itself."""
    for x in [7, (3, 4), ((1, 2), 3), ((1, (2, 3)), 4)]:
      assert weakly_congruent(x, x)


class TestCompatible:
  """`compatible(a, b)` is a partial order on shapes: `a` coarsens `b` and
  `|a| == |b|`. The Whitepaper's running examples are checked here."""

  def test_examples_from_whitepaper(self):
    # 30 ≼ (2, 15) ≼ (2, (3, 5))
    assert compatible(30, (2, 15))
    assert compatible((2, 15), (2, (3, 5)))
    assert compatible(30, (2, (3, 5)))

    # 30 ≼ (6, 5) ≼ ((3, 2), 5)
    assert compatible(30, (6, 5))
    assert compatible((6, 5), ((3, 2), 5))
    assert compatible(30, ((3, 2), 5))

    # (2, (3, 5)) and ((3, 2), 5) are NOT compatible despite same size
    assert not compatible((2, (3, 5)), ((3, 2), 5))
    assert not compatible(((3, 2), 5), (2, (3, 5)))

  def test_size_must_match(self):
    assert not compatible(24, 32)
    assert not compatible(24, (4, 8))            # |a| = 24, |b| = 32

  def test_int_compatible_with_any_shape_of_same_size(self):
    assert compatible(24, (4, 6))
    assert compatible(24, ((2, 2), 6))
    assert compatible(24, ((2, 2), (3, 2)))
    assert compatible(24, ((2, 3), 4))

  def test_singleton_tuple_is_not_an_int(self):
    """Compatible distinguishes between an `int` and a `(int,)` 1-tuple."""
    assert compatible(24, (24,))      # int ≼ (int,)
    assert not compatible((24,), 24)     # but not the reverse
    assert not compatible((24,), (4, 6)) # 1-tuple has different rank

  def test_compatible_is_reflexive(self):
    for x in [7, (3, 4), ((1, 2), 3), ((1, (2, 3)), 4)]:
      assert compatible(x, x)


class TestCompatibilityImpliesWeakCongruence:
  """If `a` is compatible with `b`, then `a` is also weakly congruent
  with `b`. The converse is not true (sizes need not match)."""

  def test_compatible_implies_weakly_congruent(self):
    pairs = [(24, (4, 6)),
             (24, ((2, 2), (3, 2))),
             ((4, 6), ((2, 2), 6)),
             ((6, 5), ((3, 2), 5))]
    for a, b in pairs:
      assert compatible(a, b)
      assert weakly_congruent(a, b)

  def test_weakly_congruent_does_not_imply_compatible(self):
    # 30 is weakly congruent with (3, 4) but not compatible (sizes differ)
    assert weakly_congruent(30, (3, 4))
    assert not compatible(30, (3, 4))


class TestCommonRefinement:
  """`common_refinement(a, b)` -- the join (least upper bound) of two shapes
  under the compatibility partial order. The result `c` is the minimal
  shape such that `a ≼ c` and `b ≼ c`."""

  def assertRefines(self, a, b):
    """Assert that `b` refines `a` (equivalently, `a ≼ b`)."""
    assert compatible(a, b), f"expected {a} ≼ {b}"

  def assertCommonRefinement(self, a, b, expected):
    c = common_refinement(a, b)
    assert c == expected
    self.assertRefines(a, c)
    self.assertRefines(b, c)
    # symmetry of the join
    assert common_refinement(b, a) == expected

  # -- existence cases --------------------------------------------------

  def test_int_int_equal(self):
    self.assertCommonRefinement(1, 1, 1)
    self.assertCommonRefinement(30, 30, 30)

  def test_int_tuple_size_match(self):
    self.assertCommonRefinement(10, (2, 5),     (2, 5))
    self.assertCommonRefinement(30, (2, (3, 5)),(2, (3, 5)))
    self.assertCommonRefinement(24, ((2,2), 6), ((2,2), 6))
    # int ≼ (int,)  (singleton tuple is more refined than int)
    self.assertCommonRefinement(10, (10,), (10,))

  def test_reflexive(self):
    for x in [7, (3, 4), ((1, 2), 3), ((1, (2, 3)), 4), (2, (3, 5))]:
      assert common_refinement(x, x) == x

  def test_examples_from_whitepaper(self):
    # The compatibility chain  30 ≼ (2, 15) ≼ (2, (3, 5))
    self.assertCommonRefinement(30, (2, 15),         (2, 15))
    self.assertCommonRefinement((2, 15), (2, (3, 5)), (2, (3, 5)))
    self.assertCommonRefinement(30, (2, (3, 5)),     (2, (3, 5)))

    # The compatibility chain  30 ≼ (6, 5) ≼ ((3, 2), 5)
    self.assertCommonRefinement(30, (6, 5),         (6, 5))
    self.assertCommonRefinement((6, 5), ((3, 2), 5),((3, 2), 5))

  def test_mixed_modes(self):
    # Same rank, both modes need to be refined separately.
    self.assertCommonRefinement((4, (3, 5)), ((2, 2), 15),
                                ((2, 2), (3, 5)))
    # Deeper nesting
    self.assertCommonRefinement((2, (6, 5)), (2, ((3, 2), 5)),
                                (2, ((3, 2), 5)))

  def test_join_picks_most_refined_on_each_mode(self):
    a = ((2, 3), 20)
    b = (6, (4, 5))
    c = common_refinement(a, b)
    assert c == ((2, 3), (4, 5))
    self.assertRefines(a, c)
    self.assertRefines(b, c)

  # -- accepts objects with a .shape attribute --------------------------

  def test_accepts_layouts(self):
    """Layouts have a `.shape` attribute; common_refinement should
    accept them by extracting the shape via `shape(...)`."""
    L1 = Layout((2, 15))
    L2 = Layout((2, (3, 5)))
    assert common_refinement(L1, L2) == (2, (3, 5))
    assert common_refinement(L1, (2, (3, 5))) == (2, (3, 5))
    assert common_refinement(30, L1) == (2, 15)

  # -- non-existence cases ----------------------------------------------

  def test_unequal_ints_raise(self):
    with pytest.raises(ValueError):
      common_refinement(3, 4)

  def test_int_tuple_size_mismatch_raises(self):
    with pytest.raises(ValueError):
      common_refinement(7, (2, 3))
    with pytest.raises(ValueError):
      common_refinement((2, 3), 7)

  def test_rank_mismatch_raises(self):
    with pytest.raises(ValueError):
      common_refinement((2, 3), (2, 3, 1))
    with pytest.raises(ValueError):
      common_refinement((6,), (2, 3))

  def test_incompatible_profiles_same_size_raise(self):
    # From Whitepaper §2.2.1: (2,(3,5)) and ((3,2),5) have the same
    # size (30) but no common refinement -- their only common compatible
    # shape is the coarsening 30.
    with pytest.raises(ValueError):
      common_refinement((2, (3, 5)), ((3, 2), 5))
    with pytest.raises(ValueError):
      common_refinement(((3, 2), 5), (2, (3, 5)))

  def test_incompatible_mode_raises(self):
    # First mode: 2 vs 3 disagrees as ints
    with pytest.raises(ValueError):
      common_refinement((2, 15), (3, 10))
    # Second mode: (3,5) vs (2,5) have different first leaf
    with pytest.raises(ValueError):
      common_refinement((4, (3, 5)), (4, (2, 5)))


class TestCommonCoarsening:
  """`common_coarsening(a, b)` -- the meet (greatest lower bound) of two shapes
  under the compatibility partial order. The result `c` is the maximal shape
  such that `c ≼ a` and `c ≼ b`. The meet exists iff `size(a) == size(b)`."""

  def assertCoarsens(self, c, x):
    """Assert that `c` coarsens `x` (equivalently, `c ≼ x`)."""
    assert compatible(c, x), f"expected {c} ≼ {x}"

  def assertCommonCoarsening(self, a, b, expected):
    c = common_coarsening(a, b)
    assert c == expected
    self.assertCoarsens(c, a)
    self.assertCoarsens(c, b)
    # symmetry of the meet
    assert common_coarsening(b, a) == expected

  # -- existence cases --------------------------------------------------

  def test_int_int_equal(self):
    self.assertCommonCoarsening(1, 1, 1)
    self.assertCommonCoarsening(30, 30, 30)

  def test_int_forces_int_meet(self):
    # If either side is an int, the meet is also forced to be that int.
    self.assertCommonCoarsening(10, (2, 5),      10)
    self.assertCommonCoarsening(30, (2, (3, 5)), 30)
    self.assertCommonCoarsening(24, ((2, 2), 6), 24)
    # (10,) is strictly more refined than 10, so the meet is the int 10.
    self.assertCommonCoarsening(10, (10,), 10)

  def test_reflexive(self):
    for x in [7, (3, 4), ((1, 2), 3), ((1, (2, 3)), 4), (2, (3, 5))]:
      assert common_coarsening(x, x) == x

  def test_examples_from_whitepaper(self):
    # The compatibility chain  30 ≼ (2, 15) ≼ (2, (3, 5))
    # meets pick the *coarser* of the two.
    self.assertCommonCoarsening((2, 15), (2, (3, 5)), (2, 15))
    self.assertCommonCoarsening(30, (2, (3, 5)),      30)
    # Whitepaper's classic example: incompatible profiles with equal size 30.
    self.assertCommonCoarsening((2, (3, 5)), ((3, 2), 5), 30)
    self.assertCommonCoarsening(((3, 2), 5), (2, (3, 5)), 30)

  def test_mode_mismatch_falls_back_to_int(self):
    # Same rank, but mode 0 mismatches as integers -> fall back to size.
    self.assertCommonCoarsening((6, 5), (2, 15), 30)
    self.assertCommonCoarsening((6, 5), (3, 10), 30)
    # Same rank but mode-0 sizes don't match -> fall back to top-level size.
    self.assertCommonCoarsening((4, 6), (3, 8), 24)

  def test_partial_tuple_meet(self):
    # Both rank-2; each mode meets succeed.
    self.assertCommonCoarsening((4, (3, 5)), ((2, 2), 15),
                                (4, 15))
    self.assertCommonCoarsening(((2, 3), 4), ((3, 2), 4),
                                (6, 4))            # mode 0 falls back, mode 1 succeeds

  def test_rank_mismatch_falls_back_to_int(self):
    # Different top-level rank -> tuple meet impossible -> integer meet.
    self.assertCommonCoarsening((2, 3),    (2, 3, 1),  6)
    self.assertCommonCoarsening((6,),      6,         6)
    self.assertCommonCoarsening((5, 3, 4), (10, 6),   60)

  # -- accepts objects with a .shape attribute --------------------------

  def test_accepts_layouts(self):
    L1 = Layout((2, 15))
    L2 = Layout((2, (3, 5)))
    assert common_coarsening(L1, L2) == (2, 15)
    assert common_coarsening(L1, (2, (3, 5))) == (2, 15)
    assert common_coarsening(30, L1) == 30

  # -- non-existence cases (only when sizes differ) ---------------------

  def test_unequal_ints_raise(self):
    with pytest.raises(ValueError):
      common_coarsening(3, 4)

  def test_int_tuple_size_mismatch_raises(self):
    with pytest.raises(ValueError):
      common_coarsening(7, (2, 3))
    with pytest.raises(ValueError):
      common_coarsening((2, 3), 7)

  def test_total_size_mismatch_raises(self):
    with pytest.raises(ValueError):
      common_coarsening((2, 3), (4, 5))         # 6 != 20
    with pytest.raises(ValueError):
      common_coarsening((2, 3, 4), (6, 5))      # 24 != 30 (rank diff also)


class TestRefinementAndCoarseningDuality:
  """Sanity-check the lattice properties relating the join (refinement)
  and the meet (coarsening) of two shapes."""

  def test_meet_is_below_refinement_when_both_exist(self):
    pairs = [(30, (2, 15)),
             ((2, 15), (2, (3, 5))),
             ((4, (3, 5)), ((2, 2), 15)),
             (((2, 3), 20), (6, (4, 5)))]
    for a, b in pairs:
      r = common_refinement(a, b)
      c = common_coarsening(a, b)
      assert compatible(c, r), f"expected meet {c} ≼ join {r} for {a}, {b}"
      assert compatible(c, a), f"{c} ≼ {a}"
      assert compatible(c, b), f"{c} ≼ {b}"
      assert compatible(a, r), f"{a} ≼ {r}"
      assert compatible(b, r), f"{b} ≼ {r}"

  def test_meet_exists_when_refinement_does(self):
    # If `common_refinement(a, b)` exists then `size(a) == size(b)`, so
    # `common_coarsening(a, b)` also exists.
    pairs = [(30, (2, 15)),
             ((2, 15), (2, (3, 5))),
             ((4, (3, 5)), ((2, 2), 15))]
    for a, b in pairs:
      common_refinement(a, b)                   # no raise
      common_coarsening(a, b)                   # no raise

  def test_meet_can_exist_when_refinement_does_not(self):
    # The Whitepaper's incompatible-but-same-size example.
    with pytest.raises(ValueError):
      common_refinement((2, (3, 5)), ((3, 2), 5))
    assert common_coarsening((2, (3, 5)), ((3, 2), 5)) == 30
