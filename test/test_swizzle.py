# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.swizzle

These tests are also worked examples for docs/06_swizzle.md.
"""

import itertools
import logging
import pytest
import sympy

from pycute import *

logger = logging.getLogger()


class TestF2:
  """`F2` is the integer-module (Z, XOR, *)."""

  def test_f2_addition_is_xor(self):
    assert int(F2(0b1010) + F2(0b1100)) == 0b0110
    assert int(F2(0) + F2(0)) == 0
    assert int(F2(7) + F2(7)) == 0        # XOR self-inverse

  def test_f2_negation_is_identity(self):
    """XOR is its own inverse, so `-a == a` and `a - b == a + b`."""
    assert -F2(5) == F2(5)
    assert F2(5) - F2(3) == F2(5) + F2(3)
    assert F2(5) - F2(5) == 0

  def test_f2_multiplication_rejects_a_negative_operand(self):
    """An operand acts through its bits, and a negative integer has no finite bit
    expansion -- shifting it right never reaches zero, so the product would spin
    while its accumulator grew without bound."""
    with pytest.raises(ValueError):
      F2(3) * -1
    with pytest.raises(ValueError):
      -2 * F2(3)

  def test_f2_self_inverse(self):
    """Addition in F2 is its own inverse: a + a = 0."""
    for a in [0, 1, 5, 17, 255, 1024]:
      assert int(F2(a) + F2(a)) == 0

  def test_f2_multiplication_is_carry_less(self):
    """`F2`'s `*` is a carry-less (polynomial) product, in both its `F2 * int`
    and `F2 * F2` forms -- an integer operand acts through its bits, not its
    value. It coincides with the integer product exactly where the schoolbook
    product carries nowhere, which a power-of-two operand always does."""
    assert int(F2(5) * 3) == 15 == 5 * 3            # carry-free: agrees with Z
    assert int(3 * F2(5)) == 15
    assert int(F2(7) * 0) == 0

    assert int(F2(3) * 3) == 0b101 != 3 * 3         # carries: 0b11 * 0b11 = 0b101
    assert int(F2(3) * F2(3)) == 0b101              # `F2 * F2` is the same product
    assert int(F2(0b1010) * 2) == 0b10100           # power of two: a plain shift

  def test_f2_construction_is_idempotent(self):
    """`F2(F2(a)) == F2(a)`, so wrapping a value that may already be an `F2` is
    always safe and never nests."""
    assert F2(F2(5)) == F2(5)
    assert int(F2(F2(F2(5)))) == 5

  def test_f2_ordering_is_by_value(self):
    """`F2`s order by their underlying value, i.e. by leading bit with the
    lower-order terms breaking ties -- the `F2` reading of stride magnitude, and
    what the layout algebra's stride sorts want."""
    assert sorted([F2(9), F2(1), F2(4), F2(8)]) == [F2(1), F2(4), F2(8), F2(9)]
    assert F2(1) < F2(8) and F2(8) > F2(1)
    assert F2(4) <= F2(4) and F2(4) >= F2(4)
    assert F2(1) < 4 and 0 < F2(1)              # comparable with ints, either way round

  def test_f2_is_stride_scalar(self):
    """`F2` is registered as a `StrideScalar` and so can appear in a `Layout`."""
    assert is_stride_scalar(F2(5))
    assert not is_int(F2(5))

  def test_f2_is_static_follows_the_value(self):
    """A concrete `F2` is statically known, so the algebra's stride sorts can
    order it; a symbolic one is not."""
    assert is_static(F2(9))
    assert not is_static(F2(sympy.symbols("n", positive=True, integer=True)))

  def test_unit_of_an_f2_is_f2(self):
    """`unit` drops a stride's magnitude but keeps its algebra, so `unit(d) * n`
    rebuilds a stride of magnitude `n` in the same algebra. For `F2` that has to
    be `F2(1)`: scaling `int 1` would leave `Z` and multiply with carries."""
    assert unit(F2(9)) == F2(1)
    assert unit(F2(0)) == F2(1)                 # magnitude is irrelevant
    assert isinstance(unit(F2(9)), F2)          # ... and so is the value: the *type* matters
    assert unit(F2(9)) * 0b11 == F2(0b11)
    assert unit(3) * 0b11 == 0b11               # unchanged for Z


class TestF2Divmod:
  """`divmod(F2, int)` is Euclidean division in `F2`: the unique `(q, r)` with
     `a == q * b + r`  and  `deg(r) < deg(b)`
  where an `F2` value's bits are the coefficients of a polynomial over the
  two-element field. Since `F2`'s `*` is carry-less, this is polynomial long
  division, not integer division."""

  DIVIDENDS = range(64)
  DIVISORS  = range(1, 20)

  def test_divmod_identity(self):
    """`a == q * b + r`, for every dividend and divisor."""
    for a in self.DIVIDENDS:
      for b in self.DIVISORS:
        q, r = divmod(F2(a), b)
        assert q * b + r == F2(a)

  def test_remainder_is_an_in_bounds_coordinate(self):
    """`deg(r) < deg(b)` implies `r < b`, so a remainder is always a valid
    coordinate of an extent-`b` mode."""
    for a in self.DIVIDENDS:
      for b in self.DIVISORS:
        _, r = divmod(F2(a), b)
        assert int(r).bit_length() < b.bit_length()
        assert int(r) < b

  def test_both_results_are_f2(self):
    """The quotient stays in `F2` so that the `divmod` chain in `idx2crd` and
    the recomposition in `crd2idx` are both carry-less."""
    q, r = divmod(F2(0b10110), 4)
    assert isinstance(q, F2) and isinstance(r, F2)

  def test_power_of_two_divisor_is_a_bit_split(self):
    """For `b == 2**k` the quotient and remainder are `a`'s high and low
    bit-fields -- the case that makes `F2` layout modes bit-disjoint."""
    for a in self.DIVIDENDS:
      for k in range(6):
        assert divmod(F2(a), 1 << k) == (F2(a >> k), F2(a & ((1 << k) - 1)))

  def test_power_of_two_quotient_is_an_ordinary_shift(self):
    """For `b == 2**k` the quotient's *value* is the plain integer `a >> k`, and
    the identity also closes with an integer `q * b`: only the quotient's type
    has to stay in `F2`, not its arithmetic."""
    for a in self.DIVIDENDS:
      for k in range(6):
        q, r = divmod(F2(a), 1 << k)
        assert int(q) == a >> k
        assert F2(int(q) * (1 << k)) + r == F2(a)

  def test_non_power_of_two_divisor_is_carry_less(self):
    """`0b1011 == 0b110 * 0b11 + 0b1` carry-lessly, while `11 == 3*3 + 2` in Z."""
    assert divmod(F2(0b1011), 0b11) == (F2(0b110), F2(0b1))
    assert divmod(11, 3) == (3, 2)

  def test_divmod_by_one(self):
    assert divmod(F2(42), 1) == (F2(42), F2(0))

  def test_divmod_by_zero(self):
    with pytest.raises(ZeroDivisionError):
      divmod(F2(42), 0)

  def test_divmod_by_non_scalar(self):
    with pytest.raises(TypeError):
      divmod(F2(42), (2, 4))


class TestF2Idx2Crd:
  """`idx2crd` decomposes an `F2` value into the natural coordinates of a shape
  by chaining `divmod(F2, int)` in colexicographical order. This is what lets a
  value drawn from an `F2` layout's codomain be fed back in as a coordinate."""

  POW2_SHAPES = [(4, 8), (2, 4, 4), (2, (4, 4)), (8, 8), ((2, 2), (4, 2))]

  def test_rank_one_is_identity(self):
    assert idx2crd(F2(22), 32) == F2(22)

  def test_bit_split(self):
    """Power-of-two extents split an `F2` value into disjoint bit-fields."""
    assert idx2crd(F2(0b10110), (4, 8))    == (F2(0b10), F2(0b101))
    assert idx2crd(F2(0b10110), (2, 4, 4)) == (F2(0b0), F2(0b11), F2(0b10))

  def test_hierarchical_shape(self):
    assert idx2crd(F2(0b10110), (2, (4, 4))) == (F2(0b0), (F2(0b11), F2(0b10)))
    assert congruent(idx2crd(F2(0b10110), (2, (4, 4))), (2, (4, 4)))

  def test_last_leaf_absorbs_out_of_bounds(self):
    """As for an integer index, the final mode keeps the full quotient."""
    assert idx2crd(F2(0b110110), (4, 8)) == (F2(0b10), F2(0b1101))

  def test_agrees_with_the_integer_decomposition(self):
    """On power-of-two extents the carry-less split *is* the integer split, so
    an `F2` index and the equal integer index give the same coordinate."""
    for S in self.POW2_SHAPES:
      for i in range(size(S)):
        assert idx2crd(F2(i), S) == idx2crd(i, S)

  def test_crd2idx_inverts_idx2crd(self):
    """`crd2idx(idx2crd(a, S), S) == a`, in and out of bounds."""
    for S in self.POW2_SHAPES:
      for i in range(2 * size(S)):
        assert crd2idx(idx2crd(F2(i), S), S) == F2(i)

  def test_carrying_extents_are_rejected(self):
    """`idx2crd` telescopes the extents carry-lessly while `crd2idx` recomposes
    with prefix products taken in Z, so the two invert each other exactly when
    the colex prefix products agree in both. Where they do not, the shape is
    rejected rather than decomposed into a coordinate that will not recompose.

    `(3, 3, 3)` is the smallest such shape: `crd2idx` would weight the last mode
    by `3 * 3 == 9`, but the `divmod` chain peels it off by `F2(3) * 3 == F2(5)`."""
    assert int(F2(3) * 3) == 0b101 != 3 * 3

    with pytest.raises(ValueError):
      idx2crd(F2(4), (3, 3, 3))
    with pytest.raises(ValueError):
      Layout((3, 3, 3), (F2(1), F2(3), F2(9)))(F2(4))

  def test_extents_need_not_be_powers_of_two(self):
    """The precondition is on the prefix *products*, not the extents. Rank-2 is
    always admissible -- the single divisor is the only weight in play -- and a
    carry-free mix of extents such as `(3, 4, 3)` is too."""
    for S in [(3, 3), (5, 5), (3, 7), (6, 6), (3, 4, 3), (4, (3, 8))]:
      for i in range(size(S)):
        assert crd2idx(idx2crd(F2(i), S), S) == F2(i)

  def test_admissible_shapes_are_exactly_the_carry_free_ones(self):
    """Every rank-3 shape is either admitted *and* round-trips, or rejected --
    the guard is neither stricter nor looser than the carry-free condition."""
    for S in itertools.product(range(1, 7), repeat=3):
      pps, carry_free = 1, True
      for s in S[:-1]:
        carry_free = carry_free and F2(pps) * s == pps * s
        pps *= s
      if carry_free:
        for i in range(size(S)):
          assert crd2idx(idx2crd(F2(i), S), S) == F2(i)
      else:
        with pytest.raises(ValueError):
          idx2crd(F2(0), S)


class TestSwizzle:
  """`Swizzle(bits, base, shift)` is the function
       offset → offset ^ ((offset & yyy_mask) >> shift)
  where yyy_mask = ((1<<bits) - 1) << (base + max(0, shift))."""

  def test_swizzle_zero_bits_is_identity(self):
    sw = Swizzle(0, 0, 0)
    for i in range(32):
      assert sw(i) == i

  def test_swizzle_xor_pattern(self):
    """Swizzle(2, 0, 2): yyy is bits [3:2], zzz is bits [1:0]."""
    sw = Swizzle(2, 0, 2)
    expected = [0, 1, 2, 3,
                5, 4, 7, 6,
                10, 11, 8, 9,
                15, 14, 13, 12]
    assert [sw(i) for i in range(16)] == expected

  def test_swizzle_is_an_involution(self):
    """`sw(sw(i)) == i` for every `i` (XOR is self-inverse)."""
    for sw in [Swizzle(0, 0, 0), Swizzle(1, 0, 1), Swizzle(2, 0, 2),
               Swizzle(2, 3, 3), Swizzle(3, 0, 3)]:
      for i in range(64):
        assert sw(sw(i)) == i

  def test_swizzle_constructor_validation(self):
    with pytest.raises(ValueError):
      Swizzle(-1, 0, 0)
    with pytest.raises(ValueError):
      Swizzle(0, -1, 0)
    # Negative shift must be at least as wide as the bit field
    with pytest.raises(ValueError):
      Swizzle(2, 0, -1)


class TestF2Layout:
  """A layout with `F2` strides evaluates `inner_product` over (Z, XOR, *)."""

  def test_f2_layout_evaluation(self):
    """A rank-2 F2 layout XORs its mode contributions."""
    # L((c0, c1)) = F2(1)*c0 + F2(8)*c1
    #            = F2(c0) + F2(8 c1)
    #            = F2(c0 XOR 8 c1)
    L = Layout((4, 8), (F2(1), F2(8)))
    for c0 in range(4):
      for c1 in range(8):
        assert int(L(c0, c1)) == c0 ^ (8 * c1)

  def test_f2_layout_self_inverse(self):
    """Repeating the same offset XOR's it back to zero."""
    L = Layout((4, 4), (F2(1), F2(4)))
    for c0 in range(4):
      for c1 in range(4):
        x = L(c0, c1)
        assert int(x + x) == 0


class TestF2LayoutOnF2Coordinates:
  """An `F2` layout maps `Z(S)` into `F2`, so composing it with itself -- or with
  its inverse -- requires evaluating a layout *at* an `F2` value. `idx2crd`
  decomposes that value into natural coordinates and `inner_product` then
  applies the strides, exactly as for an integer index."""

  def test_compact_f2_layout_is_the_identity_on_f2(self):
    """`Layout((4, 8), (F2(1), F2(4)))` is the identity `Z(4,8) -> F2`, so
    feeding its own output back in reproduces it."""
    L = Layout((4, 8), (F2(1), F2(4)))
    for i in range(32):
      assert L(F2(i)) == F2(i)

  def test_f2_index_agrees_with_integer_index(self):
    """`L(F2(i)) == L(i)`: on power-of-two extents an `F2` index decomposes into
    the same coordinate as the equal integer index."""
    for L in [Layout((8, 8), (F2(1), F2(9))),
              Layout((8, 8), (F2(9), F2(1))),
              Layout((4, (4, 3)), (F2(1), (F2(5), F2(16))))]:
      for i in range(size(L)):
        assert L(F2(i)) == L(i)

  def test_swizzle_layout_is_its_own_right_inverse(self):
    """`Layout((8, 8), (F2(1), F2(9)))` XOR's the row index into the column
    index, an involution on `F2`, so `L(L(i)) == i`. This is the post-condition
    `right_inverse` has to reproduce for `F2` strides."""
    L = Layout((8, 8), (F2(1), F2(9)))
    for i in range(64):
      assert L(L(i)) == i

  def test_not_every_swizzle_is_an_involution(self):
    """Transposing the same strides gives a bijection that is *not* its own
    inverse, so a genuine `right_inverse` is needed."""
    L = Layout((8, 8), (F2(9), F2(1)))
    assert sorted(int(L(i)) for i in range(64)) == list(range(64))
    assert any(L(L(i)) != i for i in range(64))
