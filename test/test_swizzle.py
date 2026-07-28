# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.swizzle

These tests are also worked examples for docs/06_swizzle.md.
"""

import logging
import pytest

from pycute import *

logger = logging.getLogger()


class TestF2:
  """`F2` is the integer-module (Z, XOR, *)."""

  def test_f2_addition_is_xor(self):
    assert int(F2(0b1010) + F2(0b1100)) == 0b0110
    assert int(F2(0) + F2(0)) == 0
    assert int(F2(7) + F2(7)) == 0        # XOR self-inverse

  def test_f2_self_inverse(self):
    """Addition in F2 is its own inverse: a + a = 0."""
    for a in [0, 1, 5, 17, 255, 1024]:
      assert int(F2(a) + F2(a)) == 0

  def test_f2_scalar_multiplication_is_integer(self):
    """`F2.__mul__(int)` is ordinary integer multiplication."""
    assert int(F2(5) * 3) == 15
    assert int(3 * F2(5)) == 15
    assert int(F2(7) * 0) == 0

  def test_f2_is_stride_scalar(self):
    """`F2` is registered as a `StrideScalar` and so can appear in a `Layout`."""
    assert is_stride_scalar(F2(5))
    assert not is_int(F2(5))


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
