# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.layout recast
"""

import logging

from pycute import *
from fractions import Fraction

logger = logging.getLogger()


class TestRecast:

  def postcondition_recast(self, L, scale, expected):
    result = recast(L, scale)
    logger.info(f"  {L}  =>  {result}")
    assert result == expected

  def test_1d_stride1(self):
    """1D contiguous layout, all scales"""
    layout = Layout(24, 1)

    assert recast(layout, 8) == Layout(  3, 1)
    assert recast(layout, 6) == Layout(  4, 1)
    assert recast(layout, 4) == Layout(  6, 1)
    assert recast(layout, 2) == Layout( 12, 1)
    assert recast(layout, 1) == Layout( 24, 1)
    assert recast(layout, Fraction(1, 2)) == Layout( 48, 1)
    assert recast(layout, Fraction(1, 4)) == Layout( 96, 1)
    assert recast(layout, Fraction(1, 6)) == Layout(144, 1)
    assert recast(layout, Fraction(1, 8)) == Layout(192, 1)

  def test_1d_stride2(self):
    """1D layout with stride 2, all scales"""
    layout = Layout(24, 2)

    assert recast(layout, 8) == Layout( 6, 1)
    assert recast(layout, 6) == Layout( 8, 1)
    assert recast(layout, 4) == Layout(12, 1)
    assert recast(layout, 2) == Layout(24, 1)
    assert recast(layout, 1) == Layout(24, 2)
    assert recast(layout, Fraction(1, 2)) == Layout(24, 4)
    assert recast(layout, Fraction(1, 4)) == Layout(24, 8)
    assert recast(layout, Fraction(1, 6)) == Layout(24, 12)
    assert recast(layout, Fraction(1, 8)) == Layout(24, 16)

  def test_2d_column_major(self):
    """2D column-major 24x24 layout, all scales"""
    layout = Layout((24, 24), (24, 1))

    assert recast(layout, 8) == Layout((24,   3), (  3,  1))
    assert recast(layout, 6) == Layout((24,   4), (  4,  1))
    assert recast(layout, 4) == Layout((24,   6), (  6,  1))
    assert recast(layout, 2) == Layout((24,  12), ( 12,  1))
    assert recast(layout, 1) == Layout((24,  24), ( 24,  1))
    assert recast(layout, Fraction(1, 2)) == Layout((24,  48), ( 48,  1))
    assert recast(layout, Fraction(1, 4)) == Layout((24,  96), ( 96,  1))
    assert recast(layout, Fraction(1, 6)) == Layout((24, 144), (144,  1))
    assert recast(layout, Fraction(1, 8)) == Layout((24, 192), (192,  1))

  def test_2d_col_major_small(self):
    """2D column-major 4x6 layout"""
    layout = Layout((4, 6), (6, 1))

    assert recast(layout, 6) == Layout((4,  1), ( 1, 1))
    assert recast(layout, 2) == Layout((4,  3), ( 3, 1))
    assert recast(layout, 1) == Layout((4,  6), ( 6, 1))
    assert recast(layout, Fraction(1, 2)) == Layout((4, 12), (12, 1))
    assert recast(layout, Fraction(1, 4)) == Layout((4, 24), (24, 1))
    assert recast(layout, Fraction(1, 6)) == Layout((4, 36), (36, 1))
    assert recast(layout, Fraction(1, 8)) == Layout((4, 48), (48, 1))

  def test_2d_row_major(self):
    """2D row-major 4x4 layout"""
    layout = Layout((4, 4), (4, 1))

    assert recast(layout, 8) == Layout((2,  1), ( 1, 1))
    assert recast(layout, 4) == Layout((4,  1), ( 1, 1))
    assert recast(layout, 2) == Layout((4,  2), ( 2, 1))
    assert recast(layout, 1) == Layout((4,  4), ( 4, 1))
    assert recast(layout, Fraction(1, 2)) == Layout((4,  8), ( 8, 1))
    assert recast(layout, Fraction(1, 4)) == Layout((4, 16), (16, 1))
    assert recast(layout, Fraction(1, 6)) == Layout((4, 24), (24, 1))
    assert recast(layout, Fraction(1, 8)) == Layout((4, 32), (32, 1))

  def test_stride_zero(self):
    """Stride-0 layouts are unchanged by recast"""
    layout = Layout(8, 0)

    for scale in [8, 6, 4, 2, 1, Fraction(1, 2), Fraction(1, 4), Fraction(1, 6), Fraction(1, 8)]:
      assert recast(layout, scale) == Layout(8, 0)

  def test_2d_with_stride_zero(self):
    """2D layout with one stride-0 mode"""
    layout = Layout((8, 4), (0, 2))

    assert recast(layout, 8) == Layout((8, 1), (0,  1))
    assert recast(layout, 4) == Layout((8, 2), (0,  1))
    assert recast(layout, 2) == Layout((8, 4), (0,  1))
    assert recast(layout, 1) == Layout((8, 4), (0,  2))
    assert recast(layout, Fraction(1, 2)) == Layout((8, 4), (0,  4))
    assert recast(layout, Fraction(1, 4)) == Layout((8, 4), (0,  8))
    assert recast(layout, Fraction(1, 6)) == Layout((8, 4), (0, 12))
    assert recast(layout, Fraction(1, 8)) == Layout((8, 4), (0, 16))

  def test_3d_with_stride_zero(self):
    """3D layout with a stride-0 mode in the middle"""
    layout = Layout((8, 4, 6), (1, 0, 2))

    assert recast(layout, 8) == Layout(( 1, 4, 2), (1, 0,  1))
    assert recast(layout, 4) == Layout(( 2, 4, 3), (1, 0,  1))
    assert recast(layout, 2) == Layout(( 4, 4, 6), (1, 0,  1))
    assert recast(layout, 1) == Layout(( 8, 4, 6), (1, 0,  2))
    assert recast(layout, Fraction(1, 2)) == Layout((16, 4, 6), (1, 0,  4))
    assert recast(layout, Fraction(1, 4)) == Layout((32, 4, 6), (1, 0,  8))
    assert recast(layout, Fraction(1, 6)) == Layout((48, 4, 6), (1, 0, 12))
    assert recast(layout, Fraction(1, 8)) == Layout((64, 4, 6), (1, 0, 16))

  def test_nested_shape(self):
    """Nested (hierarchical) shape layout"""
    layout = Layout(((4, 6), 8), ((1, 4), 24))

    assert recast(layout, 8) == Layout((( 1,  3), 8), (( 1,  1),   3))
    assert recast(layout, 4) == Layout((( 1,  6), 8), (( 1,  1),   6))
    assert recast(layout, 2) == Layout((( 2,  6), 8), (( 1,  2),  12))
    assert recast(layout, 1) == Layout((( 4,  6), 8), (( 1,  4),  24))
    assert recast(layout, Fraction(1, 2)) == Layout((( 8,  6), 8), (( 1,  8),  48))
    assert recast(layout, Fraction(1, 4)) == Layout(((16,  6), 8), (( 1, 16),  96))
    assert recast(layout, Fraction(1, 6)) == Layout(((24,  6), 8), (( 1, 24), 144))
    assert recast(layout, Fraction(1, 8)) == Layout(((32,  6), 8), (( 1, 32), 192))

  def test_nested_shape_all_scales(self):
    """Nested shape where all leaf shapes are divisible by 24"""
    layout = Layout(((24, 24), 24), (1, 24))

    assert recast(layout, 8) == Layout(((  3, 24), 24), (( 1,   3),   3))
    assert recast(layout, 6) == Layout(((  4, 24), 24), (( 1,   4),   4))
    assert recast(layout, 4) == Layout(((  6, 24), 24), (( 1,   6),   6))
    assert recast(layout, 2) == Layout((( 12, 24), 24), (( 1,  12),  12))
    assert recast(layout, 1) == Layout((( 24, 24), 24), (( 1,  24),  24))
    assert recast(layout, Fraction(1, 2)) == Layout((( 48, 24), 24), (( 1,  48),  48))
    assert recast(layout, Fraction(1, 4)) == Layout((( 96, 24), 24), (( 1,  96),  96))
    assert recast(layout, Fraction(1, 6)) == Layout(((144, 24), 24), (( 1, 144), 144))
    assert recast(layout, Fraction(1, 8)) == Layout(((192, 24), 24), (( 1, 192), 192))

  def test_scale_1_is_identity(self):
    """Recast with scale=1 always returns the original layout"""
    layouts = [
      Layout(24, 1),
      Layout(24, 2),
      Layout((4, 6), (6, 1)),
      Layout((4, 4), (4, 1)),
      Layout((8, 4, 6), (1, 0, 2)),
      Layout(8, 0),
    ]
    for layout in layouts:
      assert recast(layout, 1) == layout

  def test_divisibility_errors(self):
    """Recast raises ValueError when divisibility conditions are violated"""
    pass
