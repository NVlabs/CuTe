# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.blocked_product and pycute.raked_product

These tests are also worked examples for docs/04_layout_algebra.md.
"""

import logging
import pytest

from pycute import *

logger = logging.getLogger()


class TestBlockedProduct:
  """`blocked_product` is a rank-sensitive variant of `logical_product`
  that pairs the i-th mode of `tile` with the i-th mode of `grid` and
  places the tile mode first (so each tile is contiguous in memory)."""

  def postcondition_blocked_product(self, tile, grid):
    R = blocked_product(tile, grid)
    logger.info(f"  {tile} (x) {grid}  =>  {R}")

    # Result has the same rank as tile and grid
    assert rank(R) == rank(tile)
    assert rank(R) == rank(grid)

    # Total size is the product of tile and grid sizes
    assert size(R) == size(tile) * size(grid)
    return R

  def test_blocked_2d(self):
    """A 2x5 tile arranged in a 3x4 grid produces a 6x20 layout."""
    tile = Layout((2, 5), (5, 1))           # row-major 2x5 tile
    grid = Layout((3, 4), (1, 3))           # column-major 3x4 grid
    R = self.postcondition_blocked_product(tile, grid)
    assert size[0](R) == 6         # M = 2 * 3
    assert size[1](R) == 20        # N = 5 * 4

  def test_blocked_post_conditions(self):
    """For every coordinate (m, n), blocked_product produces a unique offset
    within the bounding box of the tile-of-tiles."""
    tile = Layout((2, 3))
    grid = Layout((4, 5))
    R = self.postcondition_blocked_product(tile, grid)
    # All offsets are distinct
    seen = set()
    for i in range(size(R)):
      v = R(i)
      assert v not in seen
      seen.add(v)

  def test_blocked_rank_mismatch_raises(self):
    """`blocked_product` requires `tile` and `grid` to have the same rank."""
    with pytest.raises(ValueError):
      blocked_product(Layout(4), Layout((2, 3)))


class TestRakedProduct:
  """`raked_product` is a rank-sensitive variant of `logical_product` that
  places the grid mode first ("cyclic distribution"). The same tile and
  grid produce a different visual arrangement than `blocked_product`."""

  def postcondition_raked_product(self, tile, grid):
    R = raked_product(tile, grid)
    logger.info(f"  {tile} <x> {grid}  =>  {R}")

    assert rank(R) == rank(tile)
    assert rank(R) == rank(grid)
    assert size(R) == size(tile) * size(grid)
    return R

  def test_raked_2d(self):
    """A 2x5 tile raked into a 3x4 grid still produces a 6x20 layout."""
    tile = Layout((2, 5), (5, 1))
    grid = Layout((3, 4), (1, 3))
    R = self.postcondition_raked_product(tile, grid)
    assert size[0](R) == 6
    assert size[1](R) == 20

  def test_raked_vs_blocked_differ_in_mode_order(self):
    """`blocked_product` puts the tile mode first; `raked_product` puts
    the grid mode first. The two layouts contain the same offsets but
    visit them in different orders."""
    tile = Layout((2, 2))
    grid = Layout((3, 3))
    BP = blocked_product(tile, grid)
    RP = raked_product(tile, grid)

    # Same total size and same image (set of offsets)
    assert size(BP) == size(RP)
    assert {BP(i) for i in range(size(BP))} == {RP(i) for i in range(size(RP))}

    # But the iteration order differs (the layouts are not equal)
    assert BP != RP

  def test_raked_rank_mismatch_raises(self):
    with pytest.raises(ValueError):
      raked_product(Layout(4), Layout((2, 3)))
