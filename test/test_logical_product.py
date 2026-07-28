# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.logical_product
"""

import logging
import pytest

from pycute import *

logger = logging.getLogger()

class TestLogicalProduct:
  def postcondition_logical_product(self, A, B):
    R = logical_product(A, B)

    logger.info(f"  {A} x {B}  =>  {R}")

    assert rank(R) == 2

    assert A == R[0]

    assert compatible(B, R[1])


  def test_logical_product(self):
    self.postcondition_logical_product(Layout(1,0), Layout(1,0))
    self.postcondition_logical_product(Layout(1,1), Layout(1,0))
    self.postcondition_logical_product(Layout(1,0), Layout(1,1))
    self.postcondition_logical_product(Layout(1,1), Layout(1,1))
    self.postcondition_logical_product(Layout(3,1), Layout(4,1))
    self.postcondition_logical_product(Layout(3,1), Layout(4,0))
    self.postcondition_logical_product(Layout(3,1), Layout((2,4),(1,2)))
    self.postcondition_logical_product(Layout((2,4),(1,2)), Layout(3,1))
    self.postcondition_logical_product(Layout((8,(2,2))), Layout(4,2))
    self.postcondition_logical_product(Layout((2,2)), Layout((3,3), (3,1)))
    self.postcondition_logical_product(Layout(3, 32), Layout(32, 1))
    self.postcondition_logical_product(Layout(3, 2), Layout(4, 1))
    self.postcondition_logical_product(Layout(3, 32), Layout(128, 1))
    self.postcondition_logical_product(Layout(3, 32), Layout((8,8)))
    self.postcondition_logical_product(Layout(3, 32), Layout((8,8), (8,1)))
    self.postcondition_logical_product(Layout(((4,2)), ((1,16))), Layout((4,4)))
    self.postcondition_logical_product(Layout(((4,2)), ((1,16))), Layout((4,2), (2,1)))
    self.postcondition_logical_product(Layout(((2,2),(2,2)),((1,4),(8,32))), Layout((2,2), (1,2)))
    self.postcondition_logical_product(Layout(((2,2),(2,2)),((1,4),(8,32))), Layout((2,2),(2,1)))
    self.postcondition_logical_product(Layout((4,6),(1,6)), Layout(3,1))


  def test_logical_product_tiler(self):
    # A multi-mode tiler (tuple) makes logical_product act by-mode.
    A     = Layout((6,4),(4,1))
    tiler = (Layout(2,1), Layout(2,1))
    expected = make_layout([logical_product(A[i], t) for i, t in enumerate(tiler)])
    assert logical_product(A, tiler) == expected


  def test_logical_product_types(self):
    # Behavior of logical_product across argument types
    # {None, int, tuple, Layout, Tiler, Tensor}. This encodes the CURRENT
    # behavior; comments mark where it differs from commit 15df9e1.
    A = Layout((6,4),(4,1))

    # -- A promoted from int / tuple / tiler: CHANGED. 15df9e1 raised TypeError
    #    for any non-Layout A; A is now promoted via tiler_to_layout(A), making
    #    logical_product symmetric with logical_divide. --
    assert logical_product(6,     Layout(2,1)) == logical_product(Layout(6,1),            Layout(2,1))
    assert logical_product((2,3), Layout(2,1)) == logical_product(tiler_to_layout((2,3)), Layout(2,1))

    # -- A promoted to too-low rank for a multi-mode tiler B now reaches the
    #    rank check (ValueError) rather than the old TypeError. --
    with pytest.raises(ValueError):
      logical_product(6, (Layout(2,1), Layout(2,1)))

    # # -- A = None: unchanged (TypeError; tiler_to_layout(None) is undefined) --
    with pytest.raises(TypeError):
      logical_product(None, Layout(2,1))

    # -- A = Tensor: unchanged (TypeError; logical_product has no _layout_op hook) --
    with pytest.raises(TypeError):
      logical_product(make_tensor(Layout(6,1)), Layout(2,1))

    # -- B promotions: None is a no-op; int promotes to N:1 (unchanged) --
    assert logical_product(A, None) == A
    assert logical_product(A, 2) == logical_product(A, Layout(2,1))
