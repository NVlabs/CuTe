# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.logical_divide
"""

import logging
import pytest

from pycute import *

logger = logging.getLogger()

class TestLogicalDivide:

  def postcondition_zipped_divide(self, A, B):
    # The (Tile, Grid) post-conditions below hold for *zipped_divide* for any
    # tiler B. logical_divide only satisfies them when B is a single Layout.
    R     = zipped_divide(A, B)
    tiler = tiler_to_layout(B)

    logger.info(f"  {A} / {B}  =>  {R}")

    # Check the rank (Tile,Grid)
    assert rank(R) == 2

    # Check the Tile-mode is just composition with the (flattened) tiler
    assert compatible(tiler, R[0])
    for i in range(size(tiler)):
      assert R(i,0) == A(tiler(i))

    # Check the Grid-mode
    assert weakly_congruent(R[1], coprofile(tiler))
    # Every element of A appears in R as well
    for i in range(size(A)):
      assert any(R(j) == A(i) for j in range(size(R)))


  def test_zipped_divide(self):
    self.postcondition_zipped_divide(Layout(1,0), Layout(1,0))
    self.postcondition_zipped_divide(Layout(1,0), Layout(1,1))
    self.postcondition_zipped_divide(Layout(1,1), Layout(1,0))
    self.postcondition_zipped_divide(Layout(1,1), Layout(1,1))
    self.postcondition_zipped_divide(Layout(6,1), Layout(2,1))
    self.postcondition_zipped_divide(Layout(6,1), Layout(2,3))
    self.postcondition_zipped_divide(Layout(6,1), Layout((2,3),(3,1)))
    self.postcondition_zipped_divide(Layout(6,2), Layout(2,1))
    self.postcondition_zipped_divide(Layout(6,2), Layout(2,3))
    self.postcondition_zipped_divide(Layout(6,2), Layout((2,3),(3,1)))
    self.postcondition_zipped_divide(Layout((6,6),(1,12)), Layout((6,3),(3,1)))
    self.postcondition_zipped_divide(Layout((6,6),(12,1)), Layout((6,3),(3,1)))
    self.postcondition_zipped_divide(Layout(32,1), Layout(2,8))
    self.postcondition_zipped_divide(Layout((4,1),(1,1)), Layout(2,1))
    self.postcondition_zipped_divide(Layout((4,1),(1,1)), Layout(2,2))
    self.postcondition_zipped_divide(Layout((8,8),(1,8)), Layout(32,2))
    self.postcondition_zipped_divide(Layout((8,8),(8,1)), Layout(32,2))


  def test_zipped_divide_coord(self):
    self.postcondition_zipped_divide(Layout((8,8),(9,1)), Layout(4,E(0)))
    self.postcondition_zipped_divide(Layout((8,8),(9,1)), Layout(4,E(1)))
    self.postcondition_zipped_divide(Layout((8,8),(9,1)), Layout(4,2*E(0)))
    self.postcondition_zipped_divide(Layout((8,8),(9,1)), Layout(4,2*E(1)))
    self.postcondition_zipped_divide(Layout((8,8),(9,1)), Layout((4,4),(E(1),2*E(0))))
    self.postcondition_zipped_divide(Layout((8,8),(9,1)), Layout((5,7),(3*E(1),2*E(0))))


  def test_zipped_divide_associativity(self):

    # MMA SM70 8x8x4 Associativity Example
    tiler_mn   = (8,8)
    clayout_tv = Layout(((2,2,2),(2,2,2)), ((1,16,4),(8,2,32)))

    data_layout = Layout((32,64))
    c_mma_0 = composition(zipped_divide(data_layout, tiler_mn), (clayout_tv, None))
    c_mma_1 = zipped_divide(data_layout, composition(tiler_mn, clayout_tv))
    assert c_mma_0 == c_mma_1

    data_layout = Layout((32,64), (64,1))
    c_mma_0 = composition(zipped_divide(data_layout, tiler_mn), (clayout_tv, None))
    c_mma_1 = zipped_divide(data_layout, composition(tiler_mn, clayout_tv))
    assert c_mma_0 == c_mma_1


  def test_logical_divide_tiler(self):
    # A multi-mode tiler (tuple) makes logical_divide act *by-mode*, which is
    # distinct from the (Tile, Grid) regrouping produced by zipped_divide.
    cases = [
      (Layout((6,4),(4,1)), (Layout(2,1), Layout(2,1))),
      (Layout((8,8),(8,1)), (Layout(2,1), Layout(4,1))),
    ]
    for layoutA, tiler in cases:
      # zipped_divide always meets the (Tile, Grid) post-conditions
      self.postcondition_zipped_divide(layoutA, tiler)
      # logical_divide is exactly the by-mode application
      expected = make_layout([logical_divide(layoutA[i], t) for i, t in enumerate(tiler)])
      assert logical_divide(layoutA, tiler) == expected
      # and the two operations genuinely differ on a multi-mode tiler
      assert logical_divide(layoutA, tiler) != zipped_divide(layoutA, tiler)


  def test_logical_divide_types(self):
    # Behavior of logical_divide / zipped_divide across argument types
    # {None, int, tuple, Layout, Tiler, Tensor}.
    A = Layout((6,4),(4,1))

    # -- A promoted from int / tuple / tiler
    assert logical_divide(6,     Layout(2,1)) == logical_divide(Layout(6,1),            Layout(2,1))
    assert logical_divide((6,4), Layout(2,1)) == logical_divide(tiler_to_layout((6,4)), Layout(2,1))

    # -- A = None: acts as identity
    assert logical_divide(None, Layout(2,1)) == Layout((2, 1), (1, 2))
    assert logical_divide(None, (3,5)) == Layout(((3, 1), (5, 1)), ((E(0), 3*E(0)), (E(1), 5*E(1))))
    assert zipped_divide(None, Layout(2,1)) == Layout((2, 1), (1, 2))
    assert zipped_divide(None, (3,5)) == Layout(((3, 5), (1, 1)), ((E(0), E(1)), (3*E(0), 5*E(1))))

    # -- B promotions: None is a no-op; int promotes to N:1
    assert logical_divide(A, None) == A
    assert logical_divide(A, 2) == logical_divide(A, Layout(2,1))

    # -- A = Tensor: supported
    T = make_tensor(A)
    assert logical_divide(T, Layout(2,1)).layout == logical_divide(A, Layout(2,1))

    # -- B = Tensor: nonsensical tiler
    with pytest.raises(TypeError):
      logical_divide(A, make_tensor(Layout(2,1)))
