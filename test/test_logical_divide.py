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


  def test_logical_divide_mode(self):
    # `logical_divide[mode](A, B)` divides that one mode of A by B and leaves
    # every other mode of A alone.
    A = Layout(((3, 8), 5), ((1, 3), 24))
    B = Layout(4, 2)

    assert logical_divide[0](A, B) == make_layout([logical_divide(A[0], B), A[1]])
    assert logical_divide[1](A, B) == make_layout([A[0], logical_divide(A[1], B)])
    assert logical_divide[0, 1](A, B) == make_layout(
      [make_layout([A[0][0], logical_divide(A[0][1], B)]), A[1]])

    # An empty mode is the whole layout, and the spellings of a mode agree
    assert logical_divide(A, B, mode=()) == logical_divide(A, B)
    assert logical_divide(A, B, mode=(0, 1)) == logical_divide[0, 1](A, B)
    assert logical_divide[0][1](A, B) == logical_divide[0, 1](A, B)
    assert logical_divide[0](A, B, mode=1) == logical_divide[0, 1](A, B)

    # A mode of a Tensor divides its layout
    T = make_tensor(A)
    assert logical_divide[1](T, B).layout == logical_divide[1](A, B)

    # The named mode must exist
    with pytest.raises(ValueError):
      logical_divide[2](A, B)


  def test_zipped_divide_mode(self):
    # `zipped_divide[mode](A, B)` zips the tiler into that one mode of A. The
    # tiler's rank must suit the named mode, exactly as it must suit A itself.
    A     = Layout(((9, 32), 7))
    tiler = (Layout(3, 3), Layout((2, 4), (1, 8)))

    assert zipped_divide[0](A, tiler) == make_layout([zipped_divide(A[0], tiler), A[1]])
    assert zipped_divide[0](A, tiler) == logical_divide[0](A, tiler_to_layout(tiler))
    assert zipped_divide(A, tiler, mode=(0,)) == zipped_divide[0](A, tiler)
    assert zipped_divide(A, tiler, mode=()) == zipped_divide(A, tiler)

    # A single-Layout tiler reaches a depth-0 mode
    assert zipped_divide[0, 1](A, Layout(4, 2)) == make_layout(
      [make_layout([A[0][0], zipped_divide(A[0][1], Layout(4, 2))]), A[1]])

    # Post-conditions hold of the divided mode
    R = zipped_divide[0](A, tiler)
    assert rank(get[0](R)) == 2
    assert compatible(tiler, get[0](R)[0])


  def test_logical_divide_types(self):
    # Behavior of logical_divide / zipped_divide across argument types
    # {None, int, tuple, Layout, Tiler, Tensor}.
    A = Layout((6,4),(4,1))

    # -- A promoted from int / tuple / tiler
    assert logical_divide(6,     Layout(2,1)) == logical_divide(Layout(6,1),            Layout(2,1))
    assert logical_divide((6,4), Layout(2,1)) == logical_divide(tiler_to_layout((6,4)), Layout(2,1))

    # -- A = None: the identity of unknown extents, so the grid takes B's *free*
    #    complement -- and dividing nothing by nothing is nothing.
    assert logical_divide(None, Layout(2,1)) == Layout((2, 1), (1, 2))
    assert logical_divide(None, (3,5)) == Layout(((3, 1), (5, 1)), ((E(0), 3*E(0)), (E(1), 5*E(1))))
    assert logical_divide(None, None) is None
    assert logical_divide(None, (Layout(2,1), None)) == Layout(((2, 1), (1, 1)),
                                                              ((E(0), 2*E(0)), (E(1), E(1))))
    assert zipped_divide(None, Layout(2,1)) == Layout((2, 1), (1, 2))
    assert zipped_divide(None, (3,5)) == Layout(((3, 5), (1, 1)), ((E(0), E(1)), (3*E(0), 5*E(1))))

    # -- Only the extent A would have supplied is unknown, so the grid keeps
    #    every mode B alone determines: the same divide, with the extent of A's
    #    own mode standing at 1.
    assert logical_divide(None,        Layout(4,2)) == Layout((4, (2, 1)), (2, (1, 8)))
    assert logical_divide(Layout(24),  Layout(4,2)) == Layout((4, (2, 3)), (2, (1, 8)))

    # -- A = None has no modes to select, so a mode places the result instead of
    #    selecting into A, and the tiler's own modes keep their basis paths.
    assert logical_divide[0](None, (3,5)) == make_layout([logical_divide(None, (3,5))])
    assert logical_divide[1](None, Layout(2,1)) == make_layout([Layout(1, 0),
                                                               logical_divide(None, Layout(2,1))])

    # -- B promotions: None is a no-op; int promotes to N:1
    assert logical_divide(A, None) == A
    assert logical_divide(A, 2) == logical_divide(A, Layout(2,1))

    # -- A = Tensor: supported
    T = make_tensor(A)
    assert logical_divide(T, Layout(2,1)).layout == logical_divide(A, Layout(2,1))

    # -- B = Tensor: nonsensical tiler
    with pytest.raises(TypeError):
      logical_divide(A, make_tensor(Layout(2,1)))


  def test_logical_divide_arg_matrix(self):
    # logical_divide over the full {None, int, tuple, Layout, tuple-of-Layouts}
    # matrix of A and B. A tuple B is by-mode, so its rank must suit A; a tuple A
    # is promoted to a single Layout over basis strides *before* B is applied,
    # which is what makes the by-mode results carry E(0)/E(1).
    cases = [
      # A                             B                             expected
      (None,                          None,                         None),
      (None,                          2,                            Layout((2, 1), (1, 2))),
      (None,                          (2, 2),                       Layout(((2, 1), (2, 1)), ((E(0), 2*E(0)), (E(1), 2*E(1))))),
      (None,                          Layout(2, 1),                 Layout((2, 1), (1, 2))),
      (None,                          (Layout(2, 1), Layout(2, 2)),  Layout(((2, 1), (2, (2, 1))), ((E(0), 2*E(0)), (2*E(1), (E(1), 4*E(1)))))),

      (24,                            None,                         Layout(24, 1)),
      (24,                            2,                            Layout((2, 12), (1, 2))),
      (24,                            (2, 2),                       ValueError),
      (24,                            Layout(2, 1),                 Layout((2, 12), (1, 2))),
      (24,                            (Layout(2, 1), Layout(2, 2)),  ValueError),

      ((6, 4),                        None,                         Layout((6, 4), (E(0), E(1)))),
      ((6, 4),                        2,                            Layout((2, (3, 4)), (E(0), (2*E(0), E(1))))),
      ((6, 4),                        (2, 2),                       Layout(((2, 3), (2, 2)), ((E(0), 2*E(0)), (E(1), 2*E(1))))),
      ((6, 4),                        Layout(2, 1),                 Layout((2, (3, 4)), (E(0), (2*E(0), E(1))))),
      ((6, 4),                        (Layout(2, 1), Layout(2, 2)),  Layout(((2, 3), (2, 2)), ((E(0), 2*E(0)), (2*E(1), E(1))))),

      (Layout((6, 4), (4, 1)),        None,                         Layout((6, 4), (4, 1))),
      (Layout((6, 4), (4, 1)),        2,                            Layout((2, (3, 4)), (4, (8, 1)))),
      (Layout((6, 4), (4, 1)),        (2, 2),                       Layout(((2, 3), (2, 2)), ((4, 8), (1, 2)))),
      (Layout((6, 4), (4, 1)),        Layout(2, 1),                 Layout((2, (3, 4)), (4, (8, 1)))),
      (Layout((6, 4), (4, 1)),        (Layout(2, 1), Layout(2, 2)),  Layout(((2, 3), (2, 2)), ((4, 8), (2, 1)))),

      ((Layout(6, 2), Layout(4, 1)),  None,                         Layout((6, 4), (2*E(0), E(1)))),
      ((Layout(6, 2), Layout(4, 1)),  2,                            Layout((2, (3, 4)), (2*E(0), (4*E(0), E(1))))),
      ((Layout(6, 2), Layout(4, 1)),  (2, 2),                       Layout(((2, 3), (2, 2)), ((2*E(0), 4*E(0)), (E(1), 2*E(1))))),
      ((Layout(6, 2), Layout(4, 1)),  Layout(2, 1),                 Layout((2, (3, 4)), (2*E(0), (4*E(0), E(1))))),
      ((Layout(6, 2), Layout(4, 1)),  (Layout(2, 1), Layout(2, 2)),  Layout(((2, 3), (2, 2)), ((2*E(0), 4*E(0)), (2*E(1), E(1))))),
    ]
    for A, B, expected in cases:
      if isinstance(expected, type):
        with pytest.raises(expected):
          logical_divide(A, B)
      else:
        assert logical_divide(A, B) == expected, f"logical_divide({A}, {B})"


  def test_logical_divide_promotes_tiler_A(self):
    # A non-Layout A is promoted with tiler_to_layout before any of B's by-mode
    # structure is applied, so promoting it by hand must give the same result.
    tilers = [6, 24, (6, 4), (2, 3, 4), (Layout(6, 2), Layout(4, 1)), (2, Layout(4, 1))]
    Bs     = [None, 2, (2, 2), Layout(2, 1), (Layout(2, 1), Layout(2, 2)), (Layout(2, 1), None)]
    for A in tilers:
      for B in Bs:
        if rank(A) < rank(B):
          continue
        assert logical_divide(A, B) == logical_divide(tiler_to_layout(A), B), \
               f"logical_divide({A}, {B})"


  def test_logical_divide_short_and_padded_tiler(self):
    # B rewrites rather than selects -- every element of A is in the result, just
    # refactored -- so a tuple B may run short: the modes it does not reach are
    # simply not refactored. A None entry says the same of its own mode.
    for A in [Layout((6, 4), (4, 1)), (6, 4), (Layout(6, 2), Layout(4, 1))]:
      L = tiler_to_layout(A)
      assert logical_divide(A, (Layout(2, 1),)) == make_layout([logical_divide(L[0], Layout(2, 1)), L[1]])
      assert logical_divide(A, (Layout(2, 1), None)) == logical_divide(A, (Layout(2, 1),))
      assert logical_divide(A, (None, Layout(2, 1))) == make_layout([L[0], logical_divide(L[1], Layout(2, 1))])
      assert logical_divide(A, (None, None)) == L
      assert logical_divide(A, ()) == L                  # divides nothing

      # Rewriting preserves every element of A, however short the tiler
      for B in [(Layout(2, 1),), (Layout(2, 1), None), (None, None), ()]:
        assert size(logical_divide(A, B)) == size(L)

    # B may not out-rank A
    with pytest.raises(ValueError):
      logical_divide(Layout((6, 4), (4, 1)), (Layout(2, 1), Layout(2, 1), Layout(2, 1)))


  def test_logical_divide_is_composition_with_complement(self):
    # logical_divide is A o (B, B*): it must agree with that composition spelled
    # out, for a single-Layout B on a Layout A.
    cases = [
      (Layout(24, 1),            Layout(4, 2)),
      (Layout((6, 4), (4, 1)),   Layout(2, 1)),
      (Layout((8, 8), (8, 1)),   Layout(32, 2)),
      (Layout((6, 6), (1, 12)),  Layout((6, 3), (3, 1))),
      (Layout((8, 8), (9, 1)),   Layout(4, E(1))),
    ]
    for A, B in cases:
      assert logical_divide(A, B) == composition(A, make_layout([B, complement(B, extend=shape(A))])), \
             f"logical_divide({A}, {B})"
