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


  def test_logical_product_mode(self):
    # `logical_product[mode](A, B)` reproduces that one mode of A over B and
    # leaves every other mode of A alone.
    A = Layout((3, 5), (1, 20))
    B = Layout(4, 1)

    assert logical_product[0](A, B) == make_layout([logical_product(A[0], B), A[1]])
    assert logical_product[1](A, B) == make_layout([A[0], logical_product(A[1], B)])
    assert logical_product[0](A, B) == logical_product(A, (B,))
    assert logical_product(A, B, mode=(0,)) == logical_product[0](A, B)
    assert logical_product(A, B, mode=()) == logical_product(A, B)

    # Post-conditions hold of the reproduced mode
    R = logical_product[0](A, B)
    assert rank(get[0](R)) == 2
    assert get[0](R)[0] == get[0](A)
    assert compatible(B, get[0](R)[1])

    # The named mode must exist
    with pytest.raises(ValueError):
      logical_product[2](A, B)


  def test_logical_product_types(self):
    # Behavior of logical_product across argument types
    # {None, int, tuple, Layout, Tiler, Tensor}.
    A = Layout((6,4),(4,1))

    # -- A promoted from int / tuple / tiler
    assert logical_product(6,     Layout(2,1)) == logical_product(Layout(6,1),            Layout(2,1))
    assert logical_product((2,3), Layout(2,1)) == logical_product(tiler_to_layout((2,3)), Layout(2,1))

    # -- A of lower rank than a multi-mode tiler B: rank mismatch
    with pytest.raises(ValueError):
      logical_product(6, (Layout(2,1), Layout(2,1)))

    # -- A = None: unsupported; unlike logical_divide there is no identity to
    #    reproduce, and tiler_to_layout(None) is undefined
    with pytest.raises(TypeError):
      logical_product(None, Layout(2,1))

    # -- A = Tensor: unsupported; logical_product has no Tensor hook
    with pytest.raises(TypeError):
      logical_product(make_tensor(Layout(6,1)), Layout(2,1))

    # -- B promotions: None is a no-op; int promotes to N:1
    assert logical_product(A, None) == A
    assert logical_product(A, 2) == logical_product(A, Layout(2,1))


  def test_logical_product_arg_matrix(self):
    # logical_product over the full {None, int, tuple, Layout, tuple-of-Layouts}
    # matrix of A and B. A tuple B is by-mode, so its rank must suit A; a tuple A
    # is promoted to a single Layout over basis strides *before* B is applied,
    # which is what makes the by-mode results carry E(0)/E(1).
    cases = [
      # A                             B                             expected
      (None,                          None,                         TypeError),
      (None,                          2,                            TypeError),
      (None,                          (2, 2),                       TypeError),
      (None,                          Layout(2, 1),                 TypeError),
      (None,                          (Layout(2, 1), Layout(2, 2)),  TypeError),

      (24,                            None,                         Layout(24, 1)),
      (24,                            2,                            Layout((24, 2), (1, 24))),
      (24,                            (2, 2),                       ValueError),
      (24,                            Layout(2, 1),                 Layout((24, 2), (1, 24))),
      (24,                            (Layout(2, 1), Layout(2, 2)),  ValueError),

      ((6, 4),                        None,                         Layout((6, 4), (E(0), E(1)))),
      ((6, 4),                        2,                            Layout(((6, 4), 2), ((E(0), E(1)), 4*E(1)))),
      ((6, 4),                        (2, 2),                       Layout(((6, 2), (4, 2)), ((E(0), 6*E(0)), (E(1), 4*E(1))))),
      ((6, 4),                        Layout(2, 1),                 Layout(((6, 4), 2), ((E(0), E(1)), 4*E(1)))),
      ((6, 4),                        (Layout(2, 1), Layout(2, 2)),  Layout(((6, 2), (4, 2)), ((E(0), 6*E(0)), (E(1), 8*E(1))))),

      (Layout((6, 4), (4, 1)),        None,                         Layout((6, 4), (4, 1))),
      (Layout((6, 4), (4, 1)),        2,                            Layout(((6, 4), 2), ((4, 1), 24))),
      (Layout((6, 4), (4, 1)),        (2, 2),                       Layout(((6, 2), (4, 2)), ((4, 1), (1, 4)))),
      (Layout((6, 4), (4, 1)),        Layout(2, 1),                 Layout(((6, 4), 2), ((4, 1), 24))),
      (Layout((6, 4), (4, 1)),        (Layout(2, 1), Layout(2, 2)),  Layout(((6, 2), (4, 2)), ((4, 1), (1, 8)))),

      ((Layout(6, 2), Layout(4, 1)),  None,                         Layout((6, 4), (2*E(0), E(1)))),
      ((Layout(6, 2), Layout(4, 1)),  2,                            Layout(((6, 4), 2), ((2*E(0), E(1)), E(0)))),
      ((Layout(6, 2), Layout(4, 1)),  (2, 2),                       Layout(((6, 2), (4, 2)), ((2*E(0), E(0)), (E(1), 4*E(1))))),
      ((Layout(6, 2), Layout(4, 1)),  Layout(2, 1),                 Layout(((6, 4), 2), ((2*E(0), E(1)), E(0)))),
      ((Layout(6, 2), Layout(4, 1)),  (Layout(2, 1), Layout(2, 2)),  Layout(((6, 2), (4, 2)), ((2*E(0), E(0)), (E(1), 8*E(1))))),
    ]
    for A, B, expected in cases:
      if isinstance(expected, type):
        with pytest.raises(expected):
          logical_product(A, B)
      else:
        assert logical_product(A, B) == expected, f"logical_product({A}, {B})"


  def test_logical_product_promotes_tiler_A(self):
    # A non-Layout A is promoted with tiler_to_layout before any of B's by-mode
    # structure is applied, so promoting it by hand must give the same result.
    tilers = [6, 24, (6, 4), (2, 3, 4), (Layout(6, 2), Layout(4, 1)), (2, Layout(4, 1))]
    Bs     = [None, 2, (2, 2), Layout(2, 1), (Layout(2, 1), Layout(2, 2)), (Layout(2, 1), None)]
    for A in tilers:
      for B in Bs:
        if rank(A) < rank(B):
          continue
        assert logical_product(A, B) == logical_product(tiler_to_layout(A), B), \
               f"logical_product({A}, {B})"


  def test_logical_product_short_and_padded_tiler(self):
    # A tuple B shorter than A leaves A's trailing modes untouched, and a None
    # entry leaves its own mode untouched -- both spellings of "no copies here".
    for A in [Layout((6, 4), (4, 1)), (6, 4), (Layout(6, 2), Layout(4, 1))]:
      L = tiler_to_layout(A)
      assert logical_product(A, (Layout(2, 1),)) == make_layout([logical_product(L[0], Layout(2, 1)), L[1]])
      assert logical_product(A, (Layout(2, 1), None)) == logical_product(A, (Layout(2, 1),))
      assert logical_product(A, (None, Layout(2, 1))) == make_layout([L[0], logical_product(L[1], Layout(2, 1))])
      assert logical_product(A, (None, None)) == L

    # B may not out-rank A
    with pytest.raises(ValueError):
      logical_product(Layout((6, 4), (4, 1)), (Layout(2, 1), Layout(2, 1), Layout(2, 1)))


  def test_logical_product_is_complement_composition(self):
    # logical_product is (A, A* o B): it must agree with that construction
    # spelled out, for a single-Layout B on a Layout A.
    cases = [
      (Layout(3, 1),                        Layout(4, 1)),
      (Layout((2, 2), (4, 1)),              Layout(6, 1)),
      (Layout((6, 4), (4, 1)),              Layout(2, 1)),
      (Layout(3, 32),                       Layout((8, 8), (8, 1))),
      (Layout(((2, 2), (2, 2)), ((1, 4), (8, 32))), Layout((2, 2), (2, 1))),
    ]
    for A, B in cases:
      assert logical_product(A, B) == make_layout([A, composition(complement(A), B)]), \
             f"logical_product({A}, {B})"
