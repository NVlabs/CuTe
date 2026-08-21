# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.composition
"""

import logging
import pytest
import sympy

from pycute import *

logger = logging.getLogger()


class TestComposition:
  def postcondition_composition(self, A, B):
    R = composition(A, B)

    logger.info(f"  {A} o {B}  =>  {R}")

    # Post-condition: R is compatible with B
    assert compatible(B, R)

    # Post-condition: R(c) = A(B(c)) for all coordinates c in B
    for i in range(size(R)):
      assert R(i) == A(B(i))

  def test_composition_1(self):
    # Test all combinations with shapes/strides < 4
    for AD in range(0,4):
      for BD in range(0,4):
        for AS in range(1,4):
          for BS in range(1,4):
            self.postcondition_composition(Layout(AS,AD), Layout(BS,BD))

  def test_composition_2(self):
    self.postcondition_composition(Layout(12), Layout((4,3)))
    self.postcondition_composition(Layout(12, 2), Layout((4,3)))
    self.postcondition_composition(Layout(12), Layout((4,3), (3,1)))
    self.postcondition_composition(Layout(12, 2), Layout((4,3), (3,1)))
    self.postcondition_composition(Layout(12), Layout((2,3), (2,4)))
    self.postcondition_composition(Layout((4,3)), Layout((4,3)))
    self.postcondition_composition(Layout((4,3)), Layout(12))
    self.postcondition_composition(Layout((4,3)), Layout(6, 2))
    self.postcondition_composition(Layout((4,3)), Layout((6,2), (2,1)))
    self.postcondition_composition(Layout((4,3), (3,1)), Layout((4,3)))
    self.postcondition_composition(Layout((4,3), (3,1)), Layout(12))
    self.postcondition_composition(Layout((4,3), (3,1)), Layout(6, 2))
    self.postcondition_composition(Layout((4,3), (3,1)), Layout((6,2), (2,1)))
    self.postcondition_composition(Layout((8,8)), Layout(((2,2,2), (2,2,2)),((1,16,4), (8,2,32))))
    self.postcondition_composition(Layout((8,8), (8,1)), Layout(((2,2,2), (2,2,2)),((1,16,4), (8,2,32))))
    self.postcondition_composition(Layout(((2,2,2), (2,2,2)),((1,16,4), (8,2,32))), Layout(8, 4))
    self.postcondition_composition(Layout(((4,2)), ((1,16))), Layout((4,2), (2,1)))
    self.postcondition_composition(Layout((2,2), (2,1)), Layout((2,2), (2,1)))
    self.postcondition_composition(Layout((4,8,2)), Layout((2,2,2), (2,8,1)))
    self.postcondition_composition(Layout((4,8,2), (2,8,1)), Layout((2,2,2), (1,8,2)))
    self.postcondition_composition(Layout((4,8,2), (2,8,1)), Layout((4,2,2), (2,8,1)))

    # Pre-coalesced LHS
    self.postcondition_composition(Layout((4,6,8), (1,4,7)), Layout(6, 1))

    # Mid-layout truncation
    self.postcondition_composition(Layout((4,6,8,10), (2,3,5,7)), Layout(6,12))
    self.postcondition_composition(Layout((5,126,7), (1,13,0)), Layout(21, 30))
    self.postcondition_composition(Layout((23,5), (2,120)), Layout(7, 3))

    # Over the end
    self.postcondition_composition(Layout((4,6,1), (2,3,0)), Layout(30, 4))
    self.postcondition_composition(Layout((4,6,1), (1,4,0)), Layout((6,8)))

    # Other
    self.postcondition_composition(Layout((5,5), (5,5)), Layout(5, 5))
    self.postcondition_composition(Layout((5,3), (7,1)), Layout(2, 3))
    assert composition(Layout((5,3), (7,1)), Layout(2, 3)) == Layout(2, 21)
    self.postcondition_composition(Layout((5,5), (22,5)), Layout(3, 2))
    assert composition(Layout((5,5), (22,5)), Layout(3, 2)) == Layout(3, 44)
    self.postcondition_composition(Layout((5,5,5), (1,22,5)), Layout(3, 10))
    assert composition(Layout((5,5,5), (1,22,5)), Layout(3, 10)) == Layout(3, 44)
    self.postcondition_composition(Layout(7, 11), Layout(3, 4))
    self.postcondition_composition(Layout(7, 11), Layout((3,5), (6,3)))

  def test_composition_mode(self):
    # `composition[mode](A, B)` composes that one mode of A with B and leaves
    # every other mode of A alone.
    A = Layout(((6, 2), 5), ((8, 2), 1))
    B = Layout((4, 3), (3, 1))

    assert composition[0](A, B) == make_layout([composition(A[0], B), A[1]])
    # The mode-op is a tiler naming every mode of A, so the modes it does not
    # name are kept by an explicit `None` rather than by being left off.
    assert composition[0](A, B) == composition(A, (B, None))
    assert composition[0][0](A, Layout(3, 2)) == composition[0, 0](A, Layout(3, 2))
    assert composition(A, B, mode=(0,)) == composition[0](A, B)
    assert composition(A, B, mode=()) == composition(A, B)

    # Mode 0 of a depth-0 layout is the layout itself
    assert composition[0](Layout(6, 1), Layout(3, 1)) == Layout((3,), (1,))

    # The named mode must exist
    with pytest.raises(ValueError):
      composition[2](A, B)

    # Post-conditions hold of the composed mode
    R = composition[0](A, B)
    assert compatible(B, get[0](R))
    for i in range(size(B)):
      assert get[0](R)(i) == get[0](A)(B(i))

  def test_a_tiler_and_its_layout_compose_alike(self):
    """`tiler_to_layout` promotes a by-mode tiler to the single Layout that acts
    identically under composition. That has to hold for a tiler that names only
    some of A's modes too, which is what makes a tiler's rank meaningful: the
    modes it does not name are not in the result."""
    A = Layout((4, 5))

    assert composition(A, (2,)) == composition(A, tiler_to_layout((2,)))
    assert composition(A, (2,)) == Layout((2,), (1,))

    for A, B in [(Layout((4, 5)),                  (2,)),
                 (Layout((4, 5)),                  (2, 5)),
                 (Layout((4, 5)),                  (Layout(2, 2),)),
                 (Layout((4, 5)),                  (Layout(2, 2), Layout(5, 1))),
                 (Layout((4, 5)),                  ()),
                 (Layout(((4, 2), 5), ((5, 40), 1)), (2,)),
                 (Layout(((4, 2), 5), ((5, 40), 1)), ((2, 2), 5)),
                 (Layout((6, 4), (4, 1)),          (3,)),
                 (Layout((6, 4), (4, 1)),          (3, 2))]:
      assert composition(A, B) == composition(A, tiler_to_layout(B)), f"{A} o {B}"
      assert compatible(B, composition(A, B)), f"{A} o {B}"

  def test_composition_none_lhs(self):
    # `A = None` is the identity of unknown extents: it imposes no divisibility
    # condition and leaves the coordinates B walks as the result. Composing it
    # away is invisible to whatever composes next, which is what makes it an
    # identity: composition(A, composition(None, B)) == composition(A, B).
    assert composition(None, 12)                 == Layout(12, 1)
    assert composition(None, Layout(4, 2))       == Layout(4, 2)
    assert composition(None, (3, 2))             == Layout((3, 2), (E(0), E(1)))
    assert composition(None, (Layout(3, 2), 2))  == Layout((3, 2), (2*E(0), E(1)))
    assert composition(None, None)               is None
    # A tiler leaf of None keeps a mode's extent, and there is none to keep, so
    # the mode stands at 1 and carries only its stride.
    assert composition(None, (Layout(3, 2), None)) == Layout((3, 1), (2*E(0), E(1)))

    for A, B in [(Layout(24),               12),
                 (Layout(24),               Layout(4, 2)),
                 (Layout(24),               None),
                 (Layout((6, 4), (4, 1)),   (3, 2)),
                 (Layout((6, 4), (4, 1)),   (Layout(3, 2), 2))]:
      assert composition(A, composition(None, B)) == composition(A, B)

    # `None` has no modes to select, so a mode places the result instead of
    # selecting into A, and the tiler's own modes keep their basis paths.
    assert composition[0](None, (3, 2)) == make_layout([composition(None, (3, 2))])
    assert composition[1](None, 12)     == make_layout([Layout(1, 0), Layout(12, 1)])

  def test_composition_fails(self):

    # Violates shape divisibility condition
    with pytest.raises(ValueError, match="Shape divisibility"):
      self.postcondition_composition(Layout((5,3), (7,1)), Layout(7,1))

    # Which condition a violation is reported under is meaningful, so pin the
    # Whitepaper's own examples. The first two cannot represent every third
    # element and every element respectively; of the last two, one cannot be
    # sufficiently truncated and the other cannot be coalesced.
    with pytest.raises(ValueError, match="Stride divisibility"):
      composition(Layout((4,6,8), (2,3,5)), Layout(6,3))
    with pytest.raises(ValueError, match="Shape divisibility"):
      composition(Layout((4,6,8), (2,3,5)), Layout(6,1))
    assert composition(Layout((4,2,8), (3,12,97)), Layout(3,3)) == Layout(3, 9)
    with pytest.raises(ValueError, match="Stride divisibility"):
      composition(Layout((4,2,8), (3,12,97)), Layout(4,3))
    with pytest.raises(ValueError, match="Stride divisibility"):
      composition(Layout((4,2,8), (3,15,97)), Layout(3,3))

  def test_composition_coords(self):
    # LHS Coords
    self.postcondition_composition(Layout(12, E(0)), Layout((4,3)))
    self.postcondition_composition(Layout(12, 2*E(1)), Layout((4,3)))
    self.postcondition_composition(Layout(12, E(0)), Layout((4,3), (3,1)))
    self.postcondition_composition(Layout(12, 2*E(1)), Layout((4,3), (3,1)))
    self.postcondition_composition(Layout(12, E(1,1)), Layout((2,3), (2,4)))
    self.postcondition_composition(Layout((4,3), (E(0),E(1))), Layout((4,3)))
    self.postcondition_composition(Layout((4,3), (E(0),E(1))), Layout(12))
    self.postcondition_composition(Layout((4,3), (E(0),E(1))), Layout(6, 2))
    self.postcondition_composition(Layout((4,3), (E(0),E(1))), Layout((6,2), (2,1)))
    self.postcondition_composition(Layout((4,3), (E(1),E(0))), Layout((4,3)))
    self.postcondition_composition(Layout((4,3), (E(1),E(0))), Layout(12))
    self.postcondition_composition(Layout((4,3), (E(1),E(0))), Layout(6, 2))
    self.postcondition_composition(Layout((4,3), (E(1),E(0))), Layout((6,2), (2,1)))
    self.postcondition_composition(Layout((4,3), (6*E(1),2*E(1))), Layout((4,3)))
    self.postcondition_composition(Layout((4,3), (6*E(1),2*E(1))), Layout(12))
    self.postcondition_composition(Layout((4,3), (6*E(1),2*E(1))), Layout(6, 2))
    self.postcondition_composition(Layout((4,3), (6*E(1),2*E(1))), Layout((6,2), (2,1)))
    self.postcondition_composition(Layout((4,4), (ArithTuple(1,1),ArithTuple(3,1))), Layout((4,2), (2,1)))
    self.postcondition_composition(Layout((4,6,8), (E(0),E(1),E(2))), Layout((2,2,2)))

    # RHS Coords
    self.postcondition_composition(Layout((4,4), (4,1)), Layout((4,4), (E(0),E(1))))
    self.postcondition_composition(Layout((4,4), (4,1)), Layout((4,4), (E(1),E(0))))
    self.postcondition_composition(Layout((4,5),(5,1)), Layout(30, E(0)))
    self.postcondition_composition(Layout((4,5),(5,1)), Layout(12, E(1)))
    self.postcondition_composition(Layout((4,(4,3),1),(3,(12,1),0)), Layout(12, E(1)))
    self.postcondition_composition(Layout((4,(4,3),1),(3,(12,1),0)), Layout(12, E(1,0)))
    self.postcondition_composition(Layout((4,(2,3)),(6,(3,1))), Layout((2,4), (E(1,1),E(0))))
    self.postcondition_composition(Layout((4,6,8)), Layout((2,2,2), (E(0),E(1),E(2))))
    self.postcondition_composition(Layout((4,6,8)), Layout((2,2,2), (E(2),E(0),E(1))))
    self.postcondition_composition(Layout((4,6,8), (E(0),E(1),E(2))), Layout((2,2,2), (E(0),E(1),E(2))))
    self.postcondition_composition(Layout((3,5,7,11), (E(0),E(1),E(2),E(3))), Layout(3, 4*E(2)))
    self.postcondition_composition(Layout((3,5,7,11), (E(0),E(1),E(2),E(3))), Layout(3, 4*E(2) + 2*E(3)))
    self.postcondition_composition(Layout((3,5,7,11), (E(0),E(1),E(2),E(3))), Layout(3, ArithTuple(1,0,0,1)))

    # Other

    # Diag
    self.postcondition_composition(Layout((4,4), (3,42)), Layout(4, ArithTuple(1,1)))
    # Skew Diag
    self.postcondition_composition(Layout((4,8), (3,42)), Layout(4, ArithTuple(1,2)))


  def postcondition_associativity(self, A, B, C):
    """In general, composition is associative:

    composition(composition(A, B), C) == composition(A, composition(B, C)).

    The one caveat is the *extended domain*: `composition(A, B)` only guarantees
    `R(i) == A(B(i))` for `i in Z(B)`. So the equality holds whenever each
    intermediate result is evaluated inside its guaranteed domain -- every `C(i)`
    is a coordinate of `B` (`< size(B)`) and every `B(C(i))` a coordinate of `A`
    (`< size(A)`). When that nesting holds, both groupings agree on all of `Z(C)`.
    """
    AB_C = composition(composition(A, B), C)
    A_BC = composition(A, composition(B, C))

    logger.info(f"  (({A} o {B}) o {C}) = {AB_C}")
    logger.info(f"  ({A} o ({B} o {C})) = {A_BC}")

    for i in range(size(C)):
      ABC = A(B(C(i)))
      assert AB_C(i) == ABC
      assert A_BC(i) == ABC
      assert AB_C(i) == A_BC(i)

  def test_associativity(self):
    cases = [
      (Layout((4, 3), (3, 1)), Layout((4, 3), (1, 4)), Layout((4, 3), (1, 4))),
      (Layout((4, 3), (3, 1)), Layout((4, 3), (1, 4)), Layout(6, 2)),
      (Layout((4, 3), (1, 4)), Layout((4, 3), (3, 1)), Layout((4, 3), (1, 4))),
      (Layout((4, 3), (1, 4)), Layout((4, 3), (3, 1)), Layout(6, 2)),
      (Layout((6, 4), (4, 1)), Layout((4, 3), (3, 1)), Layout((4, 3), (1, 4))),
      (Layout((6, 4), (4, 1)), Layout((4, 3), (3, 1)), Layout(6, 2)),
      (Layout(6, 2),           Layout((2, 3), (3, 1)), Layout((2, 3), (1, 2))),
      (Layout(6, 2),           Layout((2, 3), (1, 2)), Layout((2, 3), (3, 1))),
      (Layout((2, 3), (3, 1)), Layout((2, 3), (1, 2)), Layout((2, 3), (1, 2))),
      (Layout((8, 8), (8, 1)), Layout((6, 4), (4, 1)), Layout(6, 2)),
      (Layout((8, 8), (8, 1)), Layout((6, 4), (4, 1)), Layout((2, 3), (1, 2))),
      (Layout((4, 6), (1, 4)), Layout((4, 3), (3, 1)), Layout((4, 3), (1, 4))),
    ]
    for layoutA, layoutB, layoutC in cases:
      self.postcondition_associativity(layoutA, layoutB, layoutC)

  def test_composition_sympy(self):
    N, X, Y = sympy.symbols("N X Y", positive=True, integer=True)
    self.postcondition_composition(Layout(N, X), Layout(3, 4))
    self.postcondition_composition(Layout(N, X), Layout(2, 1))
    self.postcondition_composition(Layout(N, 1), Layout(3, 4))
    self.postcondition_composition(Layout(N, X), Layout((2, 3), (1, 2)))
    self.postcondition_composition(Layout((12, N), (X, Y)), Layout(6, 4))
    self.postcondition_composition(Layout((8, N), (X, Y)), Layout(4, 1))
    self.postcondition_composition(Layout((4, N), (X, Y)), Layout((2, 2), (2, 1)))

  def test_composition_sympy_fails(self):
    # Dividing the concrete tile by a *symbolic* leading shape factor is an
    # unverifiable divisibility condition, so composition must raise rather
    # than emit a silently-unchecked symbolic result.
    N, X, Y = sympy.symbols("N X Y", positive=True, integer=True)
    with pytest.raises(ValueError, match="Shape divisibility"):
      composition(Layout((N, 8), (X, Y)), Layout(4, 1))
