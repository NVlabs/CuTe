# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.right_inverse
"""

import logging
import sympy

from pycute import *

logger = logging.getLogger()


class TestRightInverse:
  def postcondition_right_inverse(self, L):
    inv_layout = right_inverse(L)

    logger.info(f"  {L}  =>  {inv_layout}")

    assert weakly_congruent(coprofile(L), shape(inv_layout))

    # Generalized right inverse condition
    for i in range(size(inv_layout)):
      assert inv_layout(L(inv_layout(i))) == inv_layout(i)

    # Canonical right inverse post-condition is satisfied with codomain Z
    if is_int(L(0)):
      for i in range(size(inv_layout)):
        assert L(inv_layout(i)) == i

  def test_right_inverse(self):
    self.postcondition_right_inverse(Layout(1,0))
    self.postcondition_right_inverse(Layout(1,1))
    self.postcondition_right_inverse(Layout(1,2))
    self.postcondition_right_inverse(Layout(1,4))
    self.postcondition_right_inverse(Layout(4,0))
    self.postcondition_right_inverse(Layout(4,1))
    self.postcondition_right_inverse(Layout(4,2))
    self.postcondition_right_inverse(Layout(4,4))
    self.postcondition_right_inverse(Layout((1,1),(0,0)))
    self.postcondition_right_inverse(Layout((3,7),(0,0)))
    self.postcondition_right_inverse(Layout((2,4),(0,2)))
    self.postcondition_right_inverse(Layout((8,4),(1,8)))
    self.postcondition_right_inverse(Layout((8,4),(4,1)))
    self.postcondition_right_inverse(Layout((2,4,6),(1,2,8)))
    self.postcondition_right_inverse(Layout((2,4,6),(4,1,8)))
    self.postcondition_right_inverse(Layout((4,2),(1,16)))

    # Non-injective
    self.postcondition_right_inverse(Layout((4,5,6),(1,1,4)))
    self.postcondition_right_inverse(Layout((7,5,9),(2,0,1)))

  def test_right_inverse_coord(self):
    self.postcondition_right_inverse(Layout((4,5),(E(0),E(1))))
    self.postcondition_right_inverse(Layout((4,5),(E(1),E(0))))
    self.postcondition_right_inverse(Layout((4,5),(E(1),E(4,1))))
    self.postcondition_right_inverse(Layout((4,5),(2*E(0),2*E(1))))

    # SM70 MMA 8x8x4 C TV inverse
    self.postcondition_right_inverse(Layout(((   2,      2,      2), (   2,      2,      2)),
                                            ((E(0), 2*E(1), 4*E(0)), (E(1), 2*E(0), 4*E(1)))))
    self.postcondition_right_inverse(Layout(((   2,      2,      2), (   2,      2,      2)),
                                            ((E(0), 2*E(1), 5*E(0)), (E(1), 2*E(0), 5*E(1)))))
    self.postcondition_right_inverse(Layout(((   2,      2,      2), (   2,      2,      2)),
                                            ((E(0), 2*E(1), 5*E(0)), (E(1), 2*E(0), 4*E(1)))))

    # SM70 MMA 8x8x4 A TV inverse
    self.postcondition_right_inverse(composition(tiler_to_layout((8,4)),
                                                 Layout(((4,2),4), ((8,4),1))))

    # SM80 MMA 16x8 TV inverse
    self.postcondition_right_inverse(composition(tiler_to_layout((16,8)),
                                                 Layout(((4,8),(2,2)), ((32,1),(16,8)))))

  def test_right_inverse_sympy(self):
    # Symbolic shapes with static strides: the bijective stride chain is
    # concrete, so the right inverse is exact. A symbolic *stride* is
    # deprioritized past the sort and ignored unless it continues the chain.
    N, M, X, DM, DN = sympy.symbols("N M X DM DN", positive=True, integer=True)

    assert right_inverse(Layout(N, 1)) == Layout(N, 1)
    assert right_inverse(Layout((4, N), (1, 4))) == Layout((4, N), (1, 4))
    assert right_inverse(Layout((N, M), (1, N))) == Layout(N*M, 1)

    # A strided (non-surjective) symbolic layout has only the trivial inverse.
    assert right_inverse(Layout(N, X)) == Layout(1, 0)

    # Symbolic strides that can't be ordered are deprioritized and ignored
    # (this previously raised in the sort); a symbolic stride that continues
    # the chain is still kept.
    assert right_inverse(Layout((M, N), (DM, DN))) == Layout(1, 0)
    assert right_inverse(Layout((N, 4), (1, N))) == Layout((N, 4), (1, N))

  def test_right_inverse_sympy_substitution(self):
    # Substituting concrete values into each symbolic right inverse must yield
    # a valid right inverse of the substituted layout, i.e. L(R(i)) == i.
    # (Asserting structural equality against the substituted form is not valid
    # because the *concrete* layout coalesces more aggressively than the
    # symbolic one, giving an equivalent but differently-shaped inverse.)
    for n in (1, 2, 3, 5):
      cases = [(Layout(n, 1),             Layout(n, 1)),
               (Layout((4, n), (1, 4)),   Layout((4, n), (1, 4))),
               (Layout((n, 4), (1, n)),   Layout((n, 4), (1, n))),
               (Layout((n, n+1), (1, n)), Layout(n*(n+1), 1))]
      for layout, inv in cases:
        for i in range(size(inv)):
          assert layout(inv(i)) == i
