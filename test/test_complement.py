# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.complement
"""

import logging
import sympy

from pycute import *

logger = logging.getLogger()


class TestComplement:

  def postcondition_complement(self, L):

    R = complement(L)

    logger.info(f" {L}  =>  {R}")

    # Post-condition: weak congruence with codomain
    assert weakly_congruent(coprofile(L), shape(R))

    # Post-condition: test orderedness and disjointness of the codomains
    for i in range(1, 10+size(R)):
      assert R(i-1) < R(i)  # Ordered
      for j in range(size(L)):
        assert R(i) != L(j)  # Disjoint

    return R

  def postcondition_complement_strong(self, L):

    # Obey all of the weak preconditions
    R = self.postcondition_complement(L)

    # Generalized inverse conditions
    completed = make_layout([L, R])
    inv_layout = right_inverse(completed)

    # Right inverse condition
    for i in range(size(inv_layout)):
      assert inv_layout(completed(inv_layout(i))) == inv_layout(i)

    # Left inverse condition -- the right inverse is a generalized reflexive inverse
    for i in range(size(completed)):
      assert completed(inv_layout(completed(i))) == completed(i)

  #########################################################

  def test_complement(self):
    self.postcondition_complement_strong(Layout(1,0))
    self.postcondition_complement_strong(Layout(1,1))
    self.postcondition_complement_strong(Layout(1,2))
    self.postcondition_complement_strong(Layout(1,4))
    self.postcondition_complement_strong(Layout((1,1),(0,0)))
    self.postcondition_complement_strong(Layout((3,7),(0,0)))
    self.postcondition_complement_strong(Layout(5,1))
    self.postcondition_complement_strong(Layout(5,3))
    self.postcondition_complement_strong(Layout(4,0))
    self.postcondition_complement_strong(Layout(4,1))
    self.postcondition_complement_strong(Layout(4,2))
    self.postcondition_complement_strong(Layout(4,4))
    self.postcondition_complement_strong(Layout((2,4),(1,2)))
    self.postcondition_complement_strong(Layout((2,3),(1,2)))
    self.postcondition_complement_strong(Layout((2,4),(1,4)))
    self.postcondition_complement_strong(Layout((8,4),(1,8)))
    self.postcondition_complement_strong(Layout((8,4),(4,1)))
    self.postcondition_complement_strong(Layout((2,4,6),(1,2,8)))
    self.postcondition_complement_strong(Layout((2,4,6),(4,1,8)))
    self.postcondition_complement_strong(Layout((2,4,8),(8,1,64)))
    self.postcondition_complement_strong(Layout((2,4,8), (32,0,2)))
    self.postcondition_complement_strong(Layout((2,4,8), (2,0,32)))
    self.postcondition_complement_strong(Layout((2,4,4,4,2), (32,0,2,0,512)))
    self.postcondition_complement_strong(Layout(((2,2),(2,2)),((1,4),(8,32))))
    self.postcondition_complement_strong(Layout((2,(3,4)),(3,(1,6))))
    self.postcondition_complement_strong(Layout((4,2),(1,16)))

  def test_complement_weak(self):
    self.postcondition_complement(Layout((4,2),(1,6)))
    self.postcondition_complement(Layout((4,2),(1,5)))
    self.postcondition_complement(Layout((4,2),(1,10)))
    self.postcondition_complement(Layout((4,2),(1,11)))
    self.postcondition_complement(Layout((2,4),(11,1)))

  def test_complement_coord(self):
    self.postcondition_complement_strong(Layout(3, E(0)))
    self.postcondition_complement_strong(Layout(3, 4*E(2)))
    self.postcondition_complement_strong(Layout((2,5,3), (4*E(1), 5*E(0), 16*E(1))))
    self.postcondition_complement_strong(Layout((2,3,5), (4*E(1), 5*E(0), 7*E(2,1))))
    self.postcondition_complement_strong(Layout((2,3,5), (4*E(1), 0, 7*E(2,1))))

  def test_complement_sympy(self):
    # Symbolic shapes with static strides: the strides are orderable so the
    # gap structure is concrete and the complement is exact. (Symbolic
    # strides can't be ordered and are filtered past the sort; that is a
    # separate, best-effort subset -- see the substitution check below for
    # the cases verified to be correct.)
    N, X = sympy.symbols("N X", positive=True, integer=True)
    assert complement(Layout(N, 1)) == Layout(1, N)
    assert complement(Layout(N, X)) == Layout((X, 1), (1, N*X))
    assert complement(Layout((4, N), (1, 4))) == Layout(1, 4*N)
    assert complement(Layout((2, N), (1, 2))) == Layout(1, 2*N)
    # A symbolic-stride mode is processed after the sorted static gaps; here
    # it resolves cleanly because the layout is contiguous.
    assert complement(Layout((N, 4), (1, N))) == Layout(1, 4*N)
    # The extend works with trailing symbolic extents
    assert Layout(256, 1)._complement(extend=(32,4,2,2,N)) == Layout((2,N), (256,512))

  def test_complement_sympy_substitution(self):
    # The symbolic complement must agree with the concrete complement under
    # any concrete substitution, and each concrete result must obey the
    # ordered / disjoint post-condition.
    for n in (1, 2, 3, 5):
      assert complement(Layout((4, n), (1, 4))) == Layout(1, 4*n)  # 4*N
      assert complement(Layout(n, 1)) == Layout(1, n)    #   N
      assert complement(Layout((n, 4), (1, n))) == Layout(1, 4*n)  # 4*N
      self.postcondition_complement(Layout((4, n), (1, 4)))
      self.postcondition_complement(Layout(n, 1))
      self.postcondition_complement(Layout((n, 4), (1, n)))
