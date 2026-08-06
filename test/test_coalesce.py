# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.coalesce
"""

import logging
import sympy

from pycute import *

logger = logging.getLogger()


class TestCoalesce:
  def postcondition_coalesce(self, L, expected=None):
    R = coalesce(L)

    logger.info(f"  {L}  =>  {R}")

    if expected is not None:
      assert R == expected

    assert depth(R) <= 1
    assert size(R) == size(L)

    for i in range(size(L)):
      assert R(i) == L(i)

    # Idempotence
    assert coalesce(R) == R

    # Composition-is-coalesced
    assert composition(L, Layout(size(L), 1)) == R


  def test_coalesce(self):
    self.postcondition_coalesce(Layout(1,0))
    self.postcondition_coalesce(Layout(1,1))
    self.postcondition_coalesce(Layout((1,1), (5,7)))
    self.postcondition_coalesce(Layout((2,4)))
    self.postcondition_coalesce(Layout((2,4), (4,1)))
    self.postcondition_coalesce(Layout((2,4,6)))
    self.postcondition_coalesce(Layout((2,4,6), (24,6,1)))
    self.postcondition_coalesce(Layout((2,(4,6))))
    self.postcondition_coalesce(Layout((2,4,6), (1,6,2)))
    self.postcondition_coalesce(Layout((2,1,6), (1,7,2)))
    self.postcondition_coalesce(Layout((2,1,6), (4,7,8)))
    self.postcondition_coalesce(Layout((2,1,6,1), (4,7,8,0)))
    self.postcondition_coalesce(Layout((2,1,6,1), (4,7,8,57)))
    self.postcondition_coalesce(Layout((2,1,6,1), (1,7,8,0)))
    self.postcondition_coalesce(Layout((2,1,6,1), (1,7,8,57)))
    self.postcondition_coalesce(Layout((2,1,6,1), (1,7,8,48)))
    self.postcondition_coalesce(Layout((2,1,3), (2,4,4)))
    self.postcondition_coalesce(Layout(((2,2),(2,2)), ((1,4),(8,32))))


  def test_coalesce_mode(self):
    # `coalesce[mode](A)` coalesces that one mode of A and leaves the others
    # alone -- it is `coalesce` with the profile lifted to `mode`.
    A = Layout(((2, (1, 6)), (3, 4)), ((1, (6, 2)), (100, 300)))

    assert coalesce[1](A) == make_layout([A[0], coalesce(A[1])])
    assert coalesce[0](A) == make_layout([coalesce(A[0]), A[1]])
    assert coalesce[1](A) == coalesce(A, (None, 1))
    assert coalesce(A, mode=(1,)) == coalesce[1](A)
    assert coalesce(A, mode=()) == coalesce(A)

    # A mode is coalesced but the layout is otherwise preserved
    R = coalesce[1](A)
    assert depth(get[1](R)) <= 1
    assert size(R) == size(A)
    for i in range(size(A)):
      assert R(i) == A(i)


  def test_coalesce_coord(self):
    self.postcondition_coalesce(Layout(1,E(0)))
    self.postcondition_coalesce(Layout(1,E(1)))
    self.postcondition_coalesce(Layout((1,1), (E(0),E(1))))
    self.postcondition_coalesce(Layout((2,4), (E(0),E(1))))
    self.postcondition_coalesce(Layout((2,4), (E(1),2*E(1))))
    self.postcondition_coalesce(Layout((2,1,6,1), (E(1,1),E(2,3),2*E(1,1),E(2,3))))
    self.postcondition_coalesce(Layout((2,1,6,1), (E(1,1),E(2,3),2*E(1,0),E(2,3))))
    self.postcondition_coalesce(Layout((2,1,6,1), (E(1,0),E(2,3),2*E(1,0),12*E(1,0))))
    self.postcondition_coalesce(Layout(((2,2),(2,2)), ((E(0),E(1)),(2*E(1),2*E(0)))))


  # F2 strides: the merge ``(s_a, s_b):(d_a, d_b) -> (s_a*s_b, d_a)`` is
  # valid iff
  #   (1) ``d_b == s_a * d_a``     (linearity at coord ``(0, 1)``)
  #   (2) ``s_a`` is a power of 2  (no clmul/int-mul carries in coord
  #                                 decomposition; encoded by linearity
  #                                 at coord ``(s_a-1, 1)``).
  # ``expected`` is the coalesced form where it is asserted; cases without it
  # check only the post-conditions (which already pin the pointwise codomain).
  def test_coalesce_f2(self):
    self.postcondition_coalesce(Layout(1, F2(0)))
    self.postcondition_coalesce(Layout(1, F2(7)))
    self.postcondition_coalesce(Layout((1, 1), (F2(5), F2(7))))
    self.postcondition_coalesce(Layout(8, F2(1)))

    # Pow-2 shape modes with natural prefix-product stride: both merge
    # conditions hold so the layout coalesces.
    self.postcondition_coalesce(Layout((2, 4), (F2(1), F2(2))))
    self.postcondition_coalesce(Layout((4, 2), (F2(1), F2(4))))

    # Pow-2 shape modes nested: every adjacent pair satisfies both merge
    # conditions, so the layout collapses to a single mode.
    self.postcondition_coalesce(Layout((2, 2, 2, 2), (F2(1), F2(2), F2(4), F2(8))), expected=Layout(16, F2(1)))
    self.postcondition_coalesce(Layout((4, 4), (F2(1), F2(4))), expected=Layout(16, F2(1)))

    # Pow-2 with non-trivial (non-1) ``d_a``: still mergeable because the
    # bit-disjointness obstruction only depends on ``s_a`` being pow-2.
    self.postcondition_coalesce(Layout((4, 4), (F2(3), F2(12))), expected=Layout(16, F2(3)))  # 4 * F2(3) == F2(12) under clmul

    # Non-pow-2 ``s_a`` with natural prefix-product stride: cond (1)
    # holds (``s_a * d_a == d_b``) but cond (2) fails, so the two modes
    # remain separate (the coalesced form equals the input).
    self.postcondition_coalesce(Layout((3, 3), (F2(1), F2(3))), expected=Layout((3, 3), (F2(1), F2(3))))  # 3 clmul F2(1) == F2(3)
    self.postcondition_coalesce(Layout((3, 4), (F2(1), F2(3))), expected=Layout((3, 4), (F2(1), F2(3))))
    self.postcondition_coalesce(Layout((5, 3), (F2(1), F2(5))), expected=Layout((5, 3), (F2(1), F2(5))))  # 5 clmul F2(1) == F2(5)
    self.postcondition_coalesce(Layout((6, 5), (F2(1), F2(6))), expected=Layout((6, 5), (F2(1), F2(6))))  # 6 clmul F2(1) == F2(6)

    # Non-pow-2 ``s_a`` with non-trivial ``d_a``.
    self.postcondition_coalesce(Layout((3, 4), (F2(2), F2(6))), expected=Layout((3, 4), (F2(2), F2(6))))  # 3 clmul F2(2) == F2(6)

    # Asymmetry: ``s_a`` (the *left* mode of the pair being merged) is
    # what must be a power of 2; ``s_b`` may be anything. The pair
    # ``(s_a=4, s_b=3)`` has pow-2 ``s_a`` so the merge is valid even
    # though the trailing shape factor is non-pow-2.
    self.postcondition_coalesce(Layout((2, 2, 3), (F2(1), F2(2), F2(4))), expected=Layout(12, F2(1)))

    # Reverse order: non-pow-2 ``s_a=3`` blocks the first merge, but the
    # pow-2 tail still folds via the ``(s_a=2, s_b=2)`` pair.
    self.postcondition_coalesce(Layout((3, 2, 2), (F2(1), F2(3), F2(6))), expected=Layout((3, 4), (F2(1), F2(3))))

    # Nested shape with mixed pow-2 / non-pow-2 leaves.
    self.postcondition_coalesce(Layout(((2, 2), (3, 2)), ((F2(1), F2(2)), (F2(4), F2(12)))))

    # Shape-1 modes interleaved with F2 strides: the ``s == 1`` rule
    # composes with the merge condition so the trailing pow-2 mode still
    # folds with the leading one.
    self.postcondition_coalesce(Layout((2, 1, 2), (F2(1), F2(7), F2(2))), expected=Layout(4, F2(1)))
    self.postcondition_coalesce(Layout((3, 1, 3), (F2(1), F2(7), F2(3))), expected=Layout((3, 3), (F2(1), F2(3))))


  def test_coalesce_f2_int_carry_obstruction(self):
    # ``Layout((3, 4), (F2(1), F2(3)))`` satisfies cond (1) of the merge
    # predicate (``3 * F2(1) == F2(3)`` under clmul) but violates cond
    # (2) because ``s_a = 3`` is not a power of 2. Collapsing it into
    # ``Layout(12, F2(1))`` would be wrong: the merged single mode
    # evaluates the index by INT linearization while the original
    # multi-mode evaluates by XOR over per-mode contributions. The two
    # diverge at any coord where ``s_a * c_b`` produces an int carry,
    # e.g. at ``i = 4``:
    #     merged: 4 * F2(1) = F2(4)
    #     actual: c_0=1, c_1=1 -> F2(1) XOR F2(3) = F2(2)
    # ``coalesce`` therefore keeps both modes and preserves every
    # codomain value of the original layout.
    layout = Layout((3, 4), (F2(1), F2(3)))
    R = coalesce(layout)

    assert R != Layout(12, F2(1))
    self.postcondition_coalesce(layout)


  def test_coalesce_sympy(self):
    # Concrete shapes still merge through symbolic strides: no concrete shape
    # is folded into a symbol, and the size stays concrete so the full
    # pointwise post-condition applies.
    X = sympy.symbols("X", positive=True, integer=True)
    self.postcondition_coalesce(Layout((2, 4), (X, 2*X)))         # -> 8:X
    self.postcondition_coalesce(Layout((2, 3, 4), (X, 2*X, 6*X))) # -> 24:X
    self.postcondition_coalesce(Layout((2, 4), (X, 3*X)))         # no merge


  def test_coalesce_sympy_static(self):
    # The merge folds two shapes into the product ``s_a*s_b``, so a concrete
    # shape must NOT merge into a symbolic one (which would bury e.g. 4 inside
    # 4*N). Two symbolic shapes may still merge. Symbolic shapes give a
    # symbolic size (no iteration), so assert the coalesced form directly.
    N, M, X, Y = sympy.symbols("N M X Y", positive=True, integer=True)

    # Blocked: a concrete and a symbolic shape stay separate.
    assert coalesce(Layout((4, N), (1, 4))) == Layout((4, N), (1, 4))
    assert coalesce(Layout((N, 4), (1, N))) == Layout((N, 4), (1, N))
    assert coalesce(Layout((2, N), (1, 2))) == Layout((2, N), (1, 2))

    # Allowed: two symbolic shapes merge.
    assert coalesce(Layout((N, M), (1, N))) == Layout(N*M, 1)

    # Unrelated strides do not merge; a size-1 mode still drops.
    assert coalesce(Layout((N, M), (X, Y))) == Layout((N, M), (X, Y))
    assert coalesce(Layout((1, N), (X, Y))) == Layout(N, Y)
