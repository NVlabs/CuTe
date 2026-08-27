# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.coalesce_z
"""

import logging
import sympy

from pycute import *

logger = logging.getLogger()


class TestCoalesceZ:
  def postcondition_coalesce_z(self, L):
    R = coalesce_z(L)

    logger.info(f"  {L}  =>  {R}")

    assert depth(R) <= 1
    assert size(R) == size(L)

    for i in range(10+size(L)):
      assert R(i) == L(i)

    # Idempotence
    assert coalesce_z(R) == R


  def test_coalesce_z(self):
    self.postcondition_coalesce_z(Layout(1,0))
    self.postcondition_coalesce_z(Layout(1,1))
    self.postcondition_coalesce_z(Layout((1,1), (5,7)))
    self.postcondition_coalesce_z(Layout((2,4)))
    self.postcondition_coalesce_z(Layout((2,4), (4,1)))
    self.postcondition_coalesce_z(Layout((2,4,6)))
    self.postcondition_coalesce_z(Layout((2,4,6), (24,6,1)))
    self.postcondition_coalesce_z(Layout((2,(4,6))))
    self.postcondition_coalesce_z(Layout((2,4,6), (1,6,2)))
    self.postcondition_coalesce_z(Layout((2,1,6), (1,7,2)))
    self.postcondition_coalesce_z(Layout((2,1,6), (4,7,8)))
    self.postcondition_coalesce_z(Layout((2,1,6,1), (4,7,8,0)))
    self.postcondition_coalesce_z(Layout((2,1,6,1), (4,7,8,57)))
    self.postcondition_coalesce_z(Layout((2,1,6,1), (1,7,8,0)))
    self.postcondition_coalesce_z(Layout((2,1,6,1), (1,7,8,57)))
    self.postcondition_coalesce_z(Layout((2,1,6,1), (1,7,8,48)))
    self.postcondition_coalesce_z(Layout((2,1,3), (2,4,4)))
    self.postcondition_coalesce_z(Layout(((2,2),(2,2)), ((1,4),(8,32))))


  def test_coalesce_z_mode(self):
    # `coalesce_z[mode](A)` coalesces that one mode of A and leaves the others
    # alone, keeping the trailing size-1 modes that `coalesce` would drop.
    A = Layout(((2, (1, 6)), (3, 4, 1)), ((1, (6, 2)), (100, 300, 0)))

    assert coalesce_z[1](A) == make_layout([A[0], coalesce_z(A[1])])
    assert coalesce_z[1](A) == coalesce_z(A, (None, 1))
    assert coalesce_z(A, mode=(1,)) == coalesce_z[1](A)
    assert coalesce_z(A, mode=()) == coalesce_z(A)
    assert shape(coalesce_z[1](A))[-1] == (12, 1)   # size-1 mode preserved


  def test_coalesce_z_coord(self):
    self.postcondition_coalesce_z(Layout(1,E(0)))
    self.postcondition_coalesce_z(Layout(1,E(1)))
    self.postcondition_coalesce_z(Layout((1,1), (E(0),E(1))))
    self.postcondition_coalesce_z(Layout((2,4), (E(0),E(1))))
    self.postcondition_coalesce_z(Layout((2,4), (E(1),2*E(1))))
    self.postcondition_coalesce_z(Layout((2,1,6,1), (E(1,1),E(2,3),2*E(1,1),E(2,3))))
    self.postcondition_coalesce_z(Layout((2,1,6,1), (E(1,1),E(2,3),2*E(1,0),E(2,3))))
    self.postcondition_coalesce_z(Layout((2,1,6,1), (E(1,0),E(2,3),2*E(1,0),12*E(1,0))))
    self.postcondition_coalesce_z(Layout(((2,2),(2,2)), ((E(0),E(1)),(2*E(1),2*E(0)))))


  # F2 strides under coalesce_z. ``coalesce_z`` shares the same
  # adjacent-mode merge predicate as ``coalesce``, so the post-condition
  # ``layoutR(i) == layout(i)`` must hold over F2 strides whose shape
  # modes mix pow-2 and non-pow-2 factors.
  def test_coalesce_z_f2(self):
    self.postcondition_coalesce_z(Layout(8, F2(1)))

    # Pow-2 modes: every adjacent pair satisfies both merge conditions.
    self.postcondition_coalesce_z(Layout((4, 4), (F2(1), F2(4))))
    self.postcondition_coalesce_z(Layout((2, 2, 2, 2), (F2(1), F2(2), F2(4), F2(8))))

    # Non-pow-2 ``s_a`` blocks the merge; both modes remain separate.
    self.postcondition_coalesce_z(Layout((3, 4), (F2(1), F2(3))))
    self.postcondition_coalesce_z(Layout((3, 3), (F2(1), F2(3))))

    # Mixed: a pow-2 ``s_a`` lets the leading pair merge, while a
    # non-pow-2 ``s_a`` blocks the next merge.
    self.postcondition_coalesce_z(Layout((2, 2, 3), (F2(1), F2(2), F2(4))))
    self.postcondition_coalesce_z(Layout((3, 2, 2), (F2(1), F2(3), F2(6))))

    # Nested shape with F2 strides.
    self.postcondition_coalesce_z(Layout(((2, 2), (3, 2)), ((F2(1), F2(2)), (F2(4), F2(12)))))

    # Shape-1 modes interleaved with F2 strides.
    self.postcondition_coalesce_z(Layout((2, 1, 2), (F2(1), F2(7), F2(2))))
    self.postcondition_coalesce_z(Layout((3, 1, 3), (F2(1), F2(7), F2(3))))


  def test_coalesce_z_sympy(self):
    # Concrete shapes merge through symbolic strides; the size stays concrete
    # so the extended-domain post-condition applies.
    X = sympy.symbols("X", positive=True, integer=True)
    self.postcondition_coalesce_z(Layout((2, 4), (X, 2*X)))         # -> 8:X
    self.postcondition_coalesce_z(Layout((2, 3, 4), (X, 2*X, 6*X))) # -> 24:X
    self.postcondition_coalesce_z(Layout((2, 4), (X, 3*X)))         # no merge


  def test_coalesce_z_sympy_shapes(self):
    # ``coalesce_z`` shares the merge, and a symbolic extent records the product
    # ``s_a*s_b`` as faithfully as a concrete one, so a symbol is no barrier.
    N, M, X, Y = sympy.symbols("N M X Y", positive=True, integer=True)

    # A symbol merges on either side of the pair, and with a symbol or a
    # concrete extent opposite it.
    assert coalesce_z(Layout((4, N), (1, 4))) == Layout(4*N, 1)
    assert coalesce_z(Layout((N, 4), (1, N))) == Layout(4*N, 1)
    assert coalesce_z(Layout((2, N), (1, 2))) == Layout(2*N, 1)
    assert coalesce_z(Layout((N, M), (1, N))) == Layout(N*M, 1)

    # Merging is order-independent even where the product is rewritten on the
    # way in: both of these form ``(N+1)*2``, which distributes to ``2*N + 2``.
    assert coalesce_z(Layout((N+1, 2), (1, N+1))) == Layout(2*N+2, 1)
    assert coalesce_z(Layout((2, N+1), (1, 2)))   == Layout(2*N+2, 1)

    # Unrelated strides do not merge; a size-1 mode still drops.
    assert coalesce_z(Layout((N, M), (X, Y))) == Layout((N, M), (X, Y))
    assert coalesce_z(Layout((1, N), (X, Y))) == Layout(N, Y)


  def test_coalesce_z_opaque_shape(self):
    # A leaf whose ``*`` mints a fresh node and compares by identity -- a traced
    # or JIT integer -- cannot record a merged extent: ``s_a*s_b`` would name a
    # value nobody can read back. The merge forms the product twice and refuses
    # when the two disagree, so the modes stay split even though the strides are
    # concrete and satisfy both linearity conditions.
    class Handle:
      def __mul__(self, other):  return Handle()
      __rmul__ = __mul__
    register_integer_type(Handle)

    L = Layout((4, Handle()), (1, 4))
    assert coalesce_z(L) == L
    assert coalesce(L) == L

    # The same leaf in a size-1 mode is dropped, not merged.
    assert coalesce(Layout((1, 4), (Handle(), 1))) == Layout(4, 1)
