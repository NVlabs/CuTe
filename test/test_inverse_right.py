# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.right_inverse
"""

import logging
import pytest
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

    return inv_layout

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

  def test_right_inverse_f2(self):
    """`right_inverse` accepts a stride that carries a component the covered modes
    already span, so it inverts swizzles as well as compact `F2` layouts."""
    self.postcondition_right_inverse(Layout(8, F2(0)))
    self.postcondition_right_inverse(Layout(8, F2(1)))
    self.postcondition_right_inverse(Layout(8, F2(2)))
    self.postcondition_right_inverse(Layout(8, F2(3)))
    self.postcondition_right_inverse(Layout((4, 8), (F2(1), F2(4))))
    self.postcondition_right_inverse(Layout((4, 8), (F2(1), 0)))
    self.postcondition_right_inverse(Layout((4, 8), (0, F2(1))))
    self.postcondition_right_inverse(Layout((4, 4), (F2(1), F2(1))))   # non-injective
    self.postcondition_right_inverse(Layout((8, 8), (F2(1), F2(9))))
    self.postcondition_right_inverse(Layout((8, 8), (F2(9), F2(1))))
    self.postcondition_right_inverse(Layout((4, 4), (F2(1), F2(5))))
    self.postcondition_right_inverse(Layout((4, (4, 3)), (F2(1), (F2(5), F2(16)))))
    self.postcondition_right_inverse(Layout((8, 8), (F2(1), F2(13))))

  def test_right_inverse_f2_is_largest(self):
    """A bijective `F2` layout inverts across its whole codomain. The chain walk
    alone could not reach this: a swizzle deliberately breaks the chain, since
    `F2(9) == F2(8) + F2(1)` folds bit 0 into the high bit-field."""
    for L in [Layout((8, 8), (F2(1), F2(9))),
              Layout((8, 8), (F2(9), F2(1))),
              Layout((4, 4), (F2(1), F2(5))),
              Layout((4, (4, 3)), (F2(1), (F2(5), F2(16))))]:
      assert len({int(L(i)) for i in range(size(L))}) == size(L)   # L is bijective
      assert size(right_inverse(L)) == size(L)                     # ... and fully inverted
      for i in range(size(L)):
        assert L(right_inverse(L)(i)) == i

  def test_right_inverse_f2_residue_must_not_leak(self):
    """A residue that fits below the chain stride can still leak past the field
    that stride opens, once a coordinate scales it -- so the residue is tested
    across the mode's whole extent. `F2(13) == F2(8) + F2(5)` has `F2(5) < F2(8)`,
    yet `4 * F2(13)` reaches bit 4, outside the field `F2(8)` opens."""
    assert F2(5) < F2(8)                                  # the residue alone fits ...
    assert 7 * F2(5) >= F2(8)                             # ... but not once scaled
    assert int(4 * F2(13)) == 0b110100                    # bit 4 is set, bit 2 is not

    L = Layout((8, 8), (F2(1), F2(13)))
    L_inv = self.postcondition_right_inverse(L)
    assert size(L_inv) == 8                    # so the chain stops at mode 0
    for i in range(8):
      assert L(L_inv(i)) == i

  def test_right_inverse_f2_valued_coordinate_axis(self):
    """A coordinate axis may carry a swizzled (`F2`) offset. As for any `Z^S`
    codomain the canonical condition is skipped, since `L(R(i))` is a coordinate
    rather than an integer."""
    Fa = lambda v, k: ScaledBasis(F2(v), (k,))
    self.postcondition_right_inverse(Layout(4, Fa(1, 0)))
    self.postcondition_right_inverse(Layout((8, 8), (Fa(1, 0), Fa(9, 0))))
    self.postcondition_right_inverse(Layout((8, 8), (Fa(1, 0), Fa(1, 1))))

  def test_right_inverse_mixed_codomain_algebras_do_not_compose(self):
    """A codomain mixing a swizzled axis with an ordinary index axis inverts each
    axis into a *different* algebra, and `inner_product` then sums the resulting
    modes with a single `+`. The inverse is built, but evaluating it past the first
    axis raises rather than silently mixing XOR with integer addition."""
    R = right_inverse(Layout((8, 8), (ScaledBasis(F2(1), (0,)), E(1))))
    assert R == Layout((8, 8), (F2(1), 8))       # an F2 stride beside an int stride
    assert R(7) == F2(7)                         # the F2 axis alone is fine
    with pytest.raises(TypeError):
      R(8)                                       # ... but the int axis cannot join it

  def test_right_inverse_f2_involution(self):
    """A swizzle that XORs one bit-field into another is its own right inverse."""
    for L in [Layout((8, 8), (F2(1), F2(9))), Layout((4, 4), (F2(1), F2(5)))]:
      assert right_inverse(L) == L
    # Transposing the strides stays bijective but is no longer self-inverse.
    assert right_inverse(Layout((8, 8), (F2(9), F2(1)))) == Layout((8, 8), (F2(8), F2(9)))

  def test_right_inverse_sympy(self):
    # Symbolic shapes with static strides: the bijective stride chain is
    # concrete, so the right inverse is exact. A symbolic *stride* is
    # deprioritized past the sort and ignored unless it continues the chain.
    N, M, X, DM, DN = sympy.symbols("N M X DM DN", positive=True, integer=True)

    assert right_inverse(Layout(N, 1)) == Layout(N, 1)
    assert right_inverse(Layout((4, N), (1, 4))) == Layout(4*N, 1)
    assert right_inverse(Layout((N, M), (1, N))) == Layout(N*M, 1)

    # A strided (non-surjective) symbolic layout has only the trivial inverse.
    assert right_inverse(Layout(N, X)) == Layout(1, 0)

    # Symbolic strides that can't be ordered are deprioritized past the sort and
    # ignored; a symbolic stride that continues the chain is still kept, and the
    # recovered modes then coalesce.
    assert right_inverse(Layout((M, N), (DM, DN))) == Layout(1, 0)
    assert right_inverse(Layout((N, 4), (1, N))) == Layout(4*N, 1)

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
