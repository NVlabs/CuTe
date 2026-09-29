# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.left_inverse
"""

import logging

import pytest
import sympy

from pycute import *

logger = logging.getLogger()


class TestLeftInverse:
  def postcondition_left_inverse(self, L):
    inv_layout = left_inverse(L)

    logger.info(f"  {L}  =>  {inv_layout}")

    assert weakly_congruent(coprofile(L), shape(inv_layout))

    # Generalized left inverse condition
    for i in range(size(L)):
      assert L(inv_layout(L(i))) == L(i)


  def test_left_inverse_f2(self):
    """An `F2` stride whose chain gap is 1 leaves the walk in integer arithmetic,
    so those cases work."""
    self.postcondition_left_inverse(Layout(8,F2(1)))
    self.postcondition_left_inverse(Layout((4,8),(F2(1),F2(4))))

  def test_left_inverse_f2_non_unit_gap_raises(self):
    """A gap between strides becomes an extent, and `F2`'s stride quotient is a
    carry-less one -- not an `Integer`. Rather than return a layout whose shape
    holds an `F2`, `left_inverse` rejects it."""
    for L in [Layout(8,F2(2)), Layout((8,8),(F2(1),F2(9))), Layout((4,4),(F2(1),F2(5)))]:
      with pytest.raises(ValueError):
        left_inverse(L)

  def test_left_inverse(self):
    self.postcondition_left_inverse(Layout(1,0))
    self.postcondition_left_inverse(Layout(1,1))
    self.postcondition_left_inverse(Layout(1,2))
    self.postcondition_left_inverse(Layout(1,4))
    self.postcondition_left_inverse(Layout((1,1),(0,0)))
    self.postcondition_left_inverse(Layout((3,7),(0,0)))
    self.postcondition_left_inverse(Layout(4,0))
    self.postcondition_left_inverse(Layout(4,1))
    self.postcondition_left_inverse(Layout(4,2))
    self.postcondition_left_inverse(Layout(4,4))
    self.postcondition_left_inverse(Layout((8,4),(1,8)))
    self.postcondition_left_inverse(Layout((8,4),(4,1)))
    self.postcondition_left_inverse(Layout((2,4,6),(1,2,8)))
    self.postcondition_left_inverse(Layout((2,4,6),(4,1,8)))
    self.postcondition_left_inverse(Layout((2,4,8), (32,0,2)))
    self.postcondition_left_inverse(Layout((2,4,8), (2,0,32)))
    self.postcondition_left_inverse(Layout((2,4,4,4,2), (32,0,2,0,512)))
    self.postcondition_left_inverse(Layout((4,2),(1,16)))
    self.postcondition_left_inverse(Layout((4,2),(1,5)))
    self.postcondition_left_inverse(Layout((4,2),(1,10)))
    self.postcondition_left_inverse(Layout((4,2),(1,11)))

    # TMEM inspired
    self.postcondition_left_inverse(Layout((32,8), (65536,1)))
    self.postcondition_left_inverse(Layout((32,12), (65536,1)))
    self.postcondition_left_inverse(Layout((32,3,8), (65536,512,1)))
    self.postcondition_left_inverse(Layout((32,8), (131072,2)))
    self.postcondition_left_inverse(Layout((((((     2, 4), 1), (2, 2)),       4), 1, (2,  2),  2),
                                           (((((262144, 4), 0), (0, 1)), 8388608), 0, (2, 16), 32)))


  def test_left_inverse_raises(self):
    # left_inverse only handles layouts whose nonzero strides form an ordered
    # chain (each stride divides the next and clears the previous mode's span).

    # Overlapping/repeated nonzero strides => the layout is non-injective.
    with pytest.raises(ValueError, match="(?i)non-injective"):
      left_inverse(Layout((63,2), (1,1)))
    with pytest.raises(ValueError, match="(?i)non-injective"):
      left_inverse(Layout((2,2), (1,1)))
    with pytest.raises(ValueError, match="(?i)non-injective"):
      left_inverse(Layout((2,3), (2,1)))

    # Coprime (non-divisible) strides => injective but unordered, so rejected as
    # a deliberate simplification even though a layout left inverse exists.
    with pytest.raises(ValueError, match="(?i)ordered chain"):
      left_inverse(Layout((2,2), (2,3)))
    with pytest.raises(ValueError, match="(?i)ordered chain"):
      left_inverse(Layout((2,2), (2,5)))


  def test_left_inverse_coord(self):
    self.postcondition_left_inverse(Layout((4,5),(E(0),E(1))))
    self.postcondition_left_inverse(Layout((4,5),(E(1),E(0))))
    self.postcondition_left_inverse(Layout((4,5),(E(1),E(4,1))))
    self.postcondition_left_inverse(Layout((4,5),(2*E(0),2*E(1))))
    self.postcondition_left_inverse(Layout((3,(2,2)), (34*E(0), (2*E(0), 2*E(1)))))

    # SM70 MMA 8x8x4 C TV inverse
    self.postcondition_left_inverse(Layout(((   2,      2,      2), (   2,      2,      2)),
                                           ((E(0), 2*E(1), 4*E(0)), (E(1), 2*E(0), 4*E(1)))))
    self.postcondition_left_inverse(Layout(((   2,      2,      2), (   2,      2,      2)),
                                           ((E(0), 2*E(1), 6*E(0)), (E(1), 2*E(0), 6*E(1)))))
    self.postcondition_left_inverse(Layout(((   2,      2,      2), (   2,      2,      2)),
                                           ((E(0), 2*E(1), 6*E(0)), (E(1), 2*E(0), 4*E(1)))))

    # SM70 MMA 8x8x4 A TV inverse
    self.postcondition_left_inverse(composition(tiler_to_layout((8,4)),
                                                Layout(((4,2),4), ((8,4),1))))

    # SM80 MMA 16x8 TV inverse
    self.postcondition_left_inverse(composition(tiler_to_layout((16,8)),
                                                Layout(((4,8),(2,2)), ((32,1),(16,8)))))


  def specialize(self, L, values):
    """`L` with each symbolic leaf replaced by its value in `values`."""
    concrete = lambda x: int(x.subs(values)) if hasattr(x, "subs") else x
    return Layout(transform_leaf(concrete, shape(L)),
                  transform_leaf(concrete, stride(L)))

  def test_left_inverse_sympy(self):
    # A left inverse has to invert the whole domain, so every mode it records
    # must be *shown* to continue the chain. With symbolic extents that is
    # decidable whenever one stride is concrete or the two are related, and the
    # gap between consecutive strides becomes an extent whose entries are holes.
    N, M, X = sympy.symbols("N M X", positive=True, integer=True)

    assert left_inverse(Layout(N, 1)) == Layout(N, 1)
    assert left_inverse(Layout((4, N), (1, 4))) == Layout(4*N, 1)
    assert left_inverse(Layout((N, 4), (1, N))) == Layout(4*N, 1)
    assert left_inverse(Layout((N, M), (1, N))) == Layout(N*M, 1)

    # Stride order reversed: the concrete stride is walked first, and the
    # symbolic one then continues the chain because M / 1 is exact.
    assert left_inverse(Layout((N, M), (M, 1))) == Layout((M, N), (N, 1))

    # A gap becomes a padded mode of holes, at stride 0.
    assert left_inverse(Layout(N, X)) == Layout((X, N), (0, 1))
    assert left_inverse(Layout((N, M), (1, 2*N))) == Layout((2*N, M), (1, N))

    # A stride-0 mode carries no information and is skipped.
    assert left_inverse(Layout(N, 0)) == Layout(1, 0)
    assert left_inverse(Layout((N, M), (0, N))) == Layout((N, M), (0, N))

  def test_left_inverse_sympy_refuses_what_it_cannot_show(self):
    # Unlike `right_inverse`, there is no smaller-but-still-valid left inverse
    # to fall back on, so a chain that cannot be *shown* to hold is refused
    # rather than assumed.
    N, M, X, DM, DN = sympy.symbols("N M X DM DN", positive=True, integer=True)

    # `DN / DM` cannot be shown to be exact, so the chain cannot continue.
    with pytest.raises(ValueError, match="(?i)ordered chain"):
      left_inverse(Layout((4, N), (DM, DN)))

    # The division *is* exact here, but whether the second mode clears the
    # first -- `X >= N` for two unrelated symbols -- is not decidable, so it is
    # refused rather than assumed. This is the case that distinguishes
    # demanding proof from deferring: deferring would return an inverse that
    # is wrong whenever X < N.
    with pytest.raises(ValueError, match="(?i)non-injective"):
      left_inverse(Layout((N, M), (1, X)))

    # ... whereas a stride sympy *can* relate to the extent is accepted.
    assert left_inverse(Layout((N, M), (1, 2*N))) == Layout((2*N, M), (1, N))

    # Symbolic extents do not hide a non-injective layout: two stride-1 modes
    # overlap whatever N is, and that is decidable.
    with pytest.raises(ValueError, match="(?i)non-injective"):
      left_inverse(Layout((N, 2), (1, 1)))

  def test_left_inverse_sympy_specializes(self):
    # The symbolic inverse must be the inverse for *every* value its symbols
    # can take, so substituting concrete extents into both the layout and its
    # symbolic inverse has to satisfy the left inverse post-condition.
    N, M, X = sympy.symbols("N M X", positive=True, integer=True)

    for L in [Layout(N, 1), Layout((4, N), (1, 4)), Layout((N, M), (1, N)),
              Layout((N, M), (M, 1)), Layout((N, M), (0, N)),
              Layout((N, M), (1, 2*N)), Layout(N, X)]:
      L_inv = left_inverse(L)
      for values in ({N: 1, M: 1, X: 1}, {N: 2, M: 3, X: 2},
                     {N: 3, M: 2, X: 5}, {N: 5, M: 4, X: 3}):
        Lc, Lc_inv = self.specialize(L, values), self.specialize(L_inv, values)
        assert weakly_congruent(coprofile(Lc), shape(Lc_inv))
        for i in range(size(Lc)):
          assert Lc(Lc_inv(Lc(i))) == Lc(i)

  def test_left_inverse_app(self):

    # A common cotiling failure
    atom_tv_layout = Layout(((32,       4), (16,    32)),
                            (( 0, 2097152), ( 1, 65536)))
    data_layout = Layout((  128, 16),
                         (65536,  1))
    # data addr -> data coord    Append 1:0 so off-the-ends get the stride-0
    inv_data_layout = make_layout([left_inverse(data_layout), Layout(1,0)])
    # (tid,vid) -> data_coord
    layout_tv_data = composition(inv_data_layout, atom_tv_layout)
    # Check validity   D o (Di o TV) == TV
    assert coalesce(composition(data_layout, layout_tv_data)) == coalesce(atom_tv_layout)
