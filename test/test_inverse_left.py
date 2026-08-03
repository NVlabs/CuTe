# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.left_inverse
"""

import logging

import pytest

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
