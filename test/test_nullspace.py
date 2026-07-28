# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.nullspace
"""

import logging

from pycute import *

logger = logging.getLogger()


class TestRightInverse:
  def postcondition_nullspace(self, L):
    null_layout = nullspace(L)

    logger.info(f"  {L}  =>  {null_layout}")

    # Generalized right inverse condition
    for i in range(size(null_layout)):
      assert L(null_layout(i)) == 0

  def test_nullspace(self):
    self.postcondition_nullspace(Layout(1,0))
    self.postcondition_nullspace(Layout(1,1))
    self.postcondition_nullspace(Layout(1,2))
    self.postcondition_nullspace(Layout(1,4))
    self.postcondition_nullspace(Layout(4,0))
    self.postcondition_nullspace(Layout(4,1))
    self.postcondition_nullspace(Layout(4,2))
    self.postcondition_nullspace(Layout(4,4))
    self.postcondition_nullspace(Layout((1,1),(0,0)))
    self.postcondition_nullspace(Layout((3,7),(0,0)))
    self.postcondition_nullspace(Layout((2,4),(0,2)))
    self.postcondition_nullspace(Layout((8,4),(1,8)))
    self.postcondition_nullspace(Layout((8,4),(4,1)))
    self.postcondition_nullspace(Layout((2,4,6),(1,2,0)))
    self.postcondition_nullspace(Layout((2,4,6),(0,1,0)))
    self.postcondition_nullspace(Layout((4,2),(1,16)))
    self.postcondition_nullspace(Layout((4,5,6),(1,1,0)))
    self.postcondition_nullspace(Layout((7,5,9),(2,0,1)))
    self.postcondition_nullspace(Layout((7,5,9),(2,0,0)))
    self.postcondition_nullspace(Layout((7,5,9),(0,0,0)))

  def test_nullspace_coord(self):
    self.postcondition_nullspace(Layout((4,5),(E(0),E(1))))
    self.postcondition_nullspace(Layout((4,5),(0,E(0))))
    self.postcondition_nullspace(Layout((4,5),(0,E(4,1))))
    self.postcondition_nullspace(Layout((4,5),(2*E(0),0)))

    # SM70 MMA 8x8x4 C TV inverse
    self.postcondition_nullspace(Layout(((   2,      2,      2), (   2,      2,      2)),
                                        ((E(0), 2*E(1), 4*E(0)), (E(1), 2*E(0), 4*E(1)))))
    self.postcondition_nullspace(Layout(((   2,      2,      2), (   2,      2,      2)),
                                        ((E(0), 2*E(1), 5*E(0)), (E(1), 0, 5*E(1)))))
    self.postcondition_nullspace(Layout(((   2,      2,      2), (   2,      2,      2)),
                                        ((E(0), 0, 5*E(0)), (E(1), 2*E(0), 4*E(1)))))

    # SM70 MMA 8x8x4 A TV inverse
    self.postcondition_nullspace(composition(tiler_to_layout((8,4)),
                                             Layout(((4,2),4), ((8,4),0))))

    # SM80 MMA 16x8 TV inverse
    self.postcondition_nullspace(composition(tiler_to_layout((16,8)),
                                             Layout(((4,8),(2,2)), ((0,1),(16,8)))))
