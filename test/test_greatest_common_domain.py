# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.greatest_common_domain

These tests are also worked examples for docs/04_layout_algebra.md.
"""

import logging
import math

from pycute import *

logger = logging.getLogger()


class TestGreatestCommonDomain:

  def postcondition_greatest_common_domain(self, A, B):
    R = greatest_common_domain(A, B)

    logger.info(f"  gcd({A}, {B})  =>  {R}")

    # Post-condition: result is a Layout
    assert is_layout(R)

    # Post-condition: depth is 1
    assert depth(R) == 1

    # Post-condition: size(R) | size(A) and size(R) | size(B)
    assert size(A) % size(R) == 0
    assert size(B) % size(R) == 0

    # Post-condition: size(R) | gcd(size(A), size(B))
    assert math.gcd(size(A), size(B)) % size(R) == 0

    # Post-condition: GCD(A,B) | A and GCD(A,B) | B
    assert is_layout(composition(shape(A), R))
    assert is_layout(composition(shape(B), R))

    # Post-condition: symmetric in A and B
    assert R == greatest_common_domain(B, A), f"asymmetric: gcd({A},{B})={R}  gcd({B},{A})={greatest_common_domain(B,A)}"

    # Post-condition: GCD(A // GCD(A,B), B // GCD(A,B)) == 1
    RA = logical_divide(shape(A), R)
    RB = logical_divide(shape(B), R)
    assert size(greatest_common_domain(RA[1], RB[1])) == 1

    return R

  #########################################################

  def test_singleton(self):
    # The empty / singleton case
    assert self.postcondition_greatest_common_domain(1, 1) == Layout((1,), (0,))

  def test_int_vs_int(self):
    # Same int -- full common domain
    R = self.postcondition_greatest_common_domain(10, 10)
    assert size(R) == 10

    # Coprime ints -- size 1
    R = self.postcondition_greatest_common_domain(7, 9)
    assert size(R) == 1

    # Shared prime factors
    R = self.postcondition_greatest_common_domain(12, 18)
    assert size(R) == 6

  def test_int_vs_tuple_coerces_via_shape(self):
    # `10` and `(10,)` have the same shape via flatten(shape(...))
    R1 = greatest_common_domain(10, 10)
    R2 = greatest_common_domain((10,), (10,))
    assert R1 == R2

  def test_known_expected_results(self):
    # Hand-checked expected results for canonical examples
    expected = [
      ((1,        1),               Layout((1,), (0,))),
      ((10,       10),              Layout((10,),    (1,))),
      (((16, 3),  (16, 3)),         Layout((16, 3),  (1, 16))),
      (((5, 2),   10),              Layout((5, 2),   (1, 5))),
      (((5, 3, 4), (10, 6)),        Layout((5, 2),   (1, 30))),
      (((5, 3, 3, 4), (10, 6, 3)),  Layout((5,),     (1,))),
      (((1, 5, 3, 3, 4), (10, 6, 3)), Layout((5,),   (1,))),
      (((5, 3, 3, 4), (10, 6, 1, 3)), Layout((5,),   (1,))),
      (((2, 21),  (3, 14)),         Layout((7,),     (6,))),
      (((6, 35),  (15, 14)),        Layout((3, 7),   (1, 30))),
      (((16, 64), (4, 16, 16)),     Layout((4, 4, 4, 16), (1, 4, 16, 64))),
      (((5, 3),   (3, 5)),          Layout((1,), (0,))),
      (((5, 5, 3), (5, 3, 5)),      Layout((5,),     (1,))),
      (((7, 2, 3, 2, 2, 3),     (504,)),
                                    Layout((7, 2, 3, 2, 2, 3), (1, 7, 14, 42, 84, 168))),
      (((7, 2, 3, 2, 2, 3, 2),  (7, 3, 2, 2, 3, 2, 2)),
                                    Layout((7, 2, 2), (1, 42, 504))),
      (((7, 2, 2, 2, 2, 3, 5),  (7, 3, 2, 2, 2, 2, 5)),
                                    Layout((7, 5),    (1, 336))),
    ]
    for (A, B), expected_R in expected:
      R = self.postcondition_greatest_common_domain(A, B)
      assert R == expected_R, f"gcd({A}, {B}) = {R}  expected {expected_R}"

  def test_coprime_leaves_yield_singleton(self):
    # Order-aligned but pairwise coprime leaves yield the trivial singleton
    assert self.postcondition_greatest_common_domain((5, 3), (3, 5)) == Layout((1,), (0,))
    assert self.postcondition_greatest_common_domain((7, 11), (11, 7)) == Layout((1,), (0,))

  def test_singleton_leaves_are_skipped(self):
    # Leading / interior `1`s in either shape are transparent to the walk
    R1 = greatest_common_domain((5, 3, 3, 4),    (10, 6, 3))
    R2 = greatest_common_domain((1, 5, 3, 3, 4), (10, 6, 3))
    R3 = greatest_common_domain((5, 3, 3, 4),    (10, 6, 1, 3))
    R4 = greatest_common_domain((5, 1, 1, 3, 1, 3, 4), (10, 6, 3))
    assert R1 == R2
    assert R1 == R3
    assert R1 == R4

  def test_depends_only_on_shape(self):
    # Result is determined by `shape(...)` only; input strides are ignored.
    L1 = Layout((5, 3, 4), (1, 5, 15))      
    L2 = Layout((5, 3, 4), (12, 4, 1))      # Different stride, same shape
    L3 = Layout((5, 3, 4), (0, 0, 0))       # Degenerate strides
    B  = Layout((10, 6),   (1, 10))

    R1 = greatest_common_domain(L1, B)
    R2 = greatest_common_domain(L2, B)
    R3 = greatest_common_domain(L3, B)
    R_shape = greatest_common_domain((5, 3, 4), (10, 6))
    assert R1 == R_shape
    assert R2 == R_shape
    assert R3 == R_shape

  def test_nested(self):
    # Nested shapes share the same leaf sequence
    R_nested = greatest_common_domain((2, (3, 4)), (6, 4))
    R_flat   = greatest_common_domain((2, 3, 4),   (24,))
    assert R_nested == R_flat
