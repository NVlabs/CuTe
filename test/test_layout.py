# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.layout

These tests are also worked examples for docs/03_layout.md.
"""

import logging
import sympy

from pycute import *

logger = logging.getLogger()


class TestLayout:

  def test_layout(self):
    A = Layout((3,(2,4)),(2,(1,6)))

    logger.info(f"  {A}")

    assert size(A) == 24
    assert A[0] == Layout(3,2)
    assert A[1] == Layout((2,4),(1,6))
    assert A[1][0] == Layout(2,1)

    assert A == make_layout([Layout(3,2), make_layout([Layout(2,1), Layout(4,6)])])

    R = [0, 2, 4, 1, 3, 5, 6, 8,10, 7, 9,11,12,14,16,13,15,17,18,20,22,19,21,23]

    for i in range(size(A[0])):
      for j in range(size(A[1])):
        I = i + j * size(A[0])
        assert R[I] == A(I)
        assert R[I] == A(i,j)


class TestLayoutConstruction:
  """The `Layout(shape, stride=1)` constructor expands a base stride via
  `prefix_product` to give a generalized column-major layout by default."""

  def test_default_stride_is_column_major(self):
    assert Layout(8) == Layout(8, 1)
    assert Layout((4, 8)) == Layout((4, 8), (1, 4))
    assert Layout((3, (2, 4))) == Layout((3, (2, 4)), (1, (3, 6)))

  def test_base_stride_scales(self):
    """`Layout(shape, k)` is `Layout(shape, prefix_product(shape, k))`."""
    assert Layout((4, 8), 2) == Layout((4, 8), (2, 8))
    assert Layout((3, (2, 4)), 5) == Layout((3, (2, 4)), (5, (15, 30)))

  def test_explicit_stride(self):
    """An explicit hierarchical stride is used as-is."""
    assert Layout((4, 8), (8, 1)).stride == (8, 1)
    assert Layout((3, (2, 4)), (24, (1, 6))).stride == (24, (1, 6))


class TestThreeCoordinateForms:
  """A `Layout` accepts integral, R-D flat, and natural h-D coordinates,
  all producing the same offset (Whitepaper, §2.4)."""

  def test_three_coords_agree(self):
    A = Layout((3, (2, 4)), (2, (1, 6)))
    for i in range(size(A)):
      crd_2d = (i % size(A[0]), i // size(A[0]))
      crd_nat = (crd_2d[0],
                 (crd_2d[1] %  shape(A[1])[0],
                  crd_2d[1] // shape(A[1])[0]))
      assert A(i) == A(crd_2d)
      assert A(i) == A(crd_nat)
      assert A(i) == A(*crd_nat)

  def test_call_with_packed_or_unpacked_coord(self):
    """`A(c)` and `A(*c)` agree (Layout.__call__ unwraps a 1-tuple)."""
    A = Layout((3, (2, 4)), (2, (1, 6)))
    assert A((2, (1, 2))) == A(2, (1, 2))

  def test_out_of_bounds_integral_coordinate(self):
    """Extended-domain integral coordinates are well-defined but may leave the image."""
    A = Layout((3, (2, 4)), (2, (1, 6)))
    assert A(100) == 99


class TestLayoutSlicing:
  """Calling `_offset_and_slice(c)` returns an offset (consuming definite
  coordinates) and a residual layout (over `None` positions)."""

  def test_slice_full_coord(self):
    A = Layout((4, 4), (4, 1))
    off, sub = A._offset_and_slice((1, 2))
    assert off == A(1, 2)
    assert rank(sub) == 0            # no remaining modes

  def test_slice_partial_coord_returns_residual_layout(self):
    A = Layout((4, 4), (4, 1))
    off, sub = A._offset_and_slice((1, None))
    assert off == 4                  # row 1 starts at offset 4
    assert sub == Layout((4,), (1,)) # the row layout

    off, sub = A._offset_and_slice((None, 2))
    assert off == 2                  # column 2 starts at offset 2
    assert sub == Layout((4,), (4,)) # the column layout

  def test_slice_all_None_returns_full_layout(self):
    A = Layout((4, 4), (4, 1))
    off, sub = A._offset_and_slice((None, None))
    assert off == 0
    assert sub == A


class TestLayoutEquality:
  """`Layout.__eq__` is structural: same shape and same stride."""

  def test_structural_equality(self):
    assert Layout((3, 4)) == Layout((3, 4), (1, 3))
    assert Layout((3, 4)) != Layout((4, 3))
    assert Layout((3, 4), (1, 3)) != Layout((3, 4), (4, 1))

  def test_functionally_equivalent_layouts_may_differ_structurally(self):
    """`Layout((4, 8), (1, 4))` and `Layout(32, 1)` produce the same
    offsets for every integer in `[0, 32)`, but they are NOT equal."""
    A = Layout((4, 8), (1, 4))
    B = Layout(32, 1)
    assert A != B
    for i in range(32):
      assert A(i) == B(i)


class TestCoshape:
  """`coshape(L)` is a bound on the image of `L`: enough storage to hold
  every offset that `L` produces. `coprofile` is its profile."""

  def test_coshape_integer_stride_is_int(self):
    assert coshape(Layout((4, 8), (1, 4))) == 32
    assert coshape(Layout((4, 8), (8, 1))) == 32
    assert coshape(Layout(8, 1)) == 8
    assert coshape(Layout(8, 0)) == 1        # broadcast

  def test_coshape_coordinate_stride_is_tuple(self):
    assert coshape(Layout((4, 8), (E(0), E(1)))) == (4, 8)
    assert coshape(Layout((4, 8), (E(1), E(0)))) == (8, 4)

  def test_coshape_f2_stride_is_a_power_of_two(self):
    """An `F2` codomain is a space of bit-vectors, not an interval, so its bound
    is the smallest power of two spanning the modes' bit-fields. XOR is not
    monotone, so summing the modes' maxima would under-count."""
    assert coshape(Layout((8, 8), (F2(1), F2(9)))) == 64
    assert coshape(Layout((8, 8), (F2(9), F2(1)))) == 64
    assert coshape(Layout((4, 8), (F2(1), F2(4)))) == 32
    assert coshape(Layout((4, 4), (F2(1), F2(1)))) == 4     # non-injective
    assert coshape(Layout(8, F2(0))) == 1                   # broadcast
    assert coshape(Layout(4, F2(3))) == 8                   # image reaches 6

  def test_coshape_bounds_the_image(self):
    """The defining property, for every codomain: `coshape` holds every offset."""
    for L in [Layout((4, 8), (1, 4)), Layout((4, 8), (8, 1)), Layout(8, 0),
              Layout((8, 8), (F2(1), F2(9))), Layout((8, 8), (F2(9), F2(1))),
              Layout((4, 4), (F2(1), F2(5))), Layout(4, F2(3)),
              Layout((4, 8), (F2(1), 0))]:
      assert int(coshape(L)) > max(int(L(i)) for i in range(size(L)))

  def test_coprofile_is_congruent_to_coshape(self):
    """`coprofile` fixes only the codomain's tuple/leaf structure, so its leaf
    *values* are not the extents -- it is congruent to `coshape`, not equal."""
    for L in [Layout((4, 8), (1, 4)), Layout(8, 0),
              Layout((4, 8), (E(0), E(1))), Layout((4, 5), (E(1), E(4, 1))),
              Layout((8, 8), (F2(1), F2(9)))]:
      assert congruent(coprofile(L), coshape(L))

  def test_coprofile_of_an_f2_layout_is_a_leaf(self):
    """An `F2` codomain is rank-1, like `Z`, so its profile is a leaf."""
    assert congruent(coprofile(Layout((8, 8), (F2(1), F2(9)))), 0)
