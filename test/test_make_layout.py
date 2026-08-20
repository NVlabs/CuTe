# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.make_layout, pycute.tiler_to_layout,
pycute.make_layout_like, and pycute.make_ordered_layout

These tests are also worked examples for docs/03_layout.md.
"""

import logging
import pytest
import sympy

from pycute import *

logger = logging.getLogger()


class TestMakeLayout:
  """`make_layout` concatenates a sequence of layouts. Each input layout
  becomes one mode of the result; the result's shape and stride are the
  concatenation of the inputs' shapes and strides."""

  def test_make_layout_basic(self):
    a = Layout(3, 1)
    b = Layout(4, 3)
    assert make_layout([a, b]) == Layout((3, 4), (1, 3))

  def test_make_layout_hierarchical(self):
    """A nested `make_layout` produces a hierarchical layout."""
    L = make_layout([Layout(3, 1),
                     make_layout([Layout(2, 1), Layout(4, 6)]),
                     Layout(2, 42)])
    assert L == Layout((3, (2, 4), 2), (1, (1, 6), 42))

  def test_make_layout_round_trip(self):
    """A layout can be deconstructed via `__getitem__` and reassembled
    via `make_layout`."""
    A = Layout((3, (2, 4)), (2, (1, 6)))
    rebuilt = make_layout([A[0],
                           make_layout([A[1][0], A[1][1]])])
    assert A == rebuilt

  def test_make_layout_of_nothing_is_trivial(self):
    assert make_layout([])       == Layout((), ())
    assert make_layout(iter([])) == Layout((), ())   # a generator, as callers pass
    assert rank(make_layout([]))  == 0
    assert size(make_layout([]))  == 1
    assert coalesce(make_layout([])) == Layout(1, 0)  # the algebra normalizes it

    A = Layout((3, 4), (4, 1))
    assert make_layout([A, make_layout([])]) == Layout(((3, 4), ()), ((4, 1), ()))
    assert all(make_layout([A, make_layout([])])(c, 0) == A(c) for c in range(size(A)))


class TestTilerToLayout:
  """`tiler_to_layout` converts a "tiler" (an int, tuple of tilers, or
  Layout) to a `Layout` that acts identically under composition."""

  def test_tiler_int(self):
    assert tiler_to_layout(3) == Layout(3, 1)

  def test_tiler_layout_is_identity(self):
    L = Layout((7, 2), (3, 1))
    assert tiler_to_layout(L) == L

  def test_tiler_shape_to_coord_layout(self):
    """A `Shape` becomes a coordinate-strided layout — basis vectors
    in each mode."""
    assert tiler_to_layout((4, 5)) == Layout((4, 5), (E(0), E(1)))
    assert tiler_to_layout((2, 3, 5)) == Layout((2, 3, 5), (E(0), E(1), E(2)))

  def test_tiler_tuple_of_layouts(self):
    """A tuple of layouts becomes a rank-2 coord layout where each
    sub-layout's stride is scaled by the matching basis."""
    assert tiler_to_layout((Layout(4, 2), Layout(5, 3))) == Layout((4, 5), (2 * E(0), 3 * E(1)))

  def test_tiler_invariant_under_composition(self):
    """The defining post-condition: composition with a tiler equals
    composition with `tiler_to_layout(tiler)` for every admissible `A`."""
    A = Layout((12, (4, 8)), (59, (13, 1)))
    for tiler in [(3, 8),
                  (Layout(3, 4), Layout(8, 1))]:
      assert composition(A, tiler) == composition(A, tiler_to_layout(tiler))

  def test_zipped_divide_equals_logical_divide_with_tiler(self):
    """`zipped_divide(A, B) == logical_divide(A, tiler_to_layout(B))`."""
    A = Layout((12, 8))
    for B in [(3, 2),
              (Layout(3, 4), Layout(2, 1))]:
      assert zipped_divide(A, B) == logical_divide(A, tiler_to_layout(B))


class TestMakeLayoutLike:
  """`make_layout_like` builds a compact layout with the same shape as the
  input whose strides follow the ordering induced by the input's strides."""

  def postcondition_make_layout_like(self, L):
    """Verify the structural post-conditions of `make_layout_like` for a
    layout with concrete (static) shape and stride."""
    result = make_layout_like(L)

    logger.info(f"  {L}  =>  {result}")

    # Post-condition: the shape is preserved exactly (including hierarchy).
    assert shape(result) == shape(L)

    # Post-condition: idempotence -- the result is already in canonical form.
    assert make_layout_like(result) == result

    src = list(leaves(L.stride))
    dst = list(leaves(result.stride))
    shp = list(leaves(L.shape))

    # Post-condition: a static stride-0 mode carries no information
    # and is pinned to stride 0; every other mode is non-zero.
    for sd, dd in zip(src, dst):
      if sd == 0:
        assert dd == 0
      else:
        assert dd != 0

    # Post-condition: the non-trivial modes form a compact (densely-packed)
    # layout whose strides are the prefix products of the shapes taken in
    # ascending order of the *source* stride -- i.e. the relative ordering of
    # the source's non-zero strides is preserved.
    modes = [(s, sd, dd) for s, sd, dd in zip(shp, src, dst) if not (s == 1 or sd == 0)]
    modes.sort(key=lambda t: t[1])           # stable sort by source stride
    current = 1
    for s, sd, dd in modes:
      assert dd == current
      current *= s

    # Post-condition: the codomain of the non-broadcast modes is exactly the
    # contiguous range [0, cosize) -- no gaps, no overlaps.
    image = sorted({result(i) for i in range(size(result))})
    assert image == list(range(len(image)))

    return result

  #########################################################

  def test_already_compact_is_identity(self):
    """A compact (generalized column-major) layout is returned unchanged."""
    assert make_layout_like(Layout(8, 1)) == Layout(8, 1)
    assert make_layout_like(Layout((4, 8), (1, 4))) == Layout((4, 8), (1, 4))
    assert make_layout_like(Layout((2, 3, 4), (1, 2, 6))) == Layout((2, 3, 4), (1, 2, 6))
    assert make_layout_like(Layout((3, 4), (4, 1))) == Layout((3, 4), (4, 1))

  def test_reorders_to_compact(self):
    """Non-compact strides are repacked while preserving their relative order."""
    assert make_layout_like(Layout((4, 8), (100, 1))) == Layout((4, 8), (8, 1))
    assert make_layout_like(Layout((2, 3, 4), (1, 100, 20))) == Layout((2, 3, 4), (1, 8, 2))
    assert make_layout_like(Layout((3, 4, 5), (1, 100, 10))) == Layout((3, 4, 5), (1, 15, 3))

  def test_cute_documented_examples(self):
    """The two examples documented on CuTe's `make_layout_like`."""
    assert make_layout_like(Layout((2, 2, 2, 2), (0, 2, 4, 1))) == Layout((2, 2, 2, 2), (0, 2, 4, 1))
    assert make_layout_like(Layout((2, 3, 4, 5), (0, 42, 1, 0))) == Layout((2, 3, 4, 5), (0, 4, 1, 0))

  def test_stride_zero_preserved(self):
    """Static stride-0 (broadcast) modes stay stride-0 in the result."""
    assert make_layout_like(Layout(1, 0)) == Layout(1, 0)
    assert make_layout_like(Layout(8, 0)) == Layout(8, 0)
    assert make_layout_like(Layout((3, 7), (0, 0))) == Layout((3, 7), (0, 0))
    assert make_layout_like(Layout((8, 4), (0, 2))) == Layout((8, 4), (0, 1))
    assert make_layout_like(Layout((8, 4, 6), (1, 0, 2))) == Layout((8, 4, 6), (1, 0, 8))

  def test_size_one_modes(self):
    """Size-1 modes carry no information and become stride-0."""
    assert make_layout_like(Layout((1, 4), (7, 2))) == Layout((1, 4), (4, 1))
    assert make_layout_like(Layout((4, 1, 8), (1, 5, 4))) == Layout((4, 1, 8), (1, 32, 4))
    assert make_layout_like(Layout((1, 1), (5, 7))) == Layout((1, 1), (1, 1))

  def test_nested_shape(self):
    """Hierarchical shapes are preserved and packed across all leaves."""
    assert make_layout_like(Layout(((2, 2), (2, 2)), ((1, 4), (8, 32)))) == Layout(((2, 2), (2, 2)), ((1, 2), (4, 8)))
    assert make_layout_like(Layout((2, (3, 4)), (50, (1, 8)))) == Layout((2, (3, 4)), (12, (1, 3)))

  def test_postconditions(self):
    """Battery of layouts exercised against the structural post-conditions."""
    layouts = [
      Layout(8, 1),
      Layout(8, 3),
      Layout((2, 4), (1, 2)),
      Layout((2, 4), (4, 1)),
      Layout((2, 4), (1, 4)),
      Layout((8, 4), (1, 8)),
      Layout((8, 4), (4, 1)),
      Layout((2, 4, 6), (1, 2, 8)),
      Layout((2, 4, 6), (4, 1, 8)),
      Layout((2, 4, 8), (8, 1, 64)),
      Layout((2, 4, 8), (32, 0, 2)),
      Layout((2, 4, 8), (2, 0, 32)),
      Layout((2, 4, 4, 4, 2), (32, 0, 2, 0, 512)),
      Layout(((2, 2), (2, 2)), ((1, 4), (8, 32))),
      Layout((2, (3, 4)), (3, (1, 6))),
      Layout((4, 2), (1, 16)),
      Layout((1, 4), (7, 2)),
      Layout((3, 7), (0, 0)),
    ]
    for layout in layouts:
      self.postcondition_make_layout_like(layout)

  def test_sympy(self):
    """Symbolic strides cannot be ordered by magnitude, so they are considered
    larger than every static stride and kept in left-to-right order. Symbolic
    shapes flow through as compact (symbolic) strides."""
    N, M, X, Y = sympy.symbols("N M X Y", positive=True, integer=True)

    # A static stride orders before a symbolic one; the symbolic mode then
    # receives the running (symbolic) product as its stride.
    assert make_layout_like(Layout((N, 4), (1, N))) == Layout((N, 4), (1, N))
    assert make_layout_like(Layout((4, N), (1, 4))) == Layout((4, N), (1, 4))

    # Static strides (1, 2) order first by magnitude; the symbolic stride X is
    # treated as large and lands last.
    assert make_layout_like(Layout((2, N, 4), (1, X, 2))) == Layout((2, N, 4), (1, 8, 2))

    # Two symbolic strides retain their original left-to-right order.
    assert make_layout_like(Layout((N, M), (X, Y))) == Layout((N, M), (1, N))

  def test_sympy_substitution(self):
    """The symbolic result must agree with the concrete result under any
    concrete substitution."""
    # n > 1: a symbolic shape is generic (does not collapse to a size-1 mode),
    # so the symbolic strides substitute to exactly the concrete strides.
    N = sympy.symbols("N", positive=True, integer=True)
    for n in (2, 3, 5):
      sym = make_layout_like(Layout((N, 4), (1, N)))
      con = make_layout_like(Layout((n, 4), (1, n)))
      assert shape(con) == (n, 4)
      # Substitute N -> n into the symbolic strides and compare.
      sym_stride = tuple(int(d.subs(N, n)) if hasattr(d, "subs") else d
                         for d in leaves(sym.stride))
      assert sym_stride == tuple(leaves(con.stride))

  def test_coordinate_strides(self):
    """ArithTuple (coordinate / basis) strides are static and orderable by
    colex (E(0) < E(1) < E(2) < ...), so `make_layout_like` repacks a
    coordinate-strided layout into a compact *integer* layout that follows the
    basis order."""
    # Identity coordinate layout -> generalized column-major.
    assert make_layout_like(Layout((2, 3, 4), (E(0), E(1), E(2)))) == Layout((2, 3, 4), (1, 2, 6))
    # Reversed basis order -> generalized row-major.
    assert make_layout_like(Layout((2, 3, 4), (E(2), E(1), E(0)))) == Layout((2, 3, 4), (12, 4, 1))
    # An arbitrary basis permutation.
    assert make_layout_like(Layout((2, 3, 4), (E(1), E(2), E(0)))) == Layout((2, 3, 4), (4, 8, 1))
    # The basis *position* drives the order; a scale only breaks ties between
    # strides sharing the same position.
    assert make_layout_like(Layout((2, 3), (2 * E(0), 3 * E(1)))) == Layout((2, 3), (1, 2))
    assert make_layout_like(Layout((4, 3), (3 * E(0), 1 * E(0)))) == Layout((4, 3), (3, 1))
    # rank-1 coordinate layouts collapse to the single contiguous mode.
    assert make_layout_like(Layout(4, E(0))) == Layout(4, 1)
    assert make_layout_like(Layout(4, E(1))) == Layout(4, 1)

  def test_coordinate_strides_with_broadcast(self):
    """A static stride-0 (broadcast) mode interleaved with coordinate strides
    stays stride-0; the coordinate modes are still packed in basis order."""
    assert make_layout_like(Layout((2, 3, 4), (E(0), 0, E(1)))) == Layout((2, 3, 4), (1, 0, 2))
    assert make_layout_like(Layout((2, 3), (0, E(0)))) == Layout((2, 3), (0, 1))

  def test_coordinate_strides_nested(self):
    """Hierarchical coordinate strides preserve the shape and pack across all
    leaves in colex order of the basis paths."""
    assert make_layout_like(Layout((2, (3, 4)), (E(0), (E(1), E(2))))) == Layout((2, (3, 4)), (1, (2, 6)))
    assert make_layout_like(Layout((2, 3, 4), (E(0, 0), E(0, 1), E(1)))) == Layout((2, 3, 4), (1, 2, 6))

  def test_coordinate_strides_postconditions(self):
    """The structural post-conditions hold for coordinate-strided layouts too:
    a coordinate layout becomes a compact integer layout of the same shape."""
    layouts = [
      Layout(4, E(0)),
      Layout((2, 3, 4), (E(0), E(1), E(2))),
      Layout((2, 3, 4), (E(2), E(1), E(0))),
      Layout((2, 3, 4), (E(1), E(2), E(0))),
      Layout((2, 3), (2 * E(0), 3 * E(1))),
      Layout((4, 3), (3 * E(0), 1 * E(0))),
      Layout((2, 3, 4), (E(0), 0, E(1))),
      Layout((2, (3, 4)), (E(0), (E(1), E(2)))),
      Layout((2, 3, 4), (E(0, 0), E(0, 1), E(1))),
    ]
    for layout in layouts:
      self.postcondition_make_layout_like(layout)


class TestMakeOrderedLayout:
  """`make_ordered_layout(shape, order)` builds a compact layout of the given
  `shape` whose modes are packed (stride 1, then prefix products) in ascending
  order of `order`. Only the relative ordering of `order` matters."""

  def postcondition_make_ordered_layout(self, shp, order):
    """Verify the structural post-conditions of `make_ordered_layout` for a
    static (concrete) shape and order."""
    result = make_ordered_layout(shp, order)

    logger.info(f"  {shp}, {order}  =>  {result}")

    # Post-condition: the shape is preserved exactly (including hierarchy).
    assert shape(result) == shape(shp)

    # Post-condition: the result is compact -- its codomain is exactly the
    # contiguous range [0, size) with no gaps and no overlaps.
    image = sorted({result(i) for i in range(size(result))})
    assert image == list(range(size(result)))

    # Post-condition: the modes receive prefix-product strides taken in
    # ascending order of `order`, with ties broken by left-to-right position.
    shp_l = list(leaves(shp))
    ord_l = list(leaves(order))
    dst   = list(leaves(result.stride))
    current = 1
    for o, i in sorted(zip(ord_l, range(len(ord_l)))):  # stable: ties keep position
      assert dst[i] == current
      current *= shp_l[i]

    return result

  #########################################################

  def test_column_and_row_major(self):
    """An ascending order is column-major; a descending order is row-major."""
    assert make_ordered_layout(8, 0) == Layout(8, 1)
    assert make_ordered_layout((4, 8), (0, 1)) == Layout((4, 8), (1, 4))
    assert make_ordered_layout((4, 8), (1, 0)) == Layout((4, 8), (8, 1))
    assert make_ordered_layout((2, 3, 4), (0, 1, 2)) == Layout((2, 3, 4), (1, 2, 6))
    assert make_ordered_layout((2, 3, 4), (2, 1, 0)) == Layout((2, 3, 4), (12, 4, 1))

  def test_cute_documented_example(self):
    """The static example documented on CuTe's `make_ordered_layout`:
    `make_ordered_layout(Shape<_2,_2,_2,_2>, Step<_0,_2,_3,_1>)`."""
    assert make_ordered_layout((2, 2, 2, 2), (0, 2, 3, 1)) == Layout((2, 2, 2, 2), (1, 4, 8, 2))

  def test_permutation(self):
    """An arbitrary permutation of mode priorities."""
    assert make_ordered_layout((2, 3, 4), (2, 0, 1)) == Layout((2, 3, 4), (12, 1, 3))

  def test_order_is_relative(self):
    """Only the relative ordering of the values matters, not their magnitudes,
    so `order` need not be a contiguous `0..rank-1` permutation."""
    assert make_ordered_layout((4, 8), (5, 7)) == make_ordered_layout((4, 8), (0, 1))
    assert make_ordered_layout((2, 3, 4, 5), (2, 67, 42, 50)) == Layout((2, 3, 4, 5), (1, 40, 2, 8))

  def test_ties_break_by_position(self):
    """Equal order values keep their left-to-right position order, independent
    of the shapes involved."""
    assert make_ordered_layout((4, 8), (5, 5)) == Layout((4, 8), (1, 4))
    assert make_ordered_layout((8, 4), (5, 5)) == Layout((8, 4), (1, 8))
    assert make_ordered_layout((2, 3, 4, 2), (0, 2, 3, 0))  == Layout((2, 3, 4, 2), (1, 4, 12, 2))

  def test_nested_shape(self):
    """Hierarchical shapes/orders are preserved and packed across all leaves
    in the global order of the flattened `order`."""
    assert make_ordered_layout(((2, 2), (2, 2)), ((0, 1), (2, 3))) == Layout(((2, 2), (2, 2)), ((1, 2), (4, 8)))
    assert make_ordered_layout((2, (3, 4)), (0, (1, 2))) == Layout((2, (3, 4)), (1, (2, 6)))
    assert make_ordered_layout((2, (3, 4)), (2, (1, 0))) == Layout((2, (3, 4)), (12, (4, 1)))

  def test_size_one_modes(self):
    """Size-1 modes carry no positional information; the running product is
    unchanged across them so the remaining modes stay densely packed."""
    assert make_ordered_layout((1, 4), (0, 1)) == Layout((1, 4), (1, 1))
    assert make_ordered_layout((4, 1, 8), (0, 1, 2)) == Layout((4, 1, 8), (1, 4, 4))

  def test_not_congruent_raises(self):
    """`order` must be congruent to `shape`."""
    with pytest.raises(ValueError):
      make_ordered_layout((4, 8), (0, 1, 2))
    with pytest.raises(ValueError):
      make_ordered_layout((4, (8, 2)), (0, 1))
    with pytest.raises(ValueError):
      make_ordered_layout(8, (0, 1))

  def test_postconditions(self):
    """Battery of shape/order pairs exercised against the structural
    post-conditions."""
    cases = [
      (8, 0),
      ((4, 8), (0, 1)),
      ((4, 8), (1, 0)),
      ((2, 3, 4), (0, 1, 2)),
      ((2, 3, 4), (2, 1, 0)),
      ((2, 3, 4), (2, 0, 1)),
      ((2, 2, 2, 2), (0, 2, 3, 1)),
      ((2, 3, 4, 5), (2, 67, 42, 50)),
      ((4, 8), (5, 5)),
      ((8, 4), (5, 5)),
      (((2, 2), (2, 2)), ((0, 1), (2, 3))),
      ((2, (3, 4)), (0, (1, 2))),
      ((2, (3, 4)), (2, (1, 0))),
      ((1, 4), (0, 1)),
      ((4, 1, 8), (0, 1, 2)),
    ]
    for shp, order in cases:
      self.postcondition_make_ordered_layout(shp, order)

  def test_sympy(self):
    """Symbolic (non-static) order values cannot be ordered by magnitude, so
    they are considered larger than every static order and retain their
    left-to-right order -- mirroring CuTe, where dynamic `order` values are
    'considered large and ordered from left to right'."""
    P, Q = sympy.symbols("P Q", positive=True, integer=True)

    # A static order sorts before a symbolic one.
    assert make_ordered_layout((4, 8), (1, P)) == Layout((4, 8), (1, 4))
    assert make_ordered_layout((4, 8), (P, 1)) == Layout((4, 8), (8, 1))

    # Two symbolic orders keep their original left-to-right order.
    assert make_ordered_layout((4, 8), (P, Q)) == Layout((4, 8), (1, 4))

    # Static orders (2, 50) place first and last by magnitude; the symbolic
    # orders land between them, in left-to-right order. This reproduces CuTe's
    # documented `make_step(Int<2>, 67, 42, Int<50>)` example.
    assert make_ordered_layout((2, 3, 4, 5), (2, P, Q, 50)) == Layout((2, 3, 4, 5), (1, 10, 30, 2))

    # A static-0 order interleaved with symbolic orders still sorts first.
    assert make_ordered_layout((2, 3, 4), (P, 0, Q)) == Layout((2, 3, 4), (3, 1, 6))
