# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.htuple

These tests are also worked examples for docs/01_htuple.md.
"""

import pytest

from pycute import *


class TestHTuple:

  def test_is_tuple(self):
    """`is_tuple(x)` is True for both Python tuples and lists."""
    assert is_tuple((1, 2, 3))
    assert is_tuple([1, 2, 3])
    assert is_tuple(())
    assert not is_tuple(7)
    assert not is_tuple(None)

  def test_wrap_unwrap(self):
    """`wrap(x)` makes `x` a 1-tuple if needed; `unwrap` strips 1-tuples
    recursively."""
    assert wrap(7) == (7,)
    assert wrap((7,)) == (7,)
    assert wrap((7, 8)) == (7, 8)

    assert unwrap(42) == 42
    assert unwrap((42,)) == 42
    assert unwrap((((42,)),)) == 42
    assert unwrap((1, 2)) == (1, 2)     # not a 1-tuple

  def test_front_back(self):
    """`front`/`back` return first/last element of a tuple, or `x` itself."""
    assert front((1, 2, 3)) == 1
    assert back((1, 2, 3)) == 3
    assert front(7) == 7
    assert back(7) == 7

  def test_replace_front_back(self):
    assert replace_front((1, 2, 3), 9) == (9, 2, 3)
    assert replace_back((1, 2, 3), 9) == (1, 2, 9)
    # Non-tuple input: replace returns the new value
    assert replace_front(0, 9) == 9
    assert replace_back(0, 9) == 9

  def test_product(self):
    assert product(2) == 2
    assert product((3,2)) == 6
    assert product((((2,3),4))) == 24

  def test_product_each(self):
    """`product_each` collapses each top-level mode but keeps the rank."""
    # Already-flat shapes are returned unchanged
    assert product_each((2, 3)) == (2, 3)
    assert product_each((2, 3, 4)) == (2, 3, 4)
    # Hierarchical modes are collapsed within each top-level mode
    assert product_each(((2, 3), 4)) == (6, 4)
    assert product_each(((2, (3, 4)), (5, 6), 7)) == (24, 30, 7)

  def test_inner_product(self):
    assert inner_product(2, 3) == 6
    assert inner_product((1,2), (3,2)) == 7
    assert inner_product(((2,3),4), ((2,1),2)) == 15

  def test_prefix_product(self):
    assert prefix_product(2) == 1
    assert prefix_product((3,2)) == (1,3)
    assert prefix_product((3,2,4)) == (1,3,6)
    assert prefix_product(((2,3),4)) == ((1,2),6)
    assert prefix_product(((2,3),(2, 1, 2),( 5,  2,  1))) == ((1,2),(6,12,12),(24,120,240))

  def test_idx2crd(self):
    # In bounds
    assert idx2crd(7, 14) == 7
    assert idx2crd(7, (3,2,4)) == (1,0,1)
    assert idx2crd(7, (3,(2,4))) == (1,(0,1))
    assert idx2crd(7, ((3,2),4)) == ((1,0),1)
    assert idx2crd(7, ((3,2),4)) == ((1,0),1)

    # Out of bounds
    assert idx2crd(7, 5) == 7
    assert idx2crd(42, (3,7,2)) == (0,0,2)
    assert idx2crd(42, (3,7,2,1)) == (0,0,0,1)
    assert idx2crd(42, (3,7,(2,1))) == (0,0,(0,1))

    # General
    shape = ((2,3),(2,1,2))
    for idx in range(7*product(shape)):
      assert idx == crd2idx(idx,shape)
      assert idx != idx2crd(idx,shape)
      assert idx == crd2idx(idx2crd(idx,shape),shape)

  def test_slice_dice(self):
    shape = ((2, 3), (5, 7, 9))

    assert slice_(0, shape) == ()
    assert dice_(0, shape) == (shape,)

    assert slice_(None, shape) == (shape,)
    assert dice_(None, shape) == ()

    assert slice_((None, 1), shape) == (shape[0],)
    assert dice_((None, 1), shape) == (shape[1],)

    assert slice_((1, (1, 1, None)), shape) == (shape[1][2],)
    assert dice_((1, (1, 1, None)), shape) == (shape[0], shape[1][0], shape[1][1])

    assert slice_((1, (1, None, None)), shape) == (shape[1][1], shape[1][2])
    assert dice_((1, (1, None, None)), shape) == (shape[0], shape[1][0])

    assert slice_((1, (None, None, None)), shape) == (shape[1][0], shape[1][1], shape[1][2])
    assert dice_((1, (None, None, None)), shape) == (shape[0],)

    assert slice_((None, (1, None, 1)), shape) == (shape[0], shape[1][1])
    assert dice_((None, (1, None, 1)), shape) == (shape[1][0], shape[1][2])

    assert slice_(((None, 1), (1, None, None)), shape) == (shape[0][0], shape[1][1], shape[1][2])
    assert dice_(((None, 1), (1, None, None)), shape) == (shape[0][1], shape[1][0])


class TestGetLift:
  """`get` and `lift` are inverses on the same path."""

  def test_get_with_path(self):
    assert get(((0, 0, (0, 0, 0, 42)),), (0, 2, 3)) == 42
    assert get((1, (2, (3, 4))), (1, 1, 0)) == 3
    assert get(7, ()) == 7             # empty path is identity

  def test_get_subscript_form(self):
    """`get[i, j, k](x)` is equivalent to `get(x, (i, j, k))`."""
    assert get[0, 2, 3](((0, 0, (0, 0, 0, 42)),)) == 42
    assert get[1, 1, 0]((1, (2, (3, 4)))) == 3

  def test_lift_round_trip(self):
    """`get(lift(x, mode), mode) == x` for any `x` and `mode`."""
    for mode in [(), (0,), (1,), (0, 2, 3), (1, 1, 0)]:
      x = 99
      assert get(lift(x, mode), mode) == x

  def test_lift_zero_padded(self):
    """`lift` produces a zero-padded structure with the value at `mode`."""
    assert lift(42, (0, 2, 3)) == ((0, 0, (0, 0, 0, 42)),)


class TestSelectTake:
  """`select` picks specific top-level modes; `take` picks a range
  `[begin, end)`. Both always return a tuple."""

  def test_select_on_tuple(self):
    assert select((1, 2, 3, 4), (0, 2)) == (1, 3)
    assert select[0, 2]((1, 2, 3, 4)) == (1, 3)
    assert select[2]((1, 2, 3, 4)) == (3,)
    assert select[0, 1, 3]((1, 2, 3, 4)) == (1, 2, 4)

  def test_select_on_layout(self):
    """`select` on a `Layout` returns a tuple of sub-layouts."""
    A = Layout((2, 3, 5, 7), (1, 2, 6, 30))
    assert select[1, 3](A) == (Layout(3, 2), Layout(7, 30))
    assert select[0, 1, 3](A) == (Layout(2, 1), Layout(3, 2), Layout(7, 30))
    assert select[2](A) == (Layout(5, 6),)

  def test_select_with_make_layout_recovers_cpp_select(self):
    """`make_layout(select[I...](A))` is the C++-style `cute::select<I...>(A)`."""
    A = Layout((2, 3, 5, 7), (1, 2, 6, 30))
    assert make_layout(select[1, 3](A)) == Layout((3, 7), (2, 30))
    assert make_layout(select[0, 1, 3](A)) == Layout((2, 3, 7), (1, 2, 30))

  def test_take_on_tuple(self):
    assert take((1, 2, 3, 4), (1, 3)) == (2, 3)
    assert take[1, 3]((1, 2, 3, 4)) == (2, 3)
    assert take[0, 4]((1, 2, 3, 4)) == (1, 2, 3, 4)     # full
    assert take[1, 1]((1, 2, 3, 4)) == ()               # empty

  def test_take_on_layout(self):
    """`take` on a `Layout` returns a tuple of consecutive sub-layouts."""
    A = Layout((2, 3, 5, 7), (1, 2, 6, 30))
    assert take[1, 3](A) == (Layout(3, 2), Layout(5, 6))
    assert take[1, 4](A) == (Layout(3, 2), Layout(5, 6), Layout(7, 30))

  def test_take_reverse_range_raises(self):
    """`take[end, begin]` with `begin > end` is a ValueError."""
    with pytest.raises(ValueError):
      take[3, 1]((1, 2, 3, 4))


class TestTransformLeaf:
  """`transform_leaf` and `transform_apply_leaf` traverse hierarchical
  tuples leaf-by-leaf, applying a function and rebuilding the result."""

  def test_transform_leaf_unary(self):
    assert transform_leaf(lambda a: a * 2, ((1, (2, 3)), 4)) == ((2, (4, 6)), 8)

  def test_transform_leaf_binary(self):
    assert (transform_leaf(lambda a, b: a + b,
                                    ((1, (2, 3)), 4),
                                    ((10, (20, 30)), 40))) == ((11, (22, 33)), 44)

  def test_transform_apply_leaf_custom_combiner(self):
    """`g=sum` reduces every level rather than rebuilding a tuple."""
    assert transform_apply_leaf(sum, lambda a: a, ((1, 2), (3, 4))) == 10


class TestFlattenUnflatten:
  """`flatten` extracts leaves; `unflatten` rebuilds them along a profile."""

  def test_flatten(self):
    assert flatten(((1, (2, 3)), 4)) == (1, 2, 3, 4)
    assert flatten(7) == (7,)

  def test_leaves_iterator(self):
    assert list(leaves(((1, (2, 3)), 4))) == [1, 2, 3, 4]

  def test_unflatten(self):
    assert unflatten(iter([10, 20, 30, 40]), ((1, (2, 3)), 4)) == ((10, (20, 30)), 40)

  def test_flatten_unflatten_round_trip(self):
    """`unflatten(iter(flatten(x)), x) == x` for any HTuple."""
    for x in [7, (3, 4), ((1, 2), 3), ((1, (2, 3)), 4)]:
      assert unflatten(iter(flatten(x)), x) == x

  def test_repeat_like(self):
    """`repeat_like(v, profile)` produces a tuple of the same shape as
    `profile`, with every leaf set to `v`."""
    assert repeat_like(0, ((1, (2, 3)), 4)) == ((0, (0, 0)), 0)
    assert repeat_like(7, 1) == 7


class TestModeOpDecorator:
  """The ModeOpDecorator allows `op[i, j, ...](x)` as shorthand for
  `op(get[i, j, ...](x))`. This applies to `shape`, `size`, `rank`,
  `depth`, `stride`, `coshape`, and `coprofile`."""

  def test_mode_indexed_shape(self):
    A = Layout(((2, 3), 4), ((4, 8), 1))
    assert shape(A) == ((2, 3), 4)
    assert shape[0](A) == (2, 3)
    assert shape[0, 1](A) == 3
    assert shape[1](A) == 4

  def test_mode_indexed_size(self):
    A = Layout(((2, 3), 4), ((4, 8), 1))
    assert size(A) == 24
    assert size[0](A) == 6
    assert size[1](A) == 4
    assert size[0, 0](A) == 2
    assert size[0, 1](A) == 3

  def test_mode_indexed_rank_and_depth(self):
    A = Layout(((2, 3), 4), ((4, 8), 1))
    assert rank(A) == 2
    assert rank[0](A) == 2              # the (2, 3) sub-mode
    assert rank[1](A) == 1              # the integer 4 mode
    assert depth(A) == 2
    assert depth[0](A) == 1
    assert depth[1](A) == 0
