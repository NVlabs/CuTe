# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CuTe Tensor

A `Tensor` is an `Accessor` composed with a `Layout`: the layout turns a
coordinate into an offset, the accessor turns that offset into a value.
"""

from __future__ import annotations

import ctypes

from .shape import *
from .layout import *
from .accessor import *
from .algebra import *

class Tensor:
  """
  A CuTe Tensor: an `Accessor` paired with a `Layout`.

  Indexing accepts a coordinate in any of the forms a `Layout` does. A
  coordinate that leaves modes unnamed, marking them with `None` or `:`, returns
  a sub-tensor view over those modes instead of an element.

  The layout algebra applies through the tensor -- `coalesce`, `composition`,
  `logical_divide` and `zipped_divide` rewrap the same accessor -- so reshaping
  or tiling a tensor never touches its data.

  Examples:
    T = make_tensor(Layout((4, 4), (4, 1)))
    T[1, 2] = 42.0
    T[1, 2] == 42.0
    shape(T) == (4, 4)
    T[1, None] == T[1, :]                          # a row, as a sub-tensor
    shape(T[1, None]) == (4,)
    T[1, None][2] == 42.0                          # sharing T's storage
    coalesce(make_tensor(Layout((4, 4), (1, 4)))).layout == Layout(16, 1)
  """

  __slots__ = ("accessor", "layout")

  def __init__(self, accessor: Accessor | MutableAccessor, layout: Layout):
    self.accessor = accessor
    if not (isinstance(self.accessor, Accessor) or isinstance(self.accessor, MutableAccessor)):
      raise ValueError(f"Expected an accessor as the first argument to Tensor({accessor}, {layout})")
    self.layout = layout
    if not is_layout(self.layout):
      raise ValueError(f"Expected a layout as the second argument to Tensor({accessor}, {layout})")

  @property
  def shape(self) -> Shape:
    """Shape of the layout domain"""
    return shape(self.layout)

  def __getitem__(self, i: Coord):
    """
    Read the element at `i`, or return the sub-tensor `i` leaves unnamed.

    Examples:
      T = make_tensor(Layout((4, 4), (4, 1)))
      T[2, 3]            == 0.0
      shape(T[None, 2])  == (4,)
    """
    offset, sliced = self.layout._offset_and_slice(i)
    return self.accessor[offset] if rank(sliced) == 0 else Tensor(self.accessor + offset, sliced)

  def __setitem__(self, i: Coord, value):
    """
    Write `value` at `i`.

    Pre-conditions:
      `i` names every mode; a slicing coordinate raises a ValueError
    """
    offset, sliced = self.layout._offset_and_slice(i)
    if rank(sliced) != 0: raise ValueError(f"Tensor.__setitem__({i}, {value}): Incomplete coordinate in setitem.")
    self.accessor[offset] = value

  def get(self, mode=()) -> Tensor:
    """Get the sub-tensor at the given (possibly nested) `mode` path."""
    return Tensor(self.accessor, get(self.layout, mode=mode))

  def __eq__(self, other) -> bool:
    """Two Tensors are equal iff their accessors and layouts are equal."""
    return self.accessor == other.accessor and self.layout == other.layout

  def _coalesce(self, profile=1) -> Tensor:
    """Coalesce the tensor's layout according to the profile."""
    return Tensor(self.accessor, coalesce(self.layout, profile))

  def _coalesce_z(self, profile=1) -> Tensor:
    """Coalesce the tensor's layout according to the profile."""
    return Tensor(self.accessor, coalesce_z(self.layout, profile))

  def _composition(self, B) -> Tensor:
    """Group composition of Tensor with B to produce a Tensor."""
    return Tensor(self.accessor, composition(self.layout, B))

  def _logical_divide(self, B) -> Tensor:
    """Logical divide of Tensor with B to produce a Tensor."""
    return Tensor(self.accessor, logical_divide(self.layout, B))

  # print and str
  def __str__(self) -> str:
    return f"{self.accessor} o {self.layout}"

  # error msgs and representation
  def __repr__(self) -> str:
    return f"Tensor({self.accessor}, {self.layout})"


def is_tensor(x) -> bool:
  """
  True iff `x` is a `Tensor`.

  Examples:
    is_tensor(make_tensor(Layout((2, 2))))  == True
    is_tensor(Layout((2, 2)))               == False
  """
  return isinstance(x, Tensor)


def identity_tensor(shape: Shape) -> Tensor:
  """
  The tensor mapping every coordinate to itself.

  Coordinate strides over an `ImplicitAccessor`, so nothing is allocated and
  reading a position yields the coordinate that reaches it. Tiling it the same
  way as a data tensor is how a tile recovers its global coordinates, which is
  what predication needs.

  Examples:
    identity_tensor((3, 4))[1, 2]   == (1, 2)
    shape(identity_tensor((3, 4)))  == (3, 4)
  """
  return Tensor(ImplicitAccessor(0), Layout(shape, make_basis_like(shape)))


def make_tensor(layout: Layout | Shape, dtype=ctypes.c_double) -> Tensor:
  """
  Allocate an `Array` of size `coshape(layout)` and bind it to `layout`.

  To bind a layout to data you already have, wrap it in a `Ptr` instead.

  Pre-conditions:
    coshape(layout) is an Integer; a coordinate codomain raises a ValueError

  Examples:
    shape(make_tensor(Layout((4, 4), (4, 1))))  == (4, 4)
    make_tensor((2, 3)).layout                  == Layout((2, 3), (1, 2))
    make_tensor(Layout((2, 2), (E(0), E(1))))   -> ValueError
  """
  if not is_layout(layout):
    if not is_tuple(layout) and not is_int(layout):
      raise ValueError(f"make_tensor({dtype}, {layout}): Invalid layout {layout}")
    layout = Layout(layout)
  N = coshape(layout)
  if not is_int(N):
    raise ValueError(f"make_tensor({dtype}, {layout}): Non-integer codomain {N}")
  return Tensor(Array(N, dtype), layout)
