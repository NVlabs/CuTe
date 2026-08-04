# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
CuTe Tensor
"""

from __future__ import annotations

import ctypes

from .shape import *
from .layout import *
from .accessor import *

class Tensor:
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
    i = transform_leaf(lambda x: None if x == slice(None) else x, i)
    offset, sliced = self.layout._offset_and_slice(i)
    return self.accessor[offset] if rank(sliced) == 0 else Tensor(self.accessor + offset, sliced)

  def __setitem__(self, i: Coord, value):
    i = transform_leaf(lambda x: None if x == slice(None) else x, i)
    offset, sliced = self.layout._offset_and_slice(i)
    if rank(sliced) != 0: raise ValueError(f"Tensor.__setitem__({i}, {value}): Incomplete coordinate in setitem.")
    self.accessor[offset] = value

  def get(self, mode=()) -> Tensor:
    return Tensor(self.accessor, get(self.layout, mode))

  def __eq__(self, other) -> bool:
    return self.accessor == other.accessor and self.layout == other.layout

  def _coalesce(self, profile=1) -> Tensor:
    """
    Coalesce the tensor's layout according to the profile.
    Returns a new Tensor with the coalesced layout.
    """
    return Tensor(self.accessor, self.layout._coalesce(profile))

  def _coalesce_z(self, profile=1) -> Tensor:
    """
    Coalesce the tensor's layout according to the profile.
    Returns a new Tensor with the coalesced layout.
    """
    return Tensor(self.accessor, self.layout._coalesce_z(profile))

  def _composition(self, B) -> Tensor:
    """
    Group composition of Tensor with B to produce a Tensor.
    Returns a new Tensor with the composed layout.
    """
    return Tensor(self.accessor, self.layout._composition(B))

  def _logical_divide(self, B) -> Tensor:
    """
    Logical divide of Tensor with B to produce a Tensor.
    Returns a new Tensor split according to B.
    """
    return Tensor(self.accessor, self.layout._logical_divide(B))

  # print and str
  def __str__(self) -> str:
    return f"{self.accessor} o {self.layout}"

  # error msgs and representation
  def __repr__(self) -> str:
    return f"Tensor({self.accessor}, {self.layout})"


def is_tensor(x) -> bool:
  return isinstance(x, Tensor)


def identity_tensor(shape: Shape) -> Tensor:
  return Tensor(ImplicitAccessor(0), Layout(shape, make_basis_like(shape)))


def make_tensor(layout: Layout | Shape, dtype=ctypes.c_double) -> Tensor:
  """
  Allocate an `Array` of size `coshape(layout)` and bind it to `layout`.

  To bind a layout to data you already have, wrap it in a `Ptr` instead.
  """
  if not is_layout(layout):
    if not is_tuple(layout) and not is_int(layout):
      raise ValueError(f"make_tensor({dtype}, {layout}): Invalid layout {layout}")
    layout = Layout(layout)
  N = coshape(layout)
  if not is_int(N):
    raise ValueError(f"make_tensor({dtype}, {layout}): Non-integer codomain {N}")
  return Tensor(Array(N, dtype), layout)
