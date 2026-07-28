# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Functions for CuTe Strides
"""

from .typedefs import *
from .htuple import *


@ModeOpDecorator
def stride(obj, mode=()) -> Stride:
  """Get an object's stride"""
  if hasattr(obj, 'stride'):       # Use .stride() or .stride if available (Layouts/Tensors/Other)
    return get(obj.stride() if callable(getattr(obj, 'stride')) else obj.stride, mode)
  if mode != ():                  # Not a Layout or Tensor, so slice once and recurse
    return stride(obj[mode[0]], mode[1:])
  if obj is None or is_stride_scalar(obj):
    return obj
  try:
    return tuple(stride(ai) for ai in obj)
  except TypeError:
    raise TypeError(f"stride({obj}, {mode})")


def inner_product(a: Coord, b: Stride) -> StrideScalar:
  """
  Sum of the leaf-wise products of two congruent HTuples: `sum(x*y)`.

  Pre-conditions:
    congruent(a, b)

  Examples:
    inner_product((1, 0, 1),    (1, 3, 6))       == 7
    inner_product((2, 3),       (1, 4))          == 14
    inner_product((1, (2, 3)),  (1, (10, 100)))  == 321
  """
  return sum(x*y for x,y in zip_leaves(a,b))


def prefix_product(a: Shape, init: Stride = 1) -> Stride:
  """
  Exclusive prefix product of the leaves of `a`, congruent with `a`.

  `init` seeds the running product and may be:
    -- a stride scalar (e.g. `int`; the default `1`), or
    -- a tuple of stride scalars weakly congruent with `a`; 
       each mode is prefix-producted independently.

  Pre-conditions:
    weakly_congruent(init, a)

  Examples:
    prefix_product((3, 2, 4))           == (1, 3, 6)
    prefix_product((3, (2, 4)))         == (1, (3, 6))
    prefix_product((4, 8), 2)           == (2, 8)               # base 2
    prefix_product(((2, 3), (4, 5)), (1, 100)) == ((1, 2), (100, 400))   # per-mode base
  """
  if is_stride_scalar(init):
    return unflatten(iter([init]+[init:=init*v for v in leaves(a)]), a)
  if is_tuple(init):
    if len(a) != len(init): raise ValueError(f"prefix_product({a}, {init})")
    return tuple(prefix_product(x,i) for x,i in zip(a,init))
  raise TypeError(f"prefix_product({a}, {init})")


@ModeOpDecorator
def coshape(obj, mode=()) -> Shape:
  """
  Shape of the codomain
  """
  if hasattr(obj, '_coshape'):       # Use ._coshape() or ._coshape if available (Layouts/Other)
    return get(obj._coshape(), mode)
  raise TypeError(f"coshape not supported for type {type(obj)}")


@ModeOpDecorator
def coprofile(obj, mode=()) -> Shape:
  """
  Profile of the codomain
  """
  return coshape(obj, mode)


def _coalesce_z(shape: Shape, stride: Stride) -> tuple[Shape, Stride]:
  """
  Return a new shape and stride that are coalesced equivalents of the input.
  This is the size-1-preserving ("_z") core fold.

  Two adjacent modes may be merged only when the merge preserves the 
  layout's evaluation. The merge condition below verifies this with two 
  O(1) checks that are jointly necessary and sufficient:

    1. ``s_a*d_a == d_b``                       (linearity at ``(0, 1)``)
    2. ``(s_a-1)*d_a + d_b == (2*s_a-1)*d_a``   (linearity at ``(s_a-1, 1)``)

  Pre-conditions:
    congruent(shape, stride)
  """
  result_s = []                                           # Accumulated shapes
  result_d = []                                           # Accumulated strides
  for s_b, d_b in zip(leaves(shape), leaves(stride)):
    while result_s and result_s[-1] == 1:                 # Drop trailing size-1 modes
      result_s.pop()
      result_d.pop()
    if result_s:
      s_a, d_a = result_s[-1], result_d[-1]
      if (is_static(s_a) == is_static(s_b)                # Don't fold concrete into symbolic
          and s_a * d_a == d_b
          and (s_a - 1) * d_a + d_b == (2 * s_a - 1) * d_a):
        result_s[-1] = s_a * s_b                          # Merge mergeable modes
        continue
    result_s.append(s_b)                                  # Else, Append
    result_d.append(d_b)
  return tuple(result_s), tuple(result_d)