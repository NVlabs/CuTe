# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Functions for CuTe Strides

A `Stride` is an HTuple of stride scalars, congruent with a layout's `Shape`,
giving the step each mode takes through the codomain (Whitepaper, §2.3). A leaf
need only be an integer-semimodule element -- an `int`, an `ArithTuple`, an `F2`
-- so the same machinery addresses linear memory, multidimensional coordinates
and swizzled offsets alike.

This module holds the stride-side operations: reading a stride (`stride`),
evaluating one against a coordinate (`inner_product`), building a compact one
from a shape (`prefix_product`), and describing the codomain a stride reaches
(`coshape`, `coprofile`).
"""

from .typedefs import *
from .htuple import *


@ModeOpDecorator
def stride(obj, *, mode=()) -> Stride:
  """
  Get an object's stride.

  Examples:
    stride(Layout((4, 8), (1, 4)))          == (1, 4)
    stride((1, (4, 8)))                     == (1, (4, 8))
    stride[1](Layout((3, (2, 4))))          == (3, 6)
    stride(Layout((4, 8), (F2(1), F2(8))))  == (F2(1), F2(8))
  """
  if hasattr(obj, 'stride'):       # Use .stride() or .stride if available (Layouts/Tensors/Other)
    return get(obj.stride() if callable(getattr(obj, 'stride')) else obj.stride, mode=mode)
  if mode != ():                  # Not a Layout or Tensor, so slice once and recurse
    return stride(obj[mode[0]], mode=mode[1:])
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
def coshape(obj, *, mode=()) -> Shape:
  """
  Shape of the codomain: an extent large enough to hold every value `obj` produces.

  Each mode contributes its maximal offset `(s-1) * d`, and where the codomain's
  addition is monotone -- `Z` and `Z^S` -- those contributions add. A codomain
  whose addition is not monotone supplies its own bound instead; `F2`, whose `+`
  is XOR, is bounded by bit-span rather than by sum.

  Examples:
    coshape(Layout((4, 8), (1, 4)))        == 32
    coshape(Layout((4, 8), (E(0), E(1))))  == (4, 8)
    coshape(Layout(4, F2(3)))              == 8
  """
  if mode != ():
    return coshape(get(obj, mode=mode))
  if hasattr(obj, '_coshape'):       # Use ._coshape() or ._coshape if available (Layouts/Other)
    return obj._coshape()
  raise TypeError(f"coshape not supported for type {type(obj)}")


@ModeOpDecorator
def coprofile(obj, *, mode=()) -> Profile:
  """
  Profile of the codomain: an HTuple congruent to `coshape(obj)` whose leaf values
  carry no meaning.

  Read straight off the strides, so unlike `coshape` it stays defined for
  codomains whose extents cannot be bounded.

  Post-conditions:
    congruent(coprofile(obj), coshape(obj))   wherever coshape is defined

  Examples:
    congruent(coprofile(Layout((4, 8), (1, 4))), 0)            == True
    congruent(coprofile(Layout((4, 8), (E(0), E(1)))), (0, 0)) == True
  """
  if mode != ():
    return coprofile(get(obj, mode=mode))
  if hasattr(obj, '_coprofile'):     # Use ._coprofile() if available (Layouts/Other)
    return obj._coprofile()
  raise TypeError(f"coprofile not supported for type {type(obj)}")


def _coalesce_z(shape: Shape, stride: Stride) -> tuple[Shape, Stride]:
  """
  Return a new shape and stride that are coalesced equivalents of the input.
  This is the size-1-preserving ("_z") core fold.

  Merging adjacent modes `(s_a, s_b):(d_a, d_b)` into `s_a*s_b : d_a` must
  preserve both what the layout evaluates to and what its shape records.
  Evaluation is preserved exactly when the map stays linear across the pair,
  which two O(1) checks decide -- jointly necessary and sufficient:

    1. `s_a*d_a == d_b`                       (linearity at `(0, 1)`)
    2. `(s_a-1)*d_a + d_b == (2*s_a-1)*d_a`   (linearity at `(s_a-1, 1)`)

  The shape records the pair only if `s_a*s_b` is a value determined by its
  factors, which a third check settles by forming the product twice:

    3. `s_a*s_b == s_a*s_b`

  For an `int` or a symbolic expression this holds by construction. It fails for
  a leaf whose `*` mints a fresh opaque handle and compares by identity -- a
  traced or JIT integer -- whose merged extent would record nothing the caller
  can read back. (1) and (2) are cheaper still and reject nearly every pair, so
  they run first and gate the two multiplications (3) costs.

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
      if (s_a * d_a == d_b                                # Linearity at (0, 1)
          and (s_a - 1) * d_a + d_b == (2 * s_a - 1) * d_a
          and (s_ab := s_a * s_b) == s_a * s_b):          # Reject an opaque product
        result_s[-1] = s_ab                               # Merge mergeable modes
        continue
    result_s.append(s_b)                                  # Else, Append
    result_d.append(d_b)
  return tuple(result_s), tuple(result_d)