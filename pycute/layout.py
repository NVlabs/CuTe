# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Definition of CuTe Layouts and functions to manipulate them
"""

from __future__ import annotations

from itertools import zip_longest

from collections.abc import Iterable
from typing import Union, TypeAlias

from .typedefs import *
from .htuple import *
from .atuple import *
from .stride import *
from .stride import _coalesce_z
from .shape import *


class LayoutBase:
  pass


def is_layout(x):
  return isinstance(x, LayoutBase)


class Layout(LayoutBase):
  """
  A CuTe Layout: a map from a coordinate domain to a codomain, 
  defined by a `shape` and a `stride`.

  The  `shape` is an HTuple of Integers.
  The `stride` is an HTuple of stride scalars and is congruent with `shape`.

  A Layout evaluates as `L(c) == inner_product(idx2crd(c, shape), stride)`,
  mapping any coordinate of `shape` -- integral, flat, natural -- to a 
  codomain value.

  The default constructor fills in a compact, column-major stride via
  `prefix_product(shape)`; pass an explicit `stride` (or a single integer base)
  to override.

  The layout algebra (composition, complement, inverses, divide, product, ...) is
  exposed as free functions in `algebra.py`; the `_`-prefixed methods here are
  their Layout implementations.

  Examples:
    Layout((4, 8))               == Layout((4, 8), (1, 4))   # default compact column-major
    Layout((4, 8), (8, 1))(2, 3) == 19                       # evaluate a coordinate
  """
  __slots__ = ("shape", "stride")

  def __init__(self, shape: Shape, stride: Stride = 1):
    """
    Construct a Layout from a `shape` and `stride`.
    """
    self.shape  = shape
    self.stride = prefix_product(shape, stride)

  @classmethod
  def _set(cls, _shape, _stride):
    """
    Construct a Layout directly from an already-computed `_shape` and `_stride`.
    """
    obj = cls.__new__(cls)  # Does not call __init__
    obj.shape  = _shape
    obj.stride = _stride    # Skip the prefix_product and congruence check
    return obj

  def _coshape(self) -> Shape:
    """
    Shape of the layout's codomain
    """
    result = inner_product(transform_leaf(lambda s: s-1, self.shape), self.stride)
    return as_tuple(result + repeat_like(1, shape(result)))

  def _coprofile(self) -> Profile:
    """
    Profile of the layout's codomain.
    """
    return as_tuple(sum(leaves(self.stride)))

  def __call__(self, *crd: Coord) -> StrideScalar:
    """
    Map a coordinate to the layout's codomain:
    `L(c) == inner_product(idx2crd(c, shape), stride)`.

    Accepts a coordinate in any form (integral, flat, or natural), passed either
    as a single argument or as separate per-mode arguments.

    Examples:
      L = Layout((4, 8), (8, 1))
      L(14) == 19
      L(2, 3) == 19
      L((2, 3)) == 19
    """
    crd = crd[0] if len(crd) == 1 else crd
    return inner_product(idx2crd(crd, self.shape), self.stride)

  def _offset_and_slice(self, crd: Coord):
    """
    Evaluate `crd` to a codomain offset AND slice the layout, returning
    `(offset, sublayout)`. `None` entries in `crd` mark the modes retained in the
    sublayout.
    """
    return (self(crd), Layout._set(slice_(crd, self.shape), slice_(crd, self.stride)))

  def __getitem__(self, i: Integer) -> Layout:
    """
    Get mode `i` of the layout as a sublayout (tuple-like indexing over modes).
    """
    if i >= rank(self):
      raise IndexError(f"Index {i} out of range for Layout {self}")
    if is_tuple(self.shape):
      return Layout._set(self.shape[i], self.stride[i])
    return Layout._set(self.shape, self.stride)

  def get(self, mode=()) -> Layout:
    """
    Get the sublayout at the given (possibly nested) `mode` path.
    """
    return Layout._set(reduce(lambda a,i: a[i], mode, self.shape),
                       reduce(lambda a,i: a[i], mode, self.stride))

  def __eq__(self, other) -> bool:
    """Two Layouts are equal iff their shapes and strides are equal."""
    if not is_layout(other):
      return NotImplemented
    return self.shape == other.shape and self.stride == other.stride

  def _coalesce_z(self, profile=1) -> Layout:
    """
    Coalesce this Layout per `profile` (size-1-preserving variant).
    """
    if profile is None:
      return self
    if is_tuple(profile):
      if rank(self) < len(profile): raise ValueError(f"Rank mismatch: coalesce_z({self}, {profile})")
      return make_layout(a._coalesce_z(p) for a,p in zip_longest(self,profile))

    new_s, new_d = _coalesce_z(self.shape, self.stride)
    if new_s == ():
      return Layout._set(1, 0)
    return Layout._set(unwrap(new_s), unwrap(new_d))

  def _coalesce(self, profile=1) -> Layout:
    """
    Coalesce this Layout per `profile`.
    """
    if profile is None:
      return self
    if is_tuple(profile):
      if rank(self) < len(profile): raise ValueError(f"Rank mismatch: coalesce({self}, {profile})")
      return make_layout(a._coalesce(p) for a,p in zip_longest(self,profile))

    new_s, new_d = _coalesce_z(self.shape, self.stride)
    if new_s == ():
      return Layout._set(1, 0)
    if len(new_s) > 1 and new_s[-1] == 1:
      return Layout._set(unwrap(new_s[:-1]), unwrap(new_d[:-1]))
    return Layout._set(unwrap(new_s), unwrap(new_d))

  def _composition(self, B: Tiler) -> Layout:
    """
    Compose this Layout with `B`.
    """
    if B is None:           # RHS None, noop
      return self
    if is_int(B):           # RHS int, A o N -> A o N:1
      B = Layout._set(B, 1)
    if is_tuple(B):         # RHS tuple, (A0,A1,...) o <X,Y,...> => (A0 o X, A1 o Y, ...)
      if rank(self) < len(B): raise ValueError(f"Rank mismatch: composition({self}, {B})")
      return make_layout(a._composition(b) for a,b in zip_longest(self,B))

    #
    # Special cases with A: Layout and B: Layout
    #

    A = self._coalesce_z(coprofile(B))

    if is_tuple(B.shape):   # RHS distributive, A o (X,Y,...) => (A o X, A o Y, ...)
      return make_layout(A._composition(b) for b in B)
    if B.stride == 0:       # Special case stride-0, A o N:0 => N:0
      return Layout._set(B.shape, 0)
    if B.shape == 1:        # Special case shape-1, A o 1:M => 1:A(M)
      return Layout._set(B.shape, A(B.stride))

    #
    # General case   (A0,A1,...) o N:M
    #

    from .algebra import layout_add
    resultL = None

    for strideB, basisB in basis_repr(B.stride):
      Ab = get(A, basisB)
      result_s, result_d = list(wrap(Ab.shape)), list(wrap(Ab.stride))

      # Truncate/extend result_s based on strideB * B.shape
      result_s[-1] = strideB * B.shape
      for i in range(len(result_s)-1):
        result_s[-1], rES = divmod(result_s[-1], result_s[i])
        if result_s[-1] == 0:
          result_s[i] = rES
          result_s = result_s[:i+1]
          result_d = result_d[:i+1]
          break
        if rES != 0:
          raise ValueError(f"Shape divisibility condition violated: composition({self}, {B})")

      # Remove result_s prefix strideB
      for i in range(len(result_s)-1):
        qSD, rSD = divmod(result_s[i], strideB)
        if rSD == 0:
          result_s[i]  = qSD
          result_d[i] *= strideB
          break
        strideB, rDS = divmod(strideB, result_s[i])
        result_s[i] = 1
        if rDS != 0:
          raise ValueError(f"Stride divisibility condition violated: composition({self}, {B})")
      else:
        result_s[-1] //= strideB
        result_d[-1]  *= strideB

      # Accumulate into resultL
      resultL = layout_add(resultL, Layout._set(result_s, result_d))

    return resultL._coalesce()

  def _right_inverse(self) -> Layout:
    """
    Largest right inverse of this Layout.
    """
    coprof   = coprofile(self)
    result_S = unflatten(iter(lambda: [ ], -1), coprof)  # Avoid aliasing [] from repeat_like
    result_D = unflatten(iter(lambda: [ ], -1), coprof)
    curr_D   = unflatten(iter(lambda: [1], -1), coprof)

    flat_s, flat_d = _coalesce_z(self.shape, self.stride)
    # The chain is followed in stride order. Only static (concrete) strides
    # can be ordered, so a symbolic stride is filtered past the sort.
    def _stride_key(dsp):
      return (0, dsp[0]) if is_static(dsp[0]) else (1, 0)

    for de, s, pps in sorted(zip(flat_d, flat_s, prefix_product(flat_s)), key=_stride_key):
      d = proj(de, de)
      result_s = proj(result_S, de)
      result_d = proj(result_D, de)
      curr_d   = proj(curr_D,   de)

      if d == 0 or s == 1:
        continue
      if d != curr_d[0]:
        continue

      result_s.append(s)
      result_d.append(pps)
      curr_d[0] = s * d

    return Layout._set(tuple(result_S), tuple(result_D))._coalesce(coprof)

  def _left_inverse(self) -> Layout:
    """
    Left inverse of this Layout.
    """
    coprof   = coprofile(self)
    result_S = unflatten(iter(lambda: [1], -1), coprof)  # Avoid aliasing [] from repeat_like
    result_D = unflatten(iter(lambda: [0], -1), coprof)
    curr_S   = unflatten(iter(lambda: [1], -1), coprof)

    flat_s, flat_d = _coalesce_z(self.shape, self.stride)
    for de, s, pps in sorted(zip(flat_d, flat_s, prefix_product(flat_s))):
      d = proj(de, de)
      result_s = proj(result_S, de)
      result_d = proj(result_D, de)
      curr_s   = proj(curr_S,   de)

      if d == 0 or s == 1:                  # Stride-0 / size-1 modes carry no information
        continue
      gap, rem = divmod(d, curr_s[0])       # gap = d_k / d_{k-1}, the span to the next stride
      if rem != 0:
        raise ValueError(f"left_inverse({self}): Strides do not form an ordered chain")
      if gap < result_s[-1]:                # d_k must clear the previous mode: d_k >= d_{k-1} * s_{k-1}
        raise ValueError(f"left_inverse({self}): Non-injective layout")

      result_s[-1] = gap                    # Pad the previous mode out to d_k (the extra entries are holes)
      curr_s[0]   *= gap                    # Advance the consumed stride to d_k
      result_s.append(s)                    # Record this mode (a later mode overwrites s with its own gap)
      result_d.append(pps)

    return Layout._set(tuple(result_S), tuple(result_D))._coalesce_z(coprof)

  def _complement(self, extend: Shape | None = None) -> Layout:
    """
    Complement of this Layout, optionally extended to cover `extend`.
    """
    coprof   = coprofile(self)
    result_S = unflatten(iter(lambda: [ ], -1), coprof)  # Avoid aliasing [] from repeat_like
    result_D = unflatten(iter(lambda: [1], -1), coprof)

    # Modes are ordered by stride and only *static* (concrete) strides can be
    # ordered, so filter symbolic-strides since their relative ordering is unknown.
    def _stride_key(ds):
      return (0, ds[0]) if is_static(ds[0]) else (1, 0)

    for de, s in sorted(zip(leaves(self.stride), leaves(self.shape)), key=_stride_key):
      d = proj(de, de)
      result_s = proj(result_S, de)
      result_d = proj(result_D, de)

      if d == 0 or s == 1:
        continue
      # The injectivity precondition is enforced only where it is statically
      # decidable; a symbolic stride or running position is taken on faith.
      if is_static(d) and is_static(result_d[-1]) and d < result_d[-1]:
        raise ValueError(f"complement({self}): Non-injective layout in complement")

      result_s.append(d // result_d[-1])
      result_d.append(d * s)

    result = transform_leaf(lambda c,rs,rd: Layout._set(tuple(rs+[1]), tuple(rd))._coalesce_z(), coprof, result_S, result_D)
    result = tiler_to_layout(result)

    # If extend is provided, extend the result
    if extend:
      def extend_complement(_, shapeC, strideC, shapeA, strideA):
        if shapeC is None:
          return Layout._set(shapeA, strideA)
        # The last extent of complement is always 1, so update it to extend
        last_strideC = back(strideC)
        sizeC = proj(last_strideC, last_strideC)
        #sizeR = (size(shapeA) + sizeC - 1) // sizeC
        shapeR = list(leaves(shapeA))
        for i, s in enumerate(shapeR):
          shapeR[i] = (s + sizeC - 1) // sizeC
          sizeC     = (s + sizeC - 1) // s
        return Layout(replace_back(shapeC, shapeR), strideC)._coalesce()
      # Extend the result
      result = transform_apply_leaf(make_layout, extend_complement,
                                    coprof, result.shape, result.stride,
                                    extend, make_basis_like(extend))

    return result

  def _logical_divide(self, B: Tiler) -> Layout:
    """
    Split this Layout into the elements of `B` (the Tile) and a Grid over those
    Tiles; tuples-of-Layouts apply by-mode and `None` is a no-op.
    """
    if B is None:
      return self
    if is_int(B):
      B = Layout._set(B, 1)
    if is_tuple(B):
      if rank(self) < len(B): raise ValueError(f"Rank mismatch: logical_divide({self}, {B})")
      return make_layout(a._logical_divide(b) for a,b in zip_longest(self,B))
    from .algebra import complement
    return self._composition(make_layout([B, complement(B, extend=self.shape)]))

  def _nullspace(self) -> Layout:
    """
    Nullspace of this Layout.
    """
    fstride = flatten(self.stride)
    iseq = [i for i,d in enumerate(fstride) if d == 0]
    if len(iseq) == 0:
      return Layout._set(1, 0)
    fshape = flatten(self.shape)
    pshape = prefix_product(fshape)
    return Layout._set(unwrap(tuple(fshape[i] for i in iseq)),
                       unwrap(tuple(pshape[i] for i in iseq)))


  def _logical_product(self, B: Tiler) -> Layout:
    """
    Reproduce this Layout over `B`.
    """
    if B is None:
      return self
    if is_int(B):
      B = Layout._set(B, 1)
    if is_tuple(B):
      if rank(self) < len(B): raise ValueError(f"Rank mismatch: logical_product({self}, {B})")
      return make_layout(a._logical_product(b) for a,b in zip_longest(self,B))

    return make_layout([self, self._complement()._composition(B)]);

  def __str__(self) -> str:
    """Compact `shape:stride` form, e.g. `(4, 8):(1, 4)`."""
    return f"{self.shape}:{self.stride}"

  def __repr__(self) -> str:
    """Constructor form, e.g. `Layout((4, 8), (1, 4))`."""
    return f"Layout({self.shape}, {self.stride})"


def make_layout(layouts: Iterable[Layout]) -> Layout:
  """
  Concatenate multiple Layouts; each input becomes one mode of the result.

  Post-conditions:
    rank(result) == len(layouts)
    result[i] == layouts[i]   for i in rank(result)

  Examples:
    make_layout([Layout(3, 1), Layout((5, 1), (7, 2)), Layout(2, 42)])
        == Layout((3, (5, 1), 2), (1, (7, 2), 42))
  """
  return Layout._set(*zip(*((a.shape,a.stride) for a in layouts)))


def make_layout_like(layout: Layout) -> Layout:
  """
  Construct a compact Layout with the same shape as `layout` whose strides
  follow the ordering induced by `layout`'s strides.

  The mode with the smallest non-zero source stride receives stride 1, and the
  remaining non-zero modes receive compact (prefix-product) strides in stable
  ascending order of the source stride magnitudes. Modes that carry no positional
  information -- a size-1 shape or a static stride of 0 -- are pinned to stride 0.

  Only static strides can be ordered by magnitude; symbolic (non-static) strides
  are considered larger than every static stride.

  Post-conditions:
    shape(result) == shape(layout)
    the non-zero modes of result form a compact (densely-packed) layout
    idempotent: make_layout_like(make_layout_like(A)) == make_layout_like(A)

  Examples:
    make_layout_like(Layout((4, 8), (1, 4)))              == Layout((4, 8), (1, 4))
    make_layout_like(Layout((4, 8), (100, 1)))            == Layout((4, 8), (8, 1))
    make_layout_like(Layout((2, 3, 4, 5), (0, 42, 4, 0))) == Layout((2, 3, 4, 5), (0, 4, 1, 0))
  """
  flat_s = list(leaves(layout.shape))
  flat_d = list(leaves(layout.stride))

  # Order the modes by source stride. Only static strides are orderable.
  def _stride_key(dsi):
    return (0, dsi[0]) if is_static(dsi[0]) else (1, 0)

  result_d = [0] * len(flat_s)
  current  = 1
  for d, s, i in sorted(zip(flat_d, flat_s, range(len(flat_s))), key=_stride_key):
    if (is_static(d) and d == 0):
      continue                # leave result stride at 0
    result_d[i] = current
    current    *= s

  return Layout._set(layout.shape, unflatten(iter(result_d), layout.shape))


def make_ordered_layout(_shape: Shape, _order: IntTuple) -> Layout:
  """
  Construct a compact Layout with the same shape as `_shape` whose strides
  follow the ordering induced by `_order`.

  The mode with the smallest `_order` receives stride 1, and the remaining
  modes receive compact (prefix-product) strides in ascending order of
  `_order`. Only the relative ordering of the `_order` values matters, not
  their magnitudes, so they need not be a contiguous `0..rank-1` permutation.

  Only static orders can be ordered by magnitude; symbolic (non-static) orders
  are considered larger than every static order and, being mutually
  incomparable, retain their left-to-right order.

  Pre-conditions:
    congruent(_shape, _order)

  Post-conditions:
    shape(result) == _shape
    the modes of result form a compact (densely-packed) layout

  Examples:
    make_ordered_layout((4, 8), (0, 1))              == Layout((4, 8), (1, 4))
    make_ordered_layout((4, 8), (1, 0))              == Layout((4, 8), (8, 1))
    make_ordered_layout((2, 3, 4, 2), (0, 2, 3, 0))  == Layout((2, 3, 4, 2), (1, 4, 12, 2))
  """
  if not congruent(_shape, _order):
    raise ValueError(f"make_ordered_layout: shape and order are not congruent")
  flat_s = list(leaves(_shape))
  flat_o = list(leaves(_order))

  # Order the modes by `_order`. Only static orders are orderable.
  def _order_key(osi):
    return (0, osi[0]) if is_static(osi[0]) else (1, 0)

  result_d = [0] * len(flat_s)
  current  = 1
  for o, s, i in sorted(zip(flat_o, flat_s, range(len(flat_s))), key=_order_key):
    result_d[i] = current
    current    *= s

  return Layout._set(_shape, unflatten(iter(result_d), _shape))


# ---------------------------------------------------------------------------
# Tiler type (Whitepaper, §3.3.5 By-mode Composition and Tilers).
#
# A ``Tiler`` is an HTuple whose leaves are either an :class:`Integer` (a mode
# extent) or a :class:`Layout`. It is the general right-hand side accepted by
# composition, ``logical_divide`` and ``logical_product``; ``tiler_to_layout``
# turns it into the equivalent single Layout.
# ---------------------------------------------------------------------------

#: An HTuple(Integer | Layout): the by-mode tiler argument to the algebra.
Tiler: TypeAlias = Union[Integer, "Layout", tuple["Tiler", ...], list["Tiler"]]


def tiler_to_layout(tiler: Tiler, e: StrideScalar = 1) -> Layout:
  """
  Transform a "Tiler" (an HTuple of Layout|Integer) into a Layout that acts
  identically under composition.

  Post-conditions:
    shape(result) == shape(tiler)
    composition(A, result) == composition(A, tiler)        for all admissible Layouts A
    logical_divide(A, result) == zipped_divide(A, tiler)   for all admissible Layouts A

  Examples:
    tiler_to_layout(3)                          == Layout(3, 1)
    tiler_to_layout(Layout((7, 2), (3, 1)))     == Layout((7, 2), (3, 1))
    tiler_to_layout((4, 5))                     == Layout((4, 5), (1@0, 1@1))
    tiler_to_layout((Layout(4, 2), Layout(5, 3))) == Layout((4, 5), (2@0, 3@1))
  """
  if is_int(tiler):
    return Layout._set(tiler, e)
  if is_tuple(tiler):
    return transform_apply_leaf(make_layout, tiler_to_layout, tiler, make_basis_like(tiler))
  if is_layout(tiler):
    return Layout._set(tiler.shape, transform_leaf(lambda d: e * d, tiler.stride))
  raise TypeError(f"tiler_to_layout({tiler}, {e})")


def recast(layout: Layout, scale) -> Layout:
  """
  Recast a Layout to a new element scale.

  Rewrites both shape and stride so the layout addresses a differently-sized
  element: `scale = 8` packs 8 source elements per new element, while
  `scale = Fraction(1, 2)` unpacks 2 new elements per source element. Each leaf
  `s:d` is rescaled by the ratio between `d` and `scale` -- shrinking the shape
  when packing, growing it when unpacking.

  Pre-conditions:
    at each leaf the stride and scale divide cleanly (one is a multiple of the
    other); otherwise a ValueError is raised.

  Examples:
    recast(Layout(24, 1), 8)          == Layout(3, 1)
    recast(Layout(24, 2), 4)          == Layout(12, 1)
    recast(Layout((4, 4), (4, 1)), 4) == Layout((4, 1), (1, 1))
  """
  def recast_elem(shape, stride):
    dd = proj(stride, stride)
    n  = proj( scale, stride)
    if dd == 0:
      return Layout._set(shape, stride)
    if dd == 1:
      return Layout._set(-(-shape // n), stride)
    qdn, rdn = divmod(dd, n)
    qnd, rnd = divmod(n, dd)
    if not (rdn == 0 or rnd == 0):
      raise ValueError(f"recast divisibility condition {shape}, {stride}, {n}")
    qs = -(-shape // (qnd if rnd == 0 else 1))
    return Layout._set(qs, unit(stride) * (qdn if rdn == 0 else 1))

  return transform_apply_leaf(make_layout, recast_elem, layout.shape, layout.stride)
