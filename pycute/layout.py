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
  """
  Marker base class for every layout.

  Subclassing it is what makes `is_layout` true, which lets the algebra and the
  visualizers recognize a layout without importing `Layout` itself.
  """
  pass


def is_layout(x):
  """
  True iff `x` is a layout, i.e. any `LayoutBase`.

  Examples:
    is_layout(Layout((4, 8)))  == True
    is_layout((4, 8))          == False
    is_layout(42)              == False
  """
  return isinstance(x, LayoutBase)


class Layout(LayoutBase):
  """
  A CuTe Layout: a map from a coordinate domain to a codomain, defined by a
  `shape` (an HTuple of Integers) and a congruent `stride` (an HTuple of stride
  scalars).

  Evaluates as `L(c) == inner_product(idx2crd(c, shape), stride)`, so every
  coordinate form -- integral, flat, natural -- reaches the same value. The
  default `stride` is the compact column-major `prefix_product(shape)`; pass one
  explicitly, or as a single integer base, to override it.

  The algebra is exposed as free functions in `algebra.py`; the `_`-prefixed
  methods here implement its core operations.

  Examples:
    Layout((4, 8))               == Layout((4, 8), (1, 4))   # default compact column-major
    Layout((4, 8), (8, 1))(2, 3) == 19                       # evaluate a coordinate
    A = Layout((3, (2, 4)), (2, (1, 6)))
    A(17) == A(2, 5) == A(2, (1, 2)) == 17                    # the three coordinate forms
    A[1][0] == Layout(2, 1)                                  # index into the modes
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
    Shape of the layout's codomain: an extent large enough to hold every value
    the layout produces.

    Each mode contributes its maximal offset `(s-1) * d`. Where the codomain's
    addition is monotone -- `Z` and `Z^S` -- those contributions add and the
    extent is one past their total. A codomain whose addition is not monotone
    cannot be bounded that way and supplies its own `_coshape_bound` instead;
    `F2`, whose `+` is XOR, is the case in point.
    """
    max_o = transform_leaf(lambda s,d: (s-1) * d, self.shape, self.stride)
    for o in leaves(max_o):
      if hasattr(o, '_coshape_bound'):     # Codomain with a non-monotone +
        return o._coshape_bound(max_o)
    result = sum(leaves(max_o))
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
      return make_layout(a._composition(b) for a,b in zip(self,B))

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
      Ab = get(A, mode=basisB)
      result_s, result_d = list(wrap(Ab.shape)), list(wrap(Ab.stride))

      # "Divide out" the first strideB elements of A.
      while len(result_s) > 1:
        qDS, rDS = divmod(strideB, result_s[0])
        if rDS != 0:
          break
        strideB = qDS                            # Step past a whole mode
        result_s, result_d = result_s[1:], result_d[1:]

      # Whatever is left of the stride, modify the head mode
      result_d[0] *= strideB
      if len(result_s) > 1:
        qSD, rSD = divmod(result_s[0], strideB)
        if rSD == 0 and qSD > 0:
          result_s[0] = qSD
        elif qSD < B.shape - 1:                  # It reaches past this mode's extent
          raise ValueError(f"Stride divisibility condition violated: composition({self}, {B})")

      # "Keep" the first B.shape elements of what remains.
      result_s[-1] = B.shape
      for i in range(len(result_s)-1):
        result_s[-1], rES = divmod(result_s[-1], result_s[i])
        if result_s[-1] == 0:                    # This mode covers what is left
          result_s[i] = rES
          result_s, result_d = result_s[:i+1], result_d[:i+1]
          break
        if rES != 0:
          raise ValueError(f"Shape divisibility condition violated: composition({self}, {B})")

      # Accumulate into resultL
      resultL = layout_add(resultL, Layout._set(result_s, result_d))

    return resultL._coalesce()

  def _right_inverse(self) -> Layout:
    """
    Largest right inverse of this Layout.

    Each codomain axis is inverted independently by walking its modes in
    increasing stride order.
    """
    flat_s, flat_d = _coalesce_z(self.shape, self.stride)

    # The chain is followed in stride order. Only static (concrete) strides
    # can be ordered, so a symbolic stride is filtered past the sort.
    def _stride_key(dsp):
      return (0, dsp[0]) if is_static(dsp[0]) else (1, 0)

    chain = sorted(zip(flat_d, flat_s, prefix_product(flat_s)), key=_stride_key)

    def invert_axis(e):
      """Invert the single codomain axis `e` by following its chain of strides."""
      curr_d = next((unit(de) for de in flat_d if de != 0 and unit(de) == e), e)
      one    = proj(curr_d, curr_d)         # `1` or `F2(1)`
      result_s, result_d = [], []

      for de, s, pps in chain:
        if de == 0 or s == 1:               # Carries no positional information
          continue

        # Back-substitution can undo a residue that cancels itself, so this tests
        # whether the residue is its own additive inverse -- XOR is,
        # integer addition is not.
        residue = curr_d - de
        if residue + residue != 0 or (s - 1) * residue >= curr_d:
          continue                          # Off-chain: the image stops being contiguous

        stride = pps * one                  # This chain stride's domain index,
        if residue != 0:                    # corrected for the part already covered
          stride += inner_product(idx2crd(residue, result_s), result_d)

        result_s.append(s)
        result_d.append(stride)
        curr_d = s * curr_d

      return Layout._set(tuple(result_s), tuple(result_d))._coalesce()

    return transform_apply_leaf(make_layout, invert_axis, make_basis_like(coprofile(self)))

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

    result = Layout._set(tuple(result_S), tuple(result_D))._coalesce_z(coprof)
    # A gap between strides becomes an extent, so a codomain whose stride
    # quotients are not Integers cannot be chained this way. F2's quotient is a
    # carry-less one, which lands an F2 in the shape.
    if not all(is_int(s) for s in leaves(shape(result))):
      raise ValueError(f"left_inverse({self}): non-integer extent in {shape(result)}")
    return result

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
        last_strideC = wrap(strideC)[-1]
        sizeC = proj(last_strideC, last_strideC)
        #sizeR = (size(shapeA) + sizeC - 1) // sizeC
        shapeR = list(leaves(shapeA))
        for i, s in enumerate(shapeR):
          shapeR[i] = (s + sizeC - 1) // sizeC
          sizeC     = (s + sizeC - 1) // s
        shapeC = wrap(shapeC)[:-1] + (shapeR,)
        return Layout(shapeC, strideC)._coalesce()
      # Extend the result
      result = transform_apply_leaf(make_layout, extend_complement,
                                    coprof, result.shape, result.stride,
                                    extend, make_basis_like(extend))

    return result


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
    result[i] == layouts[i]   for i in range(rank(result))

  Examples:
    make_layout([Layout(3, 1), Layout((5, 1), (7, 2)), Layout(2, 42)])
        == Layout((3, (5, 1), 2), (1, (7, 2), 42))
    make_layout([]) == Layout((), ())
  """
  modes = list(layouts)
  if not modes:
    return Layout._set((), ())
  return Layout._set(*zip(*((a.shape,a.stride) for a in modes)))


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
# A `Tiler` is an HTuple whose leaves are either an `Integer` (a mode
# extent) or a `Layout`. It is the general right-hand side accepted by
# composition, `logical_divide` and `logical_product`; `tiler_to_layout`
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
    tiler_to_layout((4, 5))                     == Layout((4, 5), (E(0), E(1)))       # prints as (1@0, 1@1)
    tiler_to_layout((Layout(4, 2), Layout(5, 3))) == Layout((4, 5), (2*E(0), 3*E(1))) # prints as (2@0, 3@1)
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
