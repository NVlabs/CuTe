# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generic algebraic operations that dispatch to Layout or Tensor methods.
"""

import math

from .layout import *

@ModeOpDecorator
def coalesce_z(A, profile=1, *, mode=()):
  """
  Coalesce a Layout or Tensor into a maximally-merged, equivalent form while
  preserving trailing size-1 modes.

  A non-empty `mode` coalesces only that mode of `A` and leaves every other mode
  unchanged.

  Post-conditions:
    size(result) == size(A)
    depth(result) <= 1   at each leaf of profile, within mode
    result(i) == A(i)   for all integers i

  Examples:
    coalesce_z(Layout((2, 1, 6, 1), (1, 7, 8, 0)))     == Layout((2, 6, 1), (1, 8, 0))
    coalesce_z[1](Layout((3, (2, 6)), (1, (3, 6))))    == Layout((3, 12), (1, 3))
  """
  if mode != ():
    return coalesce_z(A, lift(profile, pad=None, mode=mode))
  if hasattr(A, '_coalesce_z'):
    return A._coalesce_z(profile)
  if A is None:
    return None
  if is_int(A) or is_tuple(A):
    return coalesce_z(tiler_to_layout(A), profile)
  raise TypeError(f"coalesce_z not supported for type {type(A)}")


@ModeOpDecorator
def coalesce(A, profile=1, *, mode=()):
  """
  Coalesce a Layout or Tensor into a simpler, equivalent form.

  Like `coalesce_z`, but additionally drops a trailing size-1 mode, matching C++
  `cute::coalesce`. The integral evaluation `A(i)` is preserved for every
  in-bounds `i`; the natural-coordinate evaluation generally is not, since the
  coordinate space changes. `profile` selects whole-layout (`1`) vs by-mode
  (tuple) coalescing, and `None` is a no-op. Integers and tuples are promoted via
  `tiler_to_layout`.

  A non-empty `mode` coalesces only that mode of `A` and leaves every other mode
  unchanged: `coalesce[1](A)` is `coalesce(A, (None, 1))`.

  Post-conditions:
    size(result) == size(A)
    depth(result) <= 1   at each leaf of profile, within mode
    result(i) == A(i)   for i in range(size(A))

  Examples:
    coalesce(Layout((2, (1, 6)), (1, (6, 2))))         == Layout(12, 1)
    coalesce(Layout((2, 4, 6), (24, 6, 1)))            == Layout((2, 4, 6), (24, 6, 1))
    coalesce(Layout((2, 1, 6, 1), (1, 7, 8, 0)))       == Layout((2, 6), (1, 8))
    coalesce(Layout((2, (1, 6)), (1, (6, 2))), (1, 1)) == Layout((2, 6), (1, 2))
    coalesce[1](Layout((3, (2, 6)), (1, (3, 6))))      == Layout((3, 12), (1, 3))
  """
  if mode != ():
    return coalesce(A, lift(profile, pad=None, mode=mode))
  if hasattr(A, '_coalesce'):
    return A._coalesce(profile)
  if A is None:
    return None
  if is_int(A) or is_tuple(A):
    return coalesce(tiler_to_layout(A), profile)
  raise TypeError(f"coalesce not supported for type {type(A)}")


@ModeOpDecorator
def composition(A, B: Tiler, *, mode=()):
  """
  Group composition `A o B` of Layouts/Tensors (Whitepaper, §3.3).

  Produces the layout whose domain is `B` and whose values are `A` evaluated at
  `B`'s values: walk `B`, then map the result through `A`. A tuple `B` composes
  by-mode (`(A0, A1, ...) o <X, Y, ...> = (A0 o X, A1 o Y, ...)`) and `B=None` is
  a no-op. Integers and tuples are promoted via `tiler_to_layout`.

  An `A` of `None` is the identity of unknown extents. It imposes no
  divisibility condition, and the result is the coordinates `B` itself walks,
  `tiler_to_layout(B)`.

  A non-empty `mode` composes only that mode of `A` and leaves every other mode
  unchanged.

  Pre-conditions:
    A and B satisfy the shape- and stride-divisibility conditions
    (Whitepaper, Eqs. (20)-(21)); otherwise a ValueError is raised.
    mode names a mode of A: rank[mode[:-1]](A) > mode[-1]

  Post-conditions:
    compatible(B, get[mode](result))  -- B refines result's domain
    get[mode](result)(i) == get[mode](A)(B(i))   for i in range(size(B))

  Examples:
    composition(Layout((6, 2), (8, 2)), Layout((4, 3), (3, 1))) == Layout(((2, 2), 3), ((24, 2), 8))
    composition(Layout(20, 2), Layout((5, 4), (4, 1)))          == Layout((5, 4), (8, 2))
    composition(Layout(12), Layout((4, 3)))                     == Layout((4, 3), (1, 4))
    composition[1](Layout((4, 6), (1, 4)), Layout(3, 2))        == Layout((4, 3), (1, 8))
    composition(None, (4, 3))                                   == Layout((4, 3), (E(0), E(1)))
  """
  if A is None:
    if B is None:
      return None
    B = transform_leaf(lambda b: 1 if b is None else b, B)   # No extent to keep
    return lift(tiler_to_layout(B), pad=Layout(1, 0), make=make_layout, mode=mode)
  if mode != ():
    return composition(A, replace(repeat_like(None, shape(A)), B, mode=mode))
  if hasattr(A, '_composition'):
    return A._composition(B)
  if is_int(A) or is_tuple(A):
    return composition(tiler_to_layout(A), B)
  raise TypeError(f"composition not supported for type {type(A)}")


def right_inverse(A):
  """
  Largest right inverse of a Layout.

  Returns the largest injective layout `R` that undoes `A` on `A`'s image. When
  `A`'s codomain is the integers this is the canonical right inverse, with
  `A(R(k)) == k` over the contiguous portion of the image.

  Each codomain axis is inverted by following its chain of strides in increasing
  order. Over `Z` and `Z^S` a mode continues the chain only if its stride is
  exactly the running extent `d_{k-1} * s_{k-1}`: a smaller stride overlaps
  ground already covered and a larger one leaves holes.

  Over `F2` a stride may additionally carry any component the covered modes
  already span, since XOR-ing those bits permutes them rather than colliding, so
  swizzles invert too -- `Layout((8, 8), (F2(1), F2(9)))` is its own right
  inverse. The carried component must stay inside the covered range across the
  mode's whole extent; where it does not, the chain stops there, and the result
  is still a valid right inverse, just not the largest one.

  Post-conditions:
    result(A(result(i))) == result(i)  for i in range(size(result))

  Examples:
    right_inverse(Layout((4, 8), (1, 4)))         == Layout(32, 1)
    right_inverse(Layout((4, 8), (8, 1)))         == Layout((8, 4), (4, 1))
    right_inverse(Layout((4, 8), (1, 5)))         == Layout(4, 1)
    right_inverse(Layout((8, 8), (F2(1), F2(9)))) == Layout((8, 8), (F2(1), F2(9)))
    right_inverse(Layout((8, 8), (F2(9), F2(1)))) == Layout((8, 8), (F2(8), F2(9)))
  """
  if hasattr(A, '_right_inverse'):
    return A._right_inverse()
  if A is None:
    return None
  if is_int(A) or is_tuple(A):
    return right_inverse(tiler_to_layout(A))
  raise TypeError(f"right_inverse not supported for type {type(A)}")


def left_inverse(A):
  """
  Left inverse of a Layout.

  Returns a layout `R` with `A(R(A(k))) == A(k)`; when `A` is injective this is a
  true inverse on the domain, `R(A(k)) == k`. Unlike `right_inverse`, when `A`'s
  image is non-contiguous the left inverse extends its codomain to recover the
  original coordinate.

  Pre-conditions:
    A's nonzero strides form an ordered chain: sorting the modes by stride as
    d_0 < d_1 < ... with sizes s_0, s_1, ..., each stride divides the
    next (d_{k-1} | d_k) and satisfies (d_k >= d_{k-1} * s_{k-1}).
    This is sufficient but not necessary for injectivity, so a ValueError is
    raised both for non-injective A (overlapping strides) and for the injective
    layouts whose strides cannot be chained (e.g. coprime strides).

    A gap between strides becomes an extent of the result, so the codomain's
    stride quotients must be Integers. `F2`'s quotient is a carry-less one, so an
    `F2`-strided A is supported only where every gap is 1; otherwise a ValueError
    is raised rather than returning a layout whose shape holds an `F2`.

  Post-conditions:
    A(result(A(i))) == A(i)  for i in range(size(A))

  Examples:
    left_inverse(Layout((4, 8), (1, 4))) == Layout(32, 1)
    left_inverse(Layout((4, 8), (1, 5))) == Layout((5, 8), (1, 4))
  """
  if hasattr(A, '_left_inverse'):
    return A._left_inverse()
  if A is None:
    return None
  if is_int(A) or is_tuple(A):
    return left_inverse(tiler_to_layout(A))
  raise TypeError(f"left_inverse not supported for type {type(A)}")


def complement(A, extend: Shape = None):
  """
  Complement of a Layout, optionally extended to cover `extend`.

  Returns a layout whose image fills the codomain "holes" of `A`: it is weakly
  congruent to `A`'s codomain, strictly ordered, and disjoint from `A`. The free
  `complement(A)` is the *minimal* complement; pass `extend` (a shape) to grow it
  to a target size.

  There are two regimes, governed by whether `A`'s sorted stride chain is
  *divisible* -- i.e. each running extent `d_{k-1} * s_{k-1}` divides the next
  stride `d_k` (modes ordered by stride as `d_0 <= d_1 <= ...` with sizes `s_k`):

    -- Divisible: the complement tiles, i.e. `make_layout([A, complement(A)])` is
       a bijection onto a contiguous range. This is the strong/typical case.
    -- Not divisible: the result is the *largest* ordered, disjoint layout that
       fits; it still satisfies the post-conditions below but does NOT tile (it
       under-fills the codomain).

  Pre-conditions:
    A's nonzero strides are non-overlapping (injective): each
    `d_k >= d_{k-1} * s_{k-1}`. Enforced where statically decidable; otherwise a
    ValueError is raised.

  Post-conditions:
    weakly_congruent(coprofile(A), result)
    result(i-1) < result(i)   for i in range(1, size(result))  -- ordered
    result(i) != A(j)                                          -- disjoint

  Examples:
    complement(Layout(4, 2))                      == Layout((2, 1), (1, 8))
    complement(Layout((2, 2), (1, 6)))            == Layout((3, 1), (2, 12))
    complement(Layout(4, 2), Layout(20, 1).shape) == Layout((2, 3), (1, 8))
  """
  if hasattr(A, '_complement'):
    return A._complement(extend)
  if A is None:
    return None
  if is_int(A) or is_tuple(A):
    return complement(tiler_to_layout(A), extend)
  raise TypeError(f"complement not supported for type {type(A)}")


@ModeOpDecorator
def logical_product(A, B: Tiler, *, mode=()):
  """
  Reproduce layout `A` over the layout of tiles `B`: `A x B = (A, A* o B)`.

  The rank-2 result places a copy of `A` (mode-0) at each position of `B`
  (mode-1), where mode-1 is `A`'s complement composed with `B`. A tuple `B`
  applies by-mode and `B=None` is a no-op. An integer or tuple `A` is promoted
  via `tiler_to_layout` before `B` is applied, so a by-mode `B` sees the
  promoted Layout's modes.

  A non-empty `mode` reproduces only that mode of `A` over `B` and leaves every
  other mode unchanged.

  Post-conditions:
    rank(get[mode](result)) == 2  when is_layout(B)
    size(result) == size(A) * size(B)  when is_layout(B)
    get[mode](result)[0] == get[mode](A)
    compatible(B, get[mode](result)[1])

  Examples:
    logical_product(Layout((2, 2), (4, 1)), Layout(6, 1)) == Layout(((2, 2), (2, 3)), ((4, 1), (2, 8)))
    logical_product(Layout(3, 1), Layout(4, 1))           == Layout((3, 4), (1, 3))
    logical_product[0](Layout((3, 5), (1, 20)), Layout(4, 1))
        == Layout(((3, 4), 5), ((1, 3), 20))
  """
  if mode != ():
    return logical_product(A, lift(B, pad=None, mode=mode))
  if hasattr(A, '_logical_product'):
    return A._logical_product(B)
  if is_int(A) or is_tuple(A):
    A = tiler_to_layout(A)
  if not is_layout(A):
    raise TypeError(f"logical_product not supported for type {type(A)}")
  # A is a Layout: a tuple B applies by-mode over A's modes
  if B is None:
    return A
  if is_tuple(B):
    if rank(A) < len(B): raise ValueError(f"Rank mismatch: logical_product({A}, {B})")
    return make_layout(logical_product(a,b) for a,b in zip_longest(A,B))
  if is_int(B):
    B = Layout._set(B, 1)
  if is_layout(B):
    return make_layout([A, composition(complement(A), B)])
  raise TypeError(f"logical_product not supported for tiler type {type(B)}")


@ModeOpDecorator
def logical_divide(A, B: Tiler, *, mode=()):
  """
  Split layout `A` by the tile `B`: `A / B = A o (B, B*)`.

  Mode-0 of the result is the elements selected by `B` (the *tile*); mode-1 is
  the layout of those tiles (the *grid*). A tuple `B` divides by-mode and
  `B=None` is a no-op. An integer or tuple `A` is promoted via `tiler_to_layout`
  before `B` is applied, so a by-mode `B` sees the promoted Layout's modes.

  An `A` of `None` is the identity of unknown extents, so `A o (B, B*)` takes the
  *free* complement.

  A non-empty `mode` divides only that mode of `A` and leaves every other mode
  unchanged, so `logical_divide[0, 1](A, B)` is `A` with mode `(0, 1)` replaced
  by `logical_divide(get[0, 1](A), B)`. An `A` of `None` has no modes to select,
  so `mode` names where the result lands instead, and the modes it does not name
  are filled with `1:0`.

  Pre-conditions:
    B divides A (the underlying composition's divisibility conditions hold);
    otherwise a ValueError is raised.
    mode names a mode of A: rank[mode[:-1]](A) > mode[-1]

  Post-conditions:
    rank(get[mode](result)) == 2  when is_layout(B)
    compatible(B, get[mode](result)[0])
    get[mode](result)[0] == composition(get[mode](A), B)

  Examples:
    logical_divide(Layout(24), Layout(4, 2))         == Layout((4, (2, 3)), (2, (1, 8)))
    logical_divide[1](Layout((3, 8)), Layout(4, 2))  == Layout((3, (4, 2)), (1, (6, 3)))
    logical_divide(None, Layout(4, 2))               == Layout((4, (2, 1)), (2, (1, 8)))
  """
  if A is None:
    if B is None:
      return None
    # `A o (B, B*)`, the complement left free for want of an extent to extend to
    B = transform_leaf(lambda b: tiler_to_layout(1 if b is None else b), B)
    B = transform_leaf(lambda b: make_layout([b, complement(b)]), B)
    return composition(None, B, mode=mode)
  if mode != ():
    return logical_divide(A, lift(B, pad=None, mode=mode))
  if hasattr(A, '_logical_divide'):
    return A._logical_divide(B)
  if is_int(A) or is_tuple(A):
    A = tiler_to_layout(A)
  if not is_layout(A):
    raise TypeError(f"logical_divide not supported for type {type(A)}")
  # A is a Layout: a tuple B applies by-mode over A's modes
  if B is None:
    return A
  if is_tuple(B):
    if rank(A) < len(B): raise ValueError(f"Rank mismatch: logical_divide({A}, {B})")
    return make_layout(logical_divide(a,b) for a,b in zip_longest(A,B))
  if is_int(B):
    B = Layout._set(B, 1)
  if is_layout(B):
    return composition(A, make_layout([B, complement(B, extend=shape(A))]))
  raise TypeError(f"logical_divide not supported for tiler type {type(B)}")


@ModeOpDecorator
def zipped_divide(A, B: Tiler, *, mode=()):
  """
  Logical divide of `A` by the tiler `B`, with `B` promoted to a Layout first.

  Equivalent to `logical_divide(A, tiler_to_layout(B))`. Promoting the tiler `B`
  to a single Layout zips the tile modes together and the remainder modes
  together, so the result is `((tile...), (rest...))` rather than
  `logical_divide`'s per-mode interleaving.

  A non-empty `mode` divides only that mode of `A` and leaves every other mode
  unchanged.

  Post-conditions:
    rank(get[mode](result)) == 2
    compatible(B, get[mode](result)[0])
    get[mode](result)[0] == composition(get[mode](A), B)

  Examples:
    zipped_divide(Layout((9, 32)), (Layout(3, 3), Layout((2, 4), (1, 8))))
        == Layout(((3, (2, 4)), (3, 4)), ((3, (9, 72)), (1, 18)))
    zipped_divide[1](Layout((5, 24)), Layout(4, 2)) == Layout((5, (4, (2, 3))), (1, (10, (5, 40))))
  """
  return logical_divide(A, tiler_to_layout(B), mode=mode)


def blocked_product(A, B: Tiler):
  """
  Rank-sensitive product that lays out copies of tile `A` in a *blocked*
  arrangement over `B`.

  Computes `logical_product(A, B)` and then interleaves modes so that, per mode,
  the tile factor precedes the grid factor -- each tile stays contiguous before
  the grid steps. Contrast `raked_product`, which swaps that order.

  Pre-conditions:
    rank(A) == rank(B); otherwise a ValueError is raised.

  Post-conditions:
    rank(result) == rank(A) == rank(B)

  Examples:
    blocked_product(Layout((2, 5), (5, 1)), Layout((3, 4), (1, 3)))
        == Layout(((2, 3), (5, 4)), ((5, 10), (1, 30)))
  """
  if rank(A) != rank(B):
    raise ValueError(f"Rank mismatch: blocked_product({A}, {B})")
  result = logical_product(A, B)
  return make_layout(make_layout([x,y]) for x,y in zip(result[0], result[1]))


def raked_product(A, B: Tiler):
  """
  Rank-sensitive product that distributes copies of tile `A` *raked* (cyclically
  interleaved) through `B`.

  Computes `logical_product(A, B)` and then interleaves modes so that, per mode,
  the grid factor precedes the tile factor -- the tile elements are spread across
  grid positions instead of being contiguous. Contrast `blocked_product`.

  Pre-conditions:
    rank(A) == rank(B); otherwise a ValueError is raised.

  Post-conditions:
    rank(result) == rank(A) == rank(B)

  Examples:
    raked_product(Layout((2, 5), (5, 1)), Layout((3, 4), (1, 3)))
        == Layout(((3, 2), (4, 5)), ((10, 5), (30, 1)))
  """
  if rank(A) != rank(B):
    raise ValueError(f"Rank mismatch: raked_product({A}, {B})")
  result = logical_product(A, B)
  return make_layout(make_layout([x,y]) for x,y in zip(result[1], result[0]))


def nullspace(A) -> Layout:
  """
  Nullspace of a Layout: the layout of coordinates that `A` maps to 0.

  Collects the stride-0 modes of `A` and returns a compact layout enumerating
  exactly the coordinates `c in Z(A)` with `A(c) == 0`. `None` returns `None`;
  integers and tuples are promoted via `tiler_to_layout`.

  Post-conditions:
    A(result(i)) == 0   for i in range(size(result))

  Examples:
    nullspace(Layout((4, 5), (E(0), E(1)))) == Layout(1, 0)
    nullspace(Layout((4, 5), (0, E(1))))    == Layout(4, 1)
    nullspace(Layout((2, 4, 6), (1, 2, 0))) == Layout(6, 8)
  """
  if hasattr(A, '_nullspace'):
    return A._nullspace()
  if A is None:
    return None
  if is_int(A) or is_tuple(A):
    return Layout._set(1, 0)
  raise TypeError(f"nullspace not supported for type {type(A)}")


def layout_add(A: Layout, B: Layout) -> Layout:
  """
  Add two Layouts coordinate-wise.

  Given Layouts `A` and `B` with `size(A) == size(B)`, return Layout `R` with

      size(R) == size(A) == size(B)
      R(i)    == A(i) + B(i)        for i in range(size(R))

  `A` and `B` need not be compatible.
  The result `R` is also not required to be compatible with `A` or `B`.

  When no such Layout 'R' exists, a `ValueError` is raised.

  Pre-conditions:
    size(A) == size(B)

  Post-conditions:
    size(result) == size(A)
    result(i) == A(i) + B(i)   for i in range(size(result))
    symmetric:  layout_add(A, B) == layout_add(B, A)

  Examples:
    layout_add(Layout(12, 1),          Layout(12, 1))          == Layout(12, 2)
    layout_add(Layout(5, 0),           Layout(5, 1))           == Layout(5, 1)
    layout_add(Layout((4, 3), (1, 4)), Layout((4, 3), (3, 1))) == Layout((4, 3), (4, 5))
    layout_add(Layout((5, 3, 4), (1, 5, 15)),
               Layout((10, 6),     (1, 10)))                   == Layout(60, 2)
  """
  if B is None: return A
  if A is None: return B
  if not (is_layout(A) and is_layout(B)):
    raise TypeError(f"layout_add: arguments must be Layouts (got {type(A)} and {type(B)})")
  if size(A) != size(B):
    raise ValueError(f"layout_add: size mismatch size(A)={size(A)} vs size(B)={size(B)}")

  # Reduce A and B to canonical form so that greatest_common_domain (which
  # walks the *shapes*) sees the maximally-merged factorizations actually
  # exposed by each layout's strides.
  A_co = coalesce(A)
  B_co = coalesce(B)
  G = greatest_common_domain(A_co, B_co)

  if size(G) != size(A):
    raise ValueError(
      f"layout_add: A and B have no common refinement of size {size(A)}: "
      f"greatest_common_domain({A_co}, {B_co}) = {G} (size {size(G)})"
    )

  # G has shape S = (s_0, s_1, ...) and stride D = (d_0, d_1, ...).
  # Every i in [0, size(A)) decomposes uniquely as
  #     i = sum_k c_k * d_k    with  c_k in [0, s_k)
  # via G. Because size(G) == size(A) and G's leaves align with the coalesced
  # shapes of both A and B, A and B are linear on this lattice:
  #     A(i) = sum_k c_k * A(d_k)        B(i) = sum_k c_k * B(d_k)
  # so the result has shape S and stride leaf-wise (A(d_k) + B(d_k)).
  result_stride = transform_leaf(lambda d: A_co(d) + B_co(d), G.stride)

  return Layout._set(G.shape, result_stride)._coalesce()


def greatest_common_domain(A, B) -> Layout:
  """
  Compute a Layout which selects the *greatest common domain* of two shapes.

  The result is a `Layout` whose:
    -- shape is an ordered factorization of the common divisor that `shape(A)`
       and `shape(B)` share *in order*, and
    -- stride records the offset at which each common factor appears.

  Depends only on `shape(A)` and `shape(B)`. Inputs may be ints, tuples,
  `Layout`s, or `Tensor`s (anything with `shape(...)`).

  When the leaves of `A` and `B` are pairwise coprime in their walk order
  (e.g. `(5, 3)` vs `(3, 5)`), no aligned common factor exists and the
  result is the trivial singleton `Layout((1,), (0,))`.

  Post-conditions:
    symmetric: greatest_common_domain(A, B) == greatest_common_domain(B, A)
    depth(result) == 1
    size(result) divides math.gcd(size(A), size(B))
    composition(A, result) and composition(B, result) are always admissible
    greatest_common_domain(logical_divide(shape(A), result)[1],
                           logical_divide(shape(B), result)[1]) == Layout((1,), (0,))

  Examples:
    greatest_common_domain((10,), (10,))        == Layout((10,), (1,))
    greatest_common_domain((16, 3), (16, 3))    == Layout((16, 3), (1, 16))
    greatest_common_domain((5, 3, 4), (10, 6))  == Layout((5, 2), (1, 30))
    greatest_common_domain((6, 35), (15, 14))   == Layout((3, 7), (1, 30))
    greatest_common_domain((5, 3), (3, 5))      == Layout((1,), (0,))
  """
  A = list(leaves(shape(A)))
  B = list(leaves(shape(B)))

  result_s = []
  result_d = []

  pA = pB = 1                              # Running prefix products
  i  = j  = 0
  while i < len(A) and j < len(B):
    if A[i] == 1: i += 1; continue         # Skip size-1 leaves
    if B[j] == 1: j += 1; continue

    lcm_pab = math.lcm(pA, pB)             # Smallest offset aligned in both
    rA, rB = lcm_pab // pA, lcm_pab // pB  # Residue

    if A[i] % rA == 0 and B[j] % rB == 0:
      gcd_rab = math.gcd(A[i] // rA, B[j] // rB)
      if gcd_rab != 1:
        result_s.append(gcd_rab)
        result_d.append(lcm_pab)
        A[i] //= rA * gcd_rab
        B[j] //= rB * gcd_rab
        pA = pB = lcm_pab * gcd_rab
        continue

    # Advance the leaf that ends first, both when neither end divides the other
    eA, eB = pA * A[i], pB * B[j]          # Ends of the leaves
    qA, qB = eB % eA == 0, eA % eB == 0
    if qA or not qB: pA = eA; i += 1
    if qB or not qA: pB = eB; j += 1

  if len(result_s) == 0:
    return Layout._set((1,), (0,))
  return Layout._set(tuple(result_s), tuple(result_d))
