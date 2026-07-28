# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Functions for CuTe Shapes
"""

from functools import reduce
import operator

from .typedefs import *
from .htuple import *
from .stride import *


@ModeOpDecorator
def shape(obj, mode=()) -> Shape:
  """Get an object's shape"""
  if hasattr(obj, 'shape'):       # Use .shape() or .shape if available (Layouts/Tensors/Other)
    return get(obj.shape() if callable(getattr(obj, 'shape')) else obj.shape, mode)
  if mode != ():                  # Not a Layout or Tensor, so slice once and recurse
    return shape(obj[mode[0]], mode[1:])
  if is_int(obj) or obj is None:
    return obj
  try:
    return tuple(shape(ai) for ai in obj)
  except TypeError:
    raise TypeError(f"shape({obj}, {mode})")


@ModeOpDecorator
def size(obj, mode=()) -> Integer:
  """Get an object's size"""
  return product(shape(obj, mode))


@ModeOpDecorator
def rank(obj, mode=()) -> int:
  """Get an object's rank"""
  s = shape(obj, mode)
  return len(s) if is_tuple(s) else 1


@ModeOpDecorator
def depth(obj, mode=()) -> int:
  """Get an object's depth"""
  s = shape(obj, mode)
  return 1 + reduce(max, map(depth, s)) if is_tuple(s) else 0


def compatible(a: Shape, b: Shape) -> bool:
  """
  Test whether `a` *coarsens* `b`.

  *Compatibility*, `a ≼ b`, is a partial order on shapes. It strengthens weak
  congruence by additionally requiring sizes to agree. 
  Equivalently, every coordinate of `a` (in `Z(a)`) is also a coordinate 
  of `b` (in `Z(b)`), so `Z(a) ⊆ Z(b)`.

  We say `a ≼ b`, and `a` *coarsens* `b`, and `b` *refines* `a`.

  Notable consequences:
    -- `a ≼ b`  implies  `a ≲ b`  (compatibility implies weak congruence).
    -- The least element below any shape `b` is the integer `size(b)`.

  Accepts any object that has a CuTe shape (e.g. `Layout`, `Tensor`) via `shape(...)`.

  Examples:
    compatible(30, (2, 15))                   == True     # 30 ≼ (2, 15)
    compatible((2, 15), (2, (3, 5)))          == True
    compatible(30, (2, (3, 5)))               == True     # transitivity
    compatible(24, ((2, 2), (3, 2)))          == True
    compatible(24, 32)                        == False    # size mismatch
    compatible((4, 6), ((2, 3), 8))           == False    # mode 0: 4 != 2*3
    compatible((2, (3, 5)), ((3, 2), 5))      == False    # same size, but incompatible
    compatible(24, (24,))                     == True     # int ≼ (int,)
    compatible((24,), 24)                     == False    # but not the reverse
  """
  if not (is_int(a) or is_tuple(a)): a = shape(a)
  if not (is_int(b) or is_tuple(b)): b = shape(b)

  if is_tuple(a) and is_tuple(b):
    return len(a) == len(b) and reduce(operator.and_, [compatible(i,j) for i,j in zip(a,b)], True)
  if is_int(a):
    return a == size(b)
  if is_int(b):
    return False
  raise TypeError(f"compatible({a}, {b})")


def common_refinement(a: Shape, b: Shape) -> Shape:
  """
  Find the minimal shape `c` that *refines* both `a` and `b` (Whitepaper, §2.2.1).

  Equivalently, `c` is the *join* (least upper bound) of `a` and `b` in the
  compatibility partial order on shapes:

      a ≼ c,  b ≼ c,  and c is minimal under ≼.

  Raises `ValueError` if no such shape exists.

  Notable consequences:
    -- Symmetric: `common_refinement(a, b) == common_refinement(b, a)`.
    -- Reflexive: `common_refinement(a, a) == a`.
    -- `common_refinement` exists iff `a` and `b` share at least one common
       refinement, which requires `size(a) == size(b)` and compatible profiles.

  Accepts any object that has a CuTe shape (e.g. `Layout`, `Tensor`) via `shape(...)`.

  Examples:
    common_refinement(30, (2, 15))              == (2, 15)
    common_refinement((2, 15), (2, (3, 5)))     == (2, (3, 5))
    common_refinement(10, (10,))                == (10,)
    common_refinement(((2, 3), 20), (6, (4, 5))) == ((2, 3), (4, 5))
    common_refinement((6, 5), (2, 15))          -> ValueError    # 6 != 2 at mode 0
    common_refinement((2, (3, 5)), ((3, 2), 5)) -> ValueError    # same size, but incompatible
  """
  if not (is_int(a) or is_tuple(a)): a = shape(a)
  if not (is_int(b) or is_tuple(b)): b = shape(b)

  if is_tuple(a) and is_tuple(b):
    if len(a) != len(b):
      raise ValueError(f"common_refinement: rank mismatch {a} vs {b}")
    return tuple(common_refinement(ai, bi) for ai, bi in zip(a, b))
  if is_int(a) and is_int(b):
    if a != b:
      raise ValueError(f"common_refinement: incompatible leaves {a} vs {b}")
    return a
  if is_int(a) and is_tuple(b):
    if a != size(b):
      raise ValueError(f"common_refinement: size mismatch {a} vs {b} (sizes {a} vs {size(b)})")
    return b
  if is_tuple(a) and is_int(b):
    if size(a) != b:
      raise ValueError(f"common_refinement: size mismatch {a} vs {b} (sizes {size(a)} vs {b})")
    return a
  raise TypeError(f"common_refinement({a}, {b})")


def common_coarsening(a: Shape, b: Shape) -> Shape:
  """
  Find the maximal shape `c` that *coarsens* both `a` and `b` (Whitepaper, §2.2.1).

  Equivalently, `c` is the *meet* (greatest lower bound) of `a` and `b` in the
  compatibility partial order on shapes:

      c ≼ a,  c ≼ b,  and c is maximal under ≼.

  Raises `ValueError` if no such shape exists.

  Notable consequences:
    -- Symmetric: `common_coarsening(a, b) == common_coarsening(b, a)`.
    -- Reflexive: `common_coarsening(a, a) == a`.
    -- If `size(a) == size(b)`, a common coarsening always exists -- in the worst
       case, the integer `size(a)` itself.
    -- `common_coarsening` exists iff `size(a) == size(b)`.

  Accepts any object that has a CuTe shape (e.g. `Layout`, `Tensor`) via `shape(...)`.

  Examples:
    common_coarsening((2, 15), (2, (3, 5)))     == (2, 15)
    common_coarsening((4, (3, 5)), ((2, 2), 15)) == (4, 15)
    common_coarsening(30, (2, 15))              == 30
    common_coarsening((2, (3, 5)), ((3, 2), 5)) == 30
    common_coarsening((6, 5), (2, 15))          == 30           # mode 0 mismatch -> int
    common_coarsening((2, 3), (2, 3, 1))        == 6            # rank mismatch -> int
    common_coarsening(3, 4)                     -> ValueError   # size mismatch
    common_coarsening(7, (2, 3))                -> ValueError   # size mismatch
  """
  if not (is_int(a) or is_tuple(a)): a = shape(a)
  if not (is_int(b) or is_tuple(b)): b = shape(b)

  if is_tuple(a) and is_tuple(b) and len(a) == len(b):
    try:
      return tuple(common_coarsening(ai, bi) for ai, bi in zip(a, b))
    except ValueError:
      pass

  sa, sb = size(a), size(b)
  if sa != sb:
    raise ValueError(f"common_coarsening: no common coarsening for {a} vs {b} (sizes {sa} vs {sb})")
  return sa


def idx2crd(idx: Coord, shape: Shape) -> Coord:
  """
  Map any coordinate to a *natural* coordinate of `shape`.

  Input is decomposed in *colexicographical* order (leftmost mode varies fastest).
  The final mode keeps the full quotient (its `mod` is skipped), so an 
  out-of-bounds `idx` does not wrap -- the excess accumulates in the last leaf.

  Pre-conditions:
  -- weakly_congruent(idx, shape)

  Post-conditions:
  -- congruent(result, shape)
  -- Right-inverse of `crd2idx` on in-bounds inputs:
       crd2idx(idx2crd(i, S), S) == i for i in range(size(S))

  Examples:
    idx2crd(7,    14)          == 7
    idx2crd(7,    (3, 2, 4))   == (1, 0, 1)
    idx2crd(7,    (3, (2, 4))) == (1, (0, 1))
    idx2crd(7,    ((3, 2), 4)) == ((1, 0), 1)
    idx2crd(42,   (3, 7, 2))   == (0, 0, 2)      # out of bounds: last leaf absorbs excess
    idx2crd(None, (3, (2, 4))) == (0, (0, 0))
  """
  if idx is None:                        # 0s like shape
    return repeat_like(0, shape)
  if hasattr(idx, '_idx2crd'):           # Special fn, use it
    return idx._idx2crd(shape)
  if is_int(shape) and is_int(idx):      # Identity
    return idx
  if is_tuple(shape):
    if is_tuple(idx):                    # tuple tuple
      if len(idx) != len(shape): raise ValueError(f"Incompatible rank: idx2crd({idx}, {shape})")
      return tuple(idx2crd(i,s) for i,s in zip(idx,shape))
    if is_int(idx):                      # "int" tuple
      def divmod_seq(idx):
        for s in flatten(shape)[:-1]:
          idx,rem = divmod(idx,s)        # (idx // s, idx % s)
          yield rem
        yield idx                        # Avoid mod on last shape
      return unflatten(divmod_seq(idx), shape)
  raise TypeError(f"Unknown types: idx2crd({idx}, {shape})")


def crd2idx(crd: Coord, shape: Shape) -> Integer:
  """
  Map any coordinate of `shape` to an integral coordinate.

  Input is recomposed in *colexicographical* order (leftmost mode varies fastest).

  Pre-conditions:
  -- weakly_congruent(crd, shape)

  Post-conditions:
  -- congruent(result, 0)
  -- Inverse of `idx2crd` on in-bounds inputs:
       idx2crd(crd2idx(c, S), S) == c for c in coordinates(S)
       crd2idx(idx2crd(i, S), S) == i for i in range(size(S))

  Examples:
    crd2idx((1, 0, 1),   (3, 2, 4))   == 7
    crd2idx((1, (0, 1)), (3, (2, 4))) == 7
    crd2idx(7,           (3, (2, 4))) == 7      # integral coordinate passes through
    crd2idx((2, 5),      (3, (2, 3))) == 17     # flat coordinate of a hierarchical shape
  """
  return inner_product(crd, prefix_product(transform_leaf(lambda c,s: size(s), crd, shape)))


def coordinates(shape: Shape):
  """
  Generate all natural coordinates of `shape`, in *colexicographical* order.

  Post-conditions:
    list(coordinates(s)) == [idx2crd(i, s) for i in range(size(s))]
    [crd2idx(c, s) for c in coordinates(s)] == list(range(size(s)))

  Examples:
    list(coordinates(6))           == [0, 1, 2, 3, 4, 5]
    list(coordinates((3, 2)))      == [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)]
    list(coordinates((2, (2, 2)))) == [(0,(0,0)), (1,(0,0)), (0,(1,0)), (1,(1,0)),
                                       (0,(0,1)), (1,(0,1)), (0,(1,1)), (1,(1,1))]
  """
  if is_int(shape):
    yield from range(shape)
  if is_tuple(shape):
    if len(shape) == 0:
      yield ()
    for rest in coordinates(shape[1:]):
      for c in coordinates(shape[0]):
        yield (c,) + rest
  raise TypeError(f"coordinates({shape})")