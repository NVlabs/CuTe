# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Functions for manipulating Hierarchical Tuples
"""

from functools import reduce, update_wrapper
import operator
from itertools import zip_longest

# The ``HTuple`` / ``Profile`` aliases used to annotate these combinators (and
# the whole type vocabulary) are centralized in ``typedefs.py``.
from .typedefs import *


def is_tuple(x) -> bool:
  return isinstance(x, (tuple, list))


def congruent(a: Profile, b: Profile) -> bool:
  """
  Test whether `a` and `b` have the same hierarchical profile (Whitepaper, §2.1).

  *Congruence* is an equivalence relation on `HTuple`s: `a ~ b` iff `a` and `b`
  have matching tuple/leaf structure at every level. The leaf types/values do not
  matter -- only the shape of the tree does.

  Examples:
    congruent((4, 8), (5, 7))                 == True
    congruent(31, 42)                         == True
    congruent((4, 8), (4, (2, 4)))            == False    # different profile
    congruent(31, (4, 8))                     == False    # leaf vs tuple
    congruent((1, 1, 1), (1, 1))              == False    # different rank
  """
  if hasattr(a, '_congruent'): return a._congruent(b)
  if hasattr(b, '_congruent'): return b._congruent(a)

  if is_tuple(a) and is_tuple(b):
    return len(a) == len(b) and all(congruent(i,j) for i,j in zip(a,b))
  return not (is_tuple(a) or is_tuple(b))


def weakly_congruent(a: Profile, b: Profile) -> bool:
  """
  Test whether `a` *coarsens the profile* of `b` (Whitepaper, §2.1).

  *Weak congruence* is a partial order on `HTuple`s: `a ≲ b` iff every leaf of
  `a` may fold a sub-tree of `b`, ignoring integer values. Equivalently, the
  tuple/leaf structure of `a` can be obtained from `b` by collapsing zero or
  more sub-trees into single leaves.

  Notable consequences:
    -- An integer leaf is weakly congruent to any shape (it coarsens everything).
    -- A tuple is never weakly congruent to an integer leaf.
    -- `a ~ b`  implies  `a ≲ b`  (congruence implies weak congruence).

  Examples:
    weakly_congruent(30, (3, 4))              == True     # int coarsens any shape
    weakly_congruent(30, ((3, 4), 5))         == True
    weakly_congruent((3, 4), 30)              == False    # tuple does not coarsen int
    weakly_congruent((3, 4), (5, (6, 7)))     == True     # rank-2 vs rank-2, recurse
    weakly_congruent((3, (4, 5)), (5, 6))     == False    # (4,5) does not coarsen 6
    weakly_congruent((1, 2, 3), (1, 2))       == False    # top-level rank mismatch
  """
  if hasattr(a, '_weakly_congruent'): return a._weakly_congruent(b)

  if is_tuple(a) and is_tuple(b):
    return len(a) == len(b) and all(weakly_congruent(i,j) for i,j in zip(a,b))
  return not is_tuple(a)


def wrap(x: HTuple) -> HTuple:
  """Wrap x in a tuple if it's not already a tuple."""
  return x if is_tuple(x) else (x,)


def unwrap(x: HTuple) -> HTuple:
  """Unwrap single-element tuples recursively."""
  while is_tuple(x) and len(x) == 1: x = x[0]
  return x


def ModeOpDecorator(func):
  """
  Expose the keyword-only `mode` parameter of `func` as a subscript.

  The operation collects `mode` from its subscripts and forwards every other
  argument to `func` untouched, so an operation of any arity can be indexed.
  """
  class ModeOp:
    """
    A generic class for operations that support mode indexing.

    The subscripted modes are prepended to the `mode` given at the call site;
    all other arguments pass through to the operation:

    op(A)                <==>  op(A, mode=())        # Apply op to A with no mode filtering
    op[0](A)             <==>  op(A, mode=(0,))      # Apply op to mode 0 of A
    op[0,1](A)           <==>  op(A, mode=(0,1))     # Apply op to mode (0,1) of A
    op[0][1](A)          <==>  op(A, mode=(0,1))     # Subscripts accumulate
    op[0](A, B)          <==>  op(A, B, mode=(0,))   # Any number of arguments
    op[0](A, B, mode=1)  <==>  op(A, B, mode=(0,1))

    `mode` is keyword-only, so a mode is never mistaken for an argument of `op`.
    """

    def __init__(self, func, mode=()):
      self.func = func
      self.mode = mode
      update_wrapper(self, func)    # Present func's own name, docstring and signature

    def __call__(self, *args, mode=(), **kwargs):
      """Apply the function, prepending the subscripted modes to its `mode`."""
      return self.func(*args, mode=self.mode + tuple(wrap(mode)), **kwargs)

    def __getitem__(self, mode):
      """Return a new instance with new modes appended to existing modes."""
      return ModeOp(self.func, self.mode + tuple(wrap(mode)))

  return ModeOp(func)


@ModeOpDecorator
def get(obj: HTuple, *, mode=()) -> HTuple:
  """
  Get the mode[0]th mode, then the mode[1]th mode, etc of obj

  Args:
    obj: The object to slice into
    mode: Sequence of modes to retrieve in order

  Example:
    >>> get[0,2,3](((0, 0, (0, 0, 0, 42)),))
    42
    >>> get(((0, 0, (0, 0, 0, 42)),), mode=(0,2,3))
    42

  Post-condition:
    get[mode](lift[mode](x)) == x
  """
  if mode == ():
    return obj
  if hasattr(obj, 'get'):
    return obj.get(mode)
  return get(obj[mode[0]], mode=mode[1:])


@ModeOpDecorator
def lift(obj, *, pad=0, mode=()):
  """
  Create an object with obj as the mode-th element.

  Args:
    obj: The object to place at `mode`
    pad: The value filling the modes that `mode` does not name
    mode: Sequence of indices to apply in order

  Example:
    >>> lift[0,2,3](42)
    ((0, 0, (0, 0, 0, 42)),)
    >>> lift[1](42, pad=None)
    (None, 42)

  Post-condition:
    get[mode](lift[mode](x)) == x
  """
  result = obj
  [result := (pad,) * i + (result,) for i in reversed(mode)]
  return result


@ModeOpDecorator
def select(obj, *, mode=()):
  """
  Select the modes of obj at the given modes.
  """
  return tuple(get(obj, mode=i) for i in mode)


@ModeOpDecorator
def take(obj, *, mode=()):
  """
  Select all modes of obj between mode[0] and mode[1].
  """
  if not (len(mode) == 2 and mode[0] <= mode[1]): raise ValueError(f"take({obj}, {mode})")
  return select(obj, mode=tuple(i for i in range(mode[0], mode[1])))


def transform_apply_leaf(g, f, htuple, *tuples):
  """
  Apply f to leaf elements and g to construct results.

  (g, f, t...) = g(f(t)...)

  Args:
    gn: Function to combine results
    fn: Function to apply to leaf elements
    *tuples: Variable number of tuple arguments
  """
  if is_tuple(htuple):
    return g(transform_apply_leaf(g, f, *items) for items in zip_longest(htuple, *tuples))
  return f(htuple, *tuples)


def transform_leaf(f, *tuples):
  return transform_apply_leaf(tuple, f, *tuples)


def leaves(t: HTuple):
  if is_tuple(t):
    for x in t: yield from leaves(x)
  else:
    yield t


def zip_leaves(htuple, *tuples):
  if is_tuple(htuple):
    for x in zip_longest(htuple, *tuples): yield from zip_leaves(*x)
  else:
    yield (htuple, *tuples)


def fold_leaf(fn, v, *tuples):
  [v := fn(v, *t) for t in zip_leaves(*tuples)]
  return v


def flatten(t: HTuple, g=tuple) -> HTuple:
  return g(leaves(t))


def unflatten(iter, profile: Profile, g=tuple) -> HTuple:
  if is_tuple(profile):
    return g(unflatten(iter,p) for p in profile)
  return next(iter)


def repeat_like(x, profile: Profile) -> HTuple:
  return unflatten(iter(lambda: x, -1), profile)


def product(a: HTuple):
  return reduce(operator.mul, leaves(a), 1)


def product_each(a: HTuple) -> tuple:
  return tuple(product(x) for x in a)


def slice_(htuple: Profile, B: HTuple) -> tuple:
  """
  Recursively extract and flatten the elements of B
  where the corresponding element in htuple is None
  """
  return tuple(b for a,b in zip_leaves(htuple,B) if a is None)


def dice_(htuple: Profile, B: HTuple) -> tuple:
  """
  Recursively extract and flatten the elements of B
  where the corresponding element in htuple is not None
  """
  return tuple(b for a,b in zip_leaves(htuple,B) if a is not None)
