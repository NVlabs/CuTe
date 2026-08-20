# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Functions for manipulating Hierarchical Tuples

An *HTuple* is the container at the base of CuTe: a leaf, or a tuple/list of
HTuples. `Shape`, `Stride`, `Coord` and `Tiler` are all HTuples that differ only
in what they admit at a leaf, so the combinators here serve all of them.

Two ideas recur. A *profile* is an HTuple read for its tree alone, with the
leaves ignored; `congruent` and `weakly_congruent` compare profiles. A *mode* is
a path of indices into that tree, and every function taking one is
subscriptable, so `get[0, 2](x)` is `get(x, mode=(0, 2))`.
"""

from functools import reduce, update_wrapper
import operator
from itertools import zip_longest

# The `HTuple` / `Profile` aliases used to annotate these combinators (and
# the whole type vocabulary) are centralized in `typedefs.py`.
from .typedefs import *


def is_tuple(x) -> bool:
  """
  Test whether `x` is an HTuple internal node rather than a leaf.

  Examples:
    is_tuple((1, 2))          == True
    is_tuple([1, 2])          == True
    is_tuple(7)               == False
    is_tuple(Layout((2, 3)))  == False
  """
  return isinstance(x, (tuple, list))


def profile(obj) -> Profile:
  """
  Get an object's *profile*: its HTuple tree with the leaves left as they are.

  Congruence reads a tuple's tree and ignores whatever sits at its leaves, so
  anything that is not a `Layout` or `Tensor` already *is* its own profile.

  Notable consequences:
    -- `profile(obj) == shape(obj)` for a `Layout` or a `Tensor`.
    -- `profile(obj) is obj` for every other `HTuple`: it is already a profile.
    -- `profile` is idempotent and total: it accepts any object and rejects none.

  Examples:
    profile((2, (3, 4)))            == (2, (3, 4))
    profile(42)                     == 42
    profile(Layout((2, (3, 4))))    == (2, (3, 4))
    profile((F2(1), F2(2)))         == (F2(1), F2(2))    # a Stride has no shape
    profile((Layout(2), Layout(3))) == (Layout(2), Layout(3))    # a Tiler's leaves
  """
  if hasattr(obj, 'shape'):
    return profile(obj.shape() if callable(obj.shape) else obj.shape)
  return obj


def congruent(a: Profile, b: Profile) -> bool:
  """
  Test whether `a` and `b` have the same hierarchical profile (Whitepaper, §2.1).

  *Congruence* is an equivalence relation on `HTuple`s: `a ~ b` iff `a` and `b`
  have matching tuple/leaf structure at every level, whatever their leaves hold.

  Examples:
    congruent((4, 8), (5, 7))                 == True
    congruent(31, 42)                         == True
    congruent((4, 8), (4, (2, 4)))            == False    # different profile
    congruent(31, (4, 8))                     == False    # leaf vs tuple
    congruent((1, 1, 1), (1, 1))              == False    # different rank
    congruent((4, 8), (E(0), E(1)))           == True     # a coordinate stride
    congruent((4, 8), (F2(1), F2(8)))         == True     # an F2 stride
  """
  if is_tuple(a) and is_tuple(b):
    return len(a) == len(b) and all(congruent(i,j) for i,j in zip(a,b))
  return not (is_tuple(a) or is_tuple(b))


def weakly_congruent(a: Profile, b: Profile) -> bool:
  """
  Test whether `a` *coarsens the profile* of `b` (Whitepaper, §2.1).

  *Weak congruence* is a partial order on `HTuple`s: `a ≲ b` iff `a`'s structure
  can be obtained from `b`'s by collapsing zero or more sub-trees into leaves.

  Notable consequences:
    -- A leaf is weakly congruent to any profile (it coarsens everything).
    -- A tuple is never weakly congruent to a leaf.
    -- `a ~ b`  implies  `a ≲ b`  (congruence implies weak congruence).

  Examples:
    weakly_congruent(30, (3, 4))              == True     # a leaf coarsens any shape
    weakly_congruent(30, ((3, 4), 5))         == True
    weakly_congruent((3, 4), 30)              == False    # tuple does not coarsen leaf
    weakly_congruent((3, 4), (5, (6, 7)))     == True     # rank-2 vs rank-2, recurse
    weakly_congruent((3, (4, 5)), (5, 6))     == False    # (4,5) does not coarsen 6
    weakly_congruent((1, 2, 3), (1, 2))       == False    # top-level rank mismatch
    weakly_congruent(E(0), 8)                 == True     # a stride scalar leaf
    weakly_congruent(E(0, 0), (8, 8))         == True     # ... coarsens a shape too
  """
  if is_tuple(a) and is_tuple(b):
    return len(a) == len(b) and all(weakly_congruent(i,j) for i,j in zip(a,b))
  return not is_tuple(a)


def wrap(x: HTuple) -> HTuple:
  """
  Wrap `x` in a 1-tuple unless it is already a tuple.

  Examples:
    wrap(7)     == (7,)
    wrap((7,))  == (7,)
    wrap(())    == ()
  """
  return x if is_tuple(x) else (x,)


def unwrap(x: HTuple) -> HTuple:
  """
  Strip enclosing 1-tuples from `x`, recursively.

  Post-conditions:
    unwrap(wrap(x)) == x   for a non-tuple x; a 1-tuple is unwrapped, not restored

  Examples:
    unwrap((7,))          == 7
    unwrap(((((42,))),))  == 42
    unwrap((1, 2))        == (1, 2)
    unwrap(wrap((3,)))    == 3            # not (3,): wrap had nothing to add
  """
  while is_tuple(x) and len(x) == 1: x = x[0]
  return x


def ModeOpDecorator(func):
  """
  Expose the keyword-only `mode` parameter of `func` as a subscript.

  Subscripted modes are prepended to the `mode` given at the call site, and
  every other argument passes through untouched:

    op(A)                <==>  op(A, mode=())        # no mode filtering
    op[0](A)             <==>  op(A, mode=(0,))      # mode 0 of A
    op[0,1](A)           <==>  op(A, mode=(0,1))     # mode (0,1) of A
    op[0][1](A)          <==>  op(A, mode=(0,1))     # subscripts accumulate
    op[0](A, B)          <==>  op(A, B, mode=(0,))   # any number of arguments
    op[0](A, B, mode=1)  <==>  op(A, B, mode=(0,1))

  `mode` is keyword-only, so a mode is never mistaken for an argument of `op`.

  Examples:
    shape[1](Layout((3, (2, 4))))     == shape(Layout((3, (2, 4))), mode=(1,))
    shape[1][0](Layout((3, (2, 4))))  == 2
    size.__name__                     == 'size'
  """
  class ModeOp:

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
  Get the `mode[0]`th mode, then the `mode[1]`th mode, etc of `obj`.

  Post-conditions:
    get[mode](lift[mode](x)) == x
    get(obj) is obj

  Examples:
    get[0, 2, 3](((0, 0, (0, 0, 0, 42)),))        == 42
    get(((0, 0, (0, 0, 0, 42)),), mode=(0, 2, 3)) == 42
    get[1](Layout((3, (2, 4)), (2, (1, 6))))      == Layout((2, 4), (1, 6))
    get[1, 0]((1, (2, 3)))                        == 2
  """
  if mode == ():
    return obj
  if hasattr(obj, 'get'):
    return obj.get(mode)
  return get(obj[mode[0]], mode=mode[1:])


@ModeOpDecorator
def lift(obj, *, pad=0, make=tuple, mode=()):
  """
  Create an object with `obj` as the `mode`-th element.

  Args:
    obj: The object to place at `mode`
    pad: The value filling the modes that `mode` does not name
    make: Builds each mode created, from the sequence of its elements
    mode: Sequence of indices to apply in order

  Post-conditions:
    get[mode](lift[mode](x)) == x
    lift(x) is x

  Examples:
    lift[0, 2, 3](42)                                         == ((0, 0, (0, 0, 0, 42)),)
    lift[1](42, pad=None)                                     == (None, 42)
    lift[1](Layout(4, 2), pad=Layout(1, 0), make=make_layout)  == Layout((1, 4), (0, 2))
  """
  result = obj
  for i in reversed(mode):
    result = make((pad,) * i + (result,))
  return result


@ModeOpDecorator
def replace(obj, x, *, mode=()):
  """
  Create a copy of `obj` with its `mode`-th element replaced by `x`.

  Pre-conditions:
    `mode` names an existing element of `obj`; otherwise a ValueError is raised

  Post-conditions:
    get[mode](replace[mode](obj, x)) == x
    replace(obj, x) == x

  Examples:
    replace[1]((1, 2, 3), 42)                  == (1, 42, 3)
    replace[0, 2](((1, 2, 3), 4), 42)          == ((1, 2, 42), 4)
    replace[1](repeat_like(None, (3, 4)), 42)  == (None, 42)
    replace[3]((1, 2, 3), 42)                  -> ValueError
  """
  if mode == ():
    return x
  obj = wrap(obj)
  if mode[0] >= len(obj): raise ValueError(f"replace({obj}, {x}, {mode}): no mode {mode[0]}")
  return tuple(replace(o, x, mode=mode[1:]) if i == mode[0] else o for i,o in enumerate(obj))


@ModeOpDecorator
def select(obj, *, mode=()):
  """
  Select the modes of `obj` named by `mode`, in the order given, as a tuple.

  Post-conditions:
    len(result) == len(mode)
    result[i] == get[mode[i]](obj)

  Examples:
    A = Layout((2, 3, 5, 7), (1, 2, 6, 30))
    select[1, 3](A)               == (Layout(3, 2), Layout(7, 30))
    select[3, 1](A)               == (Layout(7, 30), Layout(3, 2))
    select[2](A)                  == (Layout(5, 6),)
    make_layout(select[1, 3](A))  == Layout((3, 7), (2, 30))
    select[0, 1]((2, (3, 4), 5))  == (2, (3, 4))
  """
  return tuple(get(obj, mode=i) for i in mode)


@ModeOpDecorator
def take(obj, *, mode=()):
  """
  Select the modes of `obj` in the half-open range `[mode[0], mode[1])`.

  Pre-conditions:
    len(mode) == 2 and mode[0] <= mode[1]; otherwise a ValueError is raised

  Post-conditions:
    take[i, j](obj) == select[tuple(range(i, j))](obj)

  Examples:
    A = Layout((2, 3, 5, 7), (1, 2, 6, 30))
    take[1, 4](A)     == (Layout(3, 2), Layout(5, 6), Layout(7, 30))
    take[1, 2](A)     == (Layout(3, 2),)
    take[2, 2](A)     == ()
    take[2, 1](A)     -> ValueError
    take[1, 2, 3](A)  -> ValueError
  """
  if not (len(mode) == 2 and mode[0] <= mode[1]): raise ValueError(f"take({obj}, {mode})")
  return select(obj, mode=tuple(i for i in range(mode[0], mode[1])))


def transform_apply_leaf(g, f, htuple, *tuples):
  """
  Rebuild `htuple` with `f` applied at every leaf and `g` at every node:
  `transform_apply_leaf(g, f, t...) == g(f(t)...)`.

  Args:
    g: Builds one node of the result from an iterable of its children
    f: Maps the corresponding leaves of every input to one leaf of the result
    htuple: The HTuple whose tree the result follows
    *tuples: Further HTuples walked alongside `htuple`

  Pre-conditions:
    weakly_congruent(htuple, t) for every t in tuples

  Examples:
    transform_apply_leaf(tuple, lambda x: x * 2, (1, (2, 3)))  == (2, (4, 6))
    transform_apply_leaf(sum, lambda x: x, (1, (2, 3)))        == 6
    transform_apply_leaf(make_layout, Layout, (2, 3), (1, 4))  == Layout((2, 3), (1, 4))
  """
  if is_tuple(htuple):
    return g(transform_apply_leaf(g, f, *items) for items in zip_longest(htuple, *tuples))
  return f(htuple, *tuples)


def transform_leaf(f, *tuples):
  """
  Apply `f` at every leaf, rebuilding the tree with plain tuples.

  `transform_apply_leaf` with `g=tuple`, which is the common case.

  Post-conditions:
    congruent(result, tuples[0])

  Examples:
    transform_leaf(lambda x: x + 1, (1, (2, 3)))       == (2, (3, 4))
    transform_leaf(lambda x, y: x * y, (2, 3), (5, 7)) == (10, 21)
  """
  return transform_apply_leaf(tuple, f, *tuples)


def leaves(t: HTuple):
  """
  Generate the leaves of `t`, left to right.

  Examples:
    tuple(leaves(((2, 3), 4)))  == (2, 3, 4)
    tuple(leaves(42))           == (42,)
    tuple(leaves(()))           == ()
  """
  if is_tuple(t):
    for x in t: yield from leaves(x)
  else:
    yield t


def zip_leaves(htuple, *tuples):
  """
  Generate the corresponding leaves of every input, as tuples.

  `htuple` drives the walk, so where it has a leaf the matching sub-trees of
  `*tuples` are yielded whole rather than descended into.

  Pre-conditions:
    weakly_congruent(htuple, t) for every t in tuples

  Examples:
    list(zip_leaves((1, 2), (3, 4)))            == [(1, 3), (2, 4)]
    list(zip_leaves((1, (2, 3)), (4, (5, 6))))  == [(1, 4), (2, 5), (3, 6)]
    list(zip_leaves((1, 2), (3, 4), (5, 6)))    == [(1, 3, 5), (2, 4, 6)]
    list(zip_leaves(1, (2, 3)))                 == [(1, (2, 3))]
  """
  if is_tuple(htuple):
    for x in zip_longest(htuple, *tuples): yield from zip_leaves(*x)
  else:
    yield (htuple, *tuples)


def fold_leaf(fn, v, *tuples):
  """
  Left-fold `fn` over the corresponding leaves of `*tuples`, starting from `v`.

  Pre-conditions:
    weakly_congruent(tuples[0], t) for every t in tuples

  Examples:
    fold_leaf(lambda acc, x: acc + x, 0, (1, (2, 3)))           == 6
    fold_leaf(lambda acc, x, y: acc + x * y, 0, (2, 3), (5, 7)) == 31
  """
  [v := fn(v, *t) for t in zip_leaves(*tuples)]
  return v


def flatten(t: HTuple, g=tuple) -> HTuple:
  """
  Collect the leaves of `t` into one flat `g`, discarding the tree.

  Post-conditions:
    depth(result) <= 1
    unflatten(iter(flatten(t)), t) == t

  Examples:
    flatten(((2, 3), 4))                          == (2, 3, 4)
    flatten(42)                                   == (42,)
    flatten((Layout(2), Layout(3)), make_layout)  == Layout((2, 3), (1, 1))
  """
  return g(leaves(t))


def unflatten(iter, profile: Profile, g=tuple) -> HTuple:
  """
  Rebuild `profile`'s tree from a flat iterator of leaves; inverse of `flatten`.

  Pre-conditions:
    `iter` yields at least as many values as `profile` has leaves

  Post-conditions:
    congruent(result, profile)
    unflatten(iter(flatten(t)), t) == t

  Examples:
    unflatten(iter([2, 3, 4]), ((0, 0), 0))  == ((2, 3), 4)
    unflatten(iter([1, 2, 3]), 0)            == 1
    unflatten(iter([2, 3]), (0, 0), list)    == [2, 3]
  """
  if is_tuple(profile):
    return g(unflatten(iter,p,g) for p in profile)
  return next(iter)


def repeat_like(x, profile: Profile) -> HTuple:
  """
  Replicate `x` at every leaf of `profile`'s tree.

  Post-conditions:
    congruent(result, profile)

  Examples:
    repeat_like(0, ((1, (2, 3)), 4))  == ((0, (0, 0)), 0)
    repeat_like(None, (3, 4))         == (None, None)
    repeat_like(0, 42)                == 0
  """
  return unflatten(iter(lambda: x, -1), profile)


def product(a: HTuple):
  """
  Multiply every leaf of `a` together.

  Examples:
    product(((2, 3), 4))  == 24
    product(42)           == 42
    product(())           == 1
  """
  return reduce(operator.mul, leaves(a), 1)


def product_each(a: HTuple) -> tuple:
  """
  The `product` of each top-level mode of `a`, as a flat tuple.

  Post-conditions:
    len(result) == len(a)
    product(result) == product(a)

  Examples:
    product_each(((2, 3), 4, (5, 6)))  == (6, 4, 30)
    product_each((2, 3))               == (2, 3)
    product_each(())                   == ()
  """
  return tuple(product(x) for x in a)


def slice_(htuple: Profile, B: HTuple, g=tuple) -> HTuple:
  """
  Collect into `g` the leaves of `B` whose counterpart in `htuple` is `None`.

  `htuple` is a coordinate whose `None` entries mark the modes a slice keeps,
  while `dice_` gathers the rest to evaluate as the slice's offset.

  Pre-conditions:
    weakly_congruent(htuple, B)

  Post-conditions:
    the leaves of slice_ and dice_ partition the leaves of B

  Examples:
    slice_((None, 1), ((2, 3), (5, 7, 9)))                 == ((2, 3),)
    slice_((None, None), (2, 3))                           == (2, 3)
    slice_((1, 2), (5, 7))                                 == ()
    slice_((None, 1), (Layout(4), Layout(8)), make_layout) == Layout((4,), (1,))
  """
  return g(b for a,b in zip_leaves(htuple,B) if a is None)


def dice_(htuple: Profile, B: HTuple, g=tuple) -> HTuple:
  """
  Collect into `g` the leaves of `B` whose counterpart in `htuple` is not `None`.

  Pre-conditions:
    weakly_congruent(htuple, B)

  Examples:
    dice_((None, 1), ((2, 3), (5, 7, 9)))  == ((5, 7, 9),)
    dice_((None, None), (2, 3))            == ()
    dice_((1, 2), (5, 7))                  == (5, 7)
  """
  return g(b for a,b in zip_leaves(htuple,B) if a is not None)
