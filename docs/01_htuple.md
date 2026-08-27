# Hierarchical Tuples (HTuples)

CuTe is built on top of one tiny idea: a **hierarchical tuple** — a tuple
whose elements are themselves either *leaves* (e.g. integers) or other
hierarchical tuples. Shapes, strides, coordinates, and tilers are all
hierarchical tuples; everything PyCuTe does ultimately reduces to a small
set of operations on them.

This chapter covers the operations defined in
[`htuple.py`](../pycute/htuple.py) and tested by
[`test_htuple.py`](../test/test_htuple.py).

## Definition

> **HTuple.** An `HTuple(T)` is either an element of `T` (a *leaf*),
> or a `tuple` of `HTuple(T)`s.
>
> *(Whitepaper, §2.1.)*

In Python, an `HTuple(int)` is just a Python integer or a Python tuple of
`HTuple(int)`s. PyCuTe never wraps these in a custom class — built-in
`int`s and `tuple`s are the carriers.

(See [`test_htuple.py::TestHTuple::test_is_tuple`](../test/test_htuple.py).)

```python
>>> 31
>>> (16, 32)
>>> (3, -8, 7)
>>> (2, (4, 1), -1)
>>> ((4, 6), (3, (2, 2), 8))
```

For purposes of `is_tuple`, both `tuple` and `list` count as tuples:

```python
>>> from pycute import *
>>> is_tuple((1, 2, 3))
True
>>> is_tuple([1, 2, 3])
True
>>> is_tuple(7)
False
```

### The `HTuple` type alias

For readable, Whitepaper-aligned signatures, PyCuTe exports a
type alias `HTuple` plus a `Profile` synonym for the
"only the structure matters" case:

```python
HTuple  = Union[Any, tuple["HTuple", ...], list["HTuple"]]   # a leaf, or a tuple/list of HTuples
Profile = HTuple                                             # an HTuple used only for its structure
```

These are *hints*, not runtime checks — the carriers are still plain
`int`/`tuple`/`list`, and the structural contracts are enforced at runtime by
`congruent` / `weakly_congruent` (below). These definitions and the integer-leaf specializations
`IntTuple`, `Shape`, and `Coord`, and the stride specialization `Stride` live in
[`typedefs.py`](../pycute/typedefs.py). All are re-exported from the
top-level `pycute` namespace.

## Profiles, congruence, and weak congruence

The **profile** of an `HTuple` is its shape after we forget the leaf values.
Two `HTuple`s have the same profile iff they are *congruent* (`~`):

* `(4, 8) ~ (5, 7)` — same profile.
* `(4, (2, 4)) ~ (7, (3, 2))` — same profile.
* `(4, 8) ~ (4, (2, 4))` — *not* congruent: one is rank-2 with depth 1, the
  other is rank-2 with depth 2.

`P` is **weakly congruent** with `S` (`P ≲ S`) iff `P` coarsens `S` — every
leaf of `P` corresponds to either a leaf or a deeper sub-tree in `S`.

* `30 ≲ (a, b) ≲ (v, (0, α))`
* `30 ≲ (a, b, c) ≲ ((0, 0), 0, 0)`

`profile(obj)` reads that tree off any object.
A `Layout` profiles as its shape and every other `HTuple` is already its own
profile.

```python
>>> profile(Layout((2, (3, 4)))), profile((F2(1), F2(2)))
((2, (3, 4)), (F1, F2))
>>> congruent(profile(Layout((2, (3, 4)))), profile((7, ("m", None))))
True
```

PyCuTe's tests for these relations live in
[`htuple.py`](../pycute/htuple.py) (`profile`, `congruent`, `weakly_congruent`).
See
[`test_compatibility.py`](../test/test_compatibility.py) for
worked examples.

## Hierarchical access: `get`, `lift`, `wrap`, `unwrap`

### `get(obj, *, mode=())`

Index into a hierarchical structure with a *path* (a tuple of indices):

```python
>>> from pycute import *
>>> get[0, 2, 3](((0, 0, (0, 0, 0, 42)),))            # subscript form
42
>>> get(((0, 0, (0, 0, 0, 42)),), mode=(0, 2, 3))     # keyword form
42
```

(See [`test_htuple.py::TestGetLift`](../test/test_htuple.py).)

The `get[i, j, ...]` form uses a small decorator (`ModeOpDecorator`) that
also lets `size`, `rank`, `depth`, `shape`, and `stride` accept a mode-path the same way:

```python
>>> A = Layout(((2, 3), 4), ((4, 8), 1))
>>> shape(A)
((2, 3), 4)
>>> shape[0](A)            # equivalent to shape(get[0](A))
(2, 3)
>>> shape[0, 1](A)         # equivalent to shape(get[0, 1](A))
3
>>> size[1](A)
4
```

This is the moral equivalent of `cute::shape<0>(A)` and `cute::size<1>(A)` in
C++ CuTe.

### `lift(obj, *, pad=0, make=tuple, mode=())`

The dual of `get`. Inserts `obj` at a given path inside an empty
zero-padded structure:

```python
>>> lift[0, 2, 3](42)
((0, 0, (0, 0, 0, 42)),)
>>> get[0, 2, 3](lift[0, 2, 3](42))
42
>>> lift[1](42, pad=None)      # `None` says nothing about the other modes
(None, 42)
```

`make` builds each mode created, so `lift` also raises a `Layout` through the
modes of a larger one, padding with the mode that goes nowhere:

```python
>>> lift[1](Layout(4, 2), pad=Layout(1, 0), make=make_layout)
Layout((1, 4), (0, 2))
```

`lift` takes exactly one positional argument, the value: `pad`, `make` and
`mode` are all keyword-only, so none can be mistaken for the value being lifted.

`get[mode](lift[mode](x)) == x` always holds.

(See [`test_htuple.py::TestGetLift::test_lift_round_trip`](../test/test_htuple.py).)

### `replace(obj, x, *, mode=())`

The counterpart of `get`: where `lift` builds a structure around a value,
`replace` puts the value into a structure that is already there.

```python
>>> replace[1]((1, 2, 3), 42)
(1, 42, 3)
>>> replace[0, 2](((1, 2, 3), 4), 42)
((1, 2, 42), 4)
```

`get[mode](replace[mode](obj, x)) == x` always holds, and naming a mode that
`obj` does not have is a `ValueError` — unlike `lift`, which would create it.

```python
>>> replace[1](repeat_like(None, (3, 4)), Layout(2, 1))
(None, Layout(2, 1))
```

(See [`test_htuple.py::TestGetLift`](../test/test_htuple.py).)

### `select(obj, *, mode=())`

Return a tuple of the *top-level* sub-objects at the given indices. The
result is always a tuple — even for a single index — so it can be passed
directly to `make_layout` or another tuple-consuming combinator:

```python
>>> A = Layout((2, 3, 5, 7), (1, 2, 6, 30))
>>> select[1, 3](A)
(Layout(3, 2), Layout(7, 30))
>>> select[0, 1, 3](A)
(Layout(2, 1), Layout(3, 2), Layout(7, 30))
>>> select[2](A)
(Layout(5, 6),)                      # always a tuple

>>> make_layout(select[1, 3](A))     # idiomatic "give me a sublayout"
Layout((3, 7), (2, 30))
```

This is the moral equivalent of `cute::select<I...>(A)` in C++ CuTe — but
because PyCuTe's `select` returns the underlying tuple of sub-layouts, you
combine it with `make_layout` to recover the C++ "single-Layout" result.

### `take(obj, *, mode=())`

Return the contiguous range of sub-objects from index `mode[0]` to
`mode[1]` (exclusive). The result is a tuple, like `select`'s. Reverse
ranges are an error; an empty range returns the empty tuple:

```python
>>> A = Layout((2, 3, 5, 7), (1, 2, 6, 30))
>>> take[1, 3](A)
(Layout(3, 2), Layout(5, 6))
>>> take[1, 4](A)
(Layout(3, 2), Layout(5, 6), Layout(7, 30))
>>> take[1, 1](A)
()                                  # empty range, empty tuple
>>> take[3, 1](A)                   # raises ValueError
```

C++ CuTe equivalent: `cute::take<Begin, End>(A)`. Use
`make_layout(take[i, j](A))` to recover the single-Layout C++ behavior.

(See [`test_htuple.py::TestSelectTake`](../test/test_htuple.py).)

### `wrap` / `unwrap`

Small helpers used pervasively in the implementation:

* `wrap(x)` — make a 1-tuple if `x` is not a tuple already; otherwise return
  `x` unchanged.
* `unwrap(x)` — repeatedly strip 1-tuples (`((((42,))))` → `42`).

## Traversal: `leaves`, `flatten`, `unflatten`, `transform_leaf`

These functions are the workhorses inside the layout algebra.

### `leaves(htuple)` and `flatten(htuple, make=tuple)`

`leaves(htuple)` is a generator over the leaf elements of `htuple` in
left-to-right order. `flatten(htuple, make=tuple)` materializes that generator
with a builder:

```python
>>> list(leaves(((1, (2, 3)), 4)))
[1, 2, 3, 4]
>>> flatten(((1, (2, 3)), 4))
(1, 2, 3, 4)
```

### `unflatten(values, profile, make=tuple)`

The inverse of `flatten`: pull leaves from `values` to fill in the same
profile as `profile`:

```python
>>> unflatten(iter([10, 20, 30, 40]), ((1, (2, 3)), 4))
((10, (20, 30)), 40)
```

### `transform_leaf(fn, *tuples)`

Apply `fn` to each tuple of corresponding leaves and rebuild the structure:

```python
>>> transform_leaf(lambda a: a * 2, ((1, (2, 3)), 4))
((2, (4, 6)), 8)
>>> transform_leaf(lambda a, b: a + b,
...                ((1, (2, 3)), 4),
...                ((10, (20, 30)), 40))
((11, (22, 33)), 44)
```

There is also a more general
`transform_apply_leaf(make, fn, htuple, *tuples)` that lets you supply both a
leaf transformer `fn` and a node builder `make` (`make=tuple` rebuilds the
structure; `make=make_layout` builds layouts).

(See [`test_htuple.py::TestTransformLeaf`](../test/test_htuple.py).)

### `repeat_like(x, profile)`

Build a structure with the same profile as `profile`, every leaf set to `x`:

```python
>>> repeat_like(0, ((1, (2, 3)), 4))
((0, (0, 0)), 0)
```

(See [`test_htuple.py::TestFlattenUnflatten`](../test/test_htuple.py)
for `flatten`/`unflatten`/`repeat_like` round-trip examples.)

## Arithmetic: `product`, `prefix_product`, `inner_product`

PyCuTe's arithmetic over `HTuple`s is identical to C++ CuTe's.

### `product(s)`

The product of all leaves of `s`:

```python
>>> product(2)
2
>>> product((3, 2))
6
>>> product(((2, 3), 4))
24
```

(See [`test_htuple.py::TestHTuple::test_product`](../test/test_htuple.py).)

### `product_each(s)`

The product *of each top-level mode* — returns a flat tuple of the same
top-level rank as `s`, where each entry is the product of the
corresponding mode's leaves. Useful for collapsing a hierarchical shape
into the flat shape that has the same per-mode size.

```python
>>> product_each((2, 3))
(2, 3)
>>> product_each(((2, 3), 4))
(6, 4)
>>> product_each(((2, (3, 4)), (5, 6), 7))
(24, 30, 7)
```

Note that `product_each` only collapses *within* each top-level mode; it
does not recurse to a single integer like `product`.

(See [`test_htuple.py::TestHTuple::test_product_each`](../test/test_htuple.py).)

## Related operations in other modules

`product` and `product_each` live here; the stride helpers `prefix_product` and
`inner_product` are defined in [`stride.py`](../pycute/stride.py) (see
[Shape and Stride](./02_shape_stride.md)). Coordinate conversion
`idx2crd` / `crd2idx` and shape compatibility `compatible` live in
[`shape.py`](../pycute/shape.py). The `zip_leaves` helper used internally by
`inner_product` is also in [`htuple.py`](../pycute/htuple.py) but is not part of
the usual user-facing API.

## Slicing helpers: `slice_` / `dice_`

When a tensor or layout is partially evaluated (i.e., *sliced*) at some
coordinate `c`, the resulting offset is `inner_product(dice_(c, ...))` and
the surviving sublayout has shape `slice_(c, shape)` and stride
`slice_(c, stride)`. This is implemented uniformly via `slice_` and
`dice_`:

* `slice_(htuple, B)` — return a tuple of leaves of `B` whose corresponding
  leaf in `htuple` is `None`.
* `dice_(htuple, B)` — return a tuple of leaves of `B` whose corresponding
  leaf in `htuple` is *not* `None`.

The convention is that `None` is the placeholder coordinate for "give me
this whole mode back", and any other value is taken as a definite coordinate
along that mode.

```python
>>> shape = ((2, 3), (5, 7, 9))
>>> slice_(0, shape)
()
>>> dice_(0, shape)
(((2, 3), (5, 7, 9)),)
>>> slice_(None, shape)
(((2, 3), (5, 7, 9)),)
>>> dice_(None, shape)
()
>>> slice_((None, 1), shape)
((2, 3),)              # mode 0 is fully retained
>>> dice_((None, 1), shape)
((5, 7, 9),)           # mode 1 is fully consumed
>>> slice_((None, (1, None, 1)), shape)
((2, 3), 7)            # mode 0 retained, only mode 1[1] retained
```

(See [`test_htuple.py::TestHTuple::test_slice_dice`](../test/test_htuple.py).)

These two functions are how PyCuTe's `Tensor.__getitem__` and
`Layout._offset_and_slice` work — the same `coord` is fed to both, with
`dice_` driving the offset and `slice_` driving the residual layout.

## Mode-indexed operators

A surprising fraction of CuTe's API consists of "do this operation, but only
to the mode at this path". PyCuTe expresses this uniformly with the
`ModeOpDecorator`, which allows any of the following to take a path:

* [`shape`](../pycute/shape.py): `shape(A)`, `shape[0](A)`,
  `shape[0, 1](A)`.
* [`size`](../pycute/shape.py): `size(A)`, `size[0](A)`.
* [`rank`](../pycute/shape.py): `rank(A)`, `rank[0](A)`.
* [`depth`](../pycute/shape.py): `depth(A)`, `depth[0](A)`.
* [`stride`](../pycute/stride.py): `stride(A)`, `stride[1](A)`.
* [`coshape`](../pycute/stride.py) and `coprofile`.
* [`get`](../pycute/htuple.py), [`lift`](../pycute/htuple.py),
  [`replace`](../pycute/htuple.py), [`select`](../pycute/htuple.py), and
  [`take`](../pycute/htuple.py).
* [`coalesce`](../pycute/algebra.py) and `coalesce_z`: `coalesce[1](A)`.
* [`composition`](../pycute/algebra.py): `composition[0](A, B)`.
* [`logical_divide`](../pycute/algebra.py) and `zipped_divide`:
  `logical_divide[0, 1](A, B)`.
* [`logical_product`](../pycute/algebra.py): `logical_product[0](A, B)`.

In C++ CuTe these read as `cute::shape<0>(a)`, `cute::size<0,1>(a)`, etc.;
PyCuTe spells the template-argument list as a Python subscript.

The path is the operation's keyword-only `mode` parameter, which it collects from
the subscripts — everything else is an ordinary argument, so an operation of any
arity can be indexed. Subscripts accumulate (`op[0][1] == op[0, 1]`), and `mode=`
names the same path as a subscript: `shape(A, mode=(0, 1))` is `shape[0, 1](A)`.
Because `mode` is keyword-only, a path can never be mistaken for an argument of
the operation itself.

The two groups read differently. A *query* returns a property of the named mode,
so `shape[0, 1](A)` is `shape(get[0, 1](A))`. An *algebra operation* rebuilds
`A`, so `logical_divide[0, 1](A, B)` is `A` with mode `(0, 1)` replaced by
`logical_divide(get[0, 1](A), B)` and every other mode untouched.

(See [`test_htuple.py::TestModeOpDecorator`](../test/test_htuple.py)
for examples of `shape[i]`, `size[i, j]`, `rank[i]`, and `depth[i]`.)

## Where this matters

* The `Layout` constructor enforces a *congruence* condition between its
  `Shape` and its `Stride`. Both are `HTuple`s of the same profile.
* Coalescing, composition, complement, divide, and product internally call
  `flatten`, `transform_leaf`, and `unflatten` from this module, and
  `prefix_product` / `inner_product` from [`stride.py`](../pycute/stride.py).
  Following these calls is a useful exercise for understanding the algebra.
* Coordinate/index conversion is documented in
  [Shape and Stride](./02_shape_stride.md); layouts accept integral, flat, and
  natural coordinates as described there.

## Source and tests

* Source: [`pycute/htuple.py`](../pycute/htuple.py)
* Tests: [`test/test_htuple.py`](../test/test_htuple.py)

## Copyright

Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
