# Layouts

`Layout` is CuTe's core abstraction:

> A **`Layout`** is a function from coordinates to offsets, defined by a
> `Shape` and a `Stride` of the same hierarchical profile.
>
> *(Whitepaper, §2.4.)*

This chapter walks through PyCuTe's [`Layout`](../pycute/layout.py)
class: how to construct one, how to evaluate it, how to slice it, and how
to print it. The next chapter, [Layout Algebra](./04_layout_algebra.md),
covers the operations that combine layouts.

## Constructing a Layout

The signature is `Layout(shape, stride=1)`. The constructor accepts an
arbitrary [hierarchical tuple](./01_htuple.md) for `shape`, and either a
matching hierarchical tuple of stride scalars for `stride`, or a single
integer that is taken as the *base* stride.

```python
>>> from pycute import *

>>> Layout(8)                     # rank-1, default stride 1
Layout(8, 1)

>>> Layout((4, 8))                # rank-2, default column-major
Layout((4, 8), (1, 4))

>>> Layout((4, 8), (8, 1))        # rank-2, row-major
Layout((4, 8), (8, 1))

>>> Layout((4, 8), 2)             # base stride 2 ⇒ column-major (2, 8)
Layout((4, 8), (2, 8))

>>> Layout((3, (2, 4)), (24, (1, 6)))  # hierarchical
Layout((3, (2, 4)), (24, (1, 6)))
```

The constructor calls
[`prefix_product(shape, stride)`](../pycute/stride.py) on the integer
case to expand a base stride to a full stride. This means
`Layout((4, 8))` is exactly the same as `Layout((4, 8), 1)`, which is
exactly the same as `Layout((4, 8), (1, 4))`.

A layout's `Shape` and `Stride` must be **congruent** — same hierarchical
profile. This is enforced by the algebra at every operation. PyCuTe does not
run this check at construction time (because `Layout._set` skips it for speed),
but you will see compatibility errors if you violate it later.

In annotations, `Layout.__init__(self, shape: Shape, stride: Stride = 1)` uses
the [`Shape`](./02_shape_stride.md) and [`Stride`](./02_shape_stride.md) type
aliases; `Layout.__call__(self, *crd: Coord)` uses `Coord` (which admits `None`
slice-markers). See [Shapes, Strides, and Integer-Modules](./02_shape_stride.md).

For shape-and-stride combinations that have already been computed (no
prefix-product expansion required), use the private constructor:

```python
>>> Layout._set((4, 8), (8, 1))   # used by the algebra internally
Layout((4, 8), (8, 1))
```

(See [`test_layout.py::TestLayoutConstruction`](../test/test_layout.py)
for the canonical construction tests including default-stride expansion
and base-stride scaling.)

### Concatenation: `make_layout`

To build a layout out of several existing layouts, one mode per layout, use
[`make_layout`](../pycute/layout.py):

```python
>>> a = Layout(3, 1)
>>> b = Layout(4, 3)
>>> make_layout([a, b])
Layout((3, 4), (1, 3))

>>> make_layout([Layout(3, 1), Layout((5, 1), (7, 2)), Layout(2, 42)])
Layout((3, (5, 1), 2), (1, (7, 2), 42))
```

`make_layout` returns a `Layout` whose shape is the concatenation of the
input shapes and whose stride is the concatenation of the input strides.

(See [`test_make_layout.py::TestMakeLayout`](../test/test_make_layout.py).)

### From a tiler: `tiler_to_layout`

A *tiler* is a layout, an integer, or a tuple of tilers — captured by the
[`Tiler`](../pycute/layout.py) type alias
(`HTuple(Integer | Layout)`). PyCuTe converts tilers to layouts wherever it
can. The function [`tiler_to_layout`](../pycute/layout.py) makes the
conversion explicit:

```python
>>> tiler_to_layout(3)
Layout(3, 1)
>>> tiler_to_layout(Layout((7, 2), (3, 1)))
Layout((7, 2), (3, 1))
>>> tiler_to_layout((4, 5))
Layout((4, 5), (1@0, 1@1))           # coord layout
>>> tiler_to_layout((Layout(4, 2), Layout(5, 3)))
Layout((4, 5), (2@0, 3@1))           # coord-strided product
```

A `Shape` is interpreted as "tilers with stride 1 in each mode" (the
appropriate basis vectors). A tuple of layouts becomes a single rank-2
coordinate layout. Tilers are how PyCuTe expresses *by-mode* operations;
see *(Whitepaper, §3.3.5 By-mode Composition and Tilers)*.

(See [`test_make_layout.py::TestTilerToLayout`](../test/test_make_layout.py)
for the conversion table and the
`composition(A, T) == composition(A, tiler_to_layout(T))` invariant.)

## Properties of a Layout

A `Layout L = S:D` exposes its shape and stride directly:

```python
>>> A = Layout(((2, 3), 4), ((4, 8), 1))
>>> A.shape
((2, 3), 4)
>>> A.stride
((4, 8), 1)
```

It also follows the [`HTuple` mode operations](./01_htuple.md):

```python
>>> shape(A)        # same as A.shape
((2, 3), 4)
>>> shape[0](A)
(2, 3)
>>> shape[0, 1](A)
3

>>> rank(A)         # number of top-level modes
2
>>> rank[0](A)
2

>>> depth(A)        # depth of the deepest tuple in shape
2

>>> size(A)         # |L| = product of leaves of shape
24
>>> size[0](A)
6
>>> size[1](A)
4
```

`coshape(A)` returns the *codomain* bound: the smallest shape that contains
every offset `A(c)` for in-bounds coordinates `c`. `coprofile(A)` fixes only
that codomain's hierarchical structure — it is congruent to `coshape(A)`, but
its leaf values are not the extents:

```python
>>> coshape(Layout((4, 8), (1, 4)))
32                                   # max offset is 31, so codomain is 32
>>> coshape(Layout((4, 8), (E(0), E(1))))
(4, 8)
>>> coshape(Layout((4, 8), (E(0), 6 * E(1))))
(4, 43)
```

(See [`stride.py`](../pycute/stride.py) for `coshape`/`coprofile`.)

## Calling a Layout

A `Layout` is a function. Call it on any compatible coordinate.

### Integral coordinates

```python
>>> A = Layout((3, (2, 4)), (2, (1, 6)))
>>> A(0)
0
>>> A(7)
8
>>> A(23)
23
```

### Flat coordinates

A flat coordinate has one entry per top-level mode. Each entry can itself
be an integer that PyCuTe will internally pass through `idx2crd` to obtain
the natural coordinate.

```python
>>> A(1, 2)
8
>>> A(0, 7)             # 7 is a 1-D coord into the (2, 4) sub-mode
23
```

### Natural coordinates

A natural coordinate has the same hierarchy as the shape:

```python
>>> A(1, (0, 1))
8
>>> A((0, (1, 3)))      # parens are optional
23
```

All three coordinate forms produce the same offsets. The Whitepaper proves
this round-trip in *(§2.2.2 Coordinates)*. PyCuTe simply applies
`inner_product(idx2crd(crd, shape), stride)` to evaluate `A(crd)`.

(See
[`test_layout.py::TestThreeCoordinateForms`](../test/test_layout.py)
for the checked equivalence on every in-bounds coordinate.)

### Calling on out-of-bounds coordinates

`Layout.__call__` accepts integer coordinates beyond `size(A)` (the
"extended domain" in the Whitepaper). The result is a well-defined integer
offset, but it may not be in the image of `A`:

```python
>>> A(100)
99
```

This is sometimes useful for predicate generation, where the caller wants
to ask "if I extended this layout, what would the offset be?". For
in-bounds coordinates, the result is always a valid offset.

## Sublayouts: `[i]` and `.get(mode)`

`Layout.__getitem__` returns the sublayout at top-level mode `i`:

```python
>>> A = Layout((3, (2, 4)), (2, (1, 6)))
>>> A[0]
Layout(3, 2)
>>> A[1]
Layout((2, 4), (1, 6))
>>> A[1][0]
Layout(2, 1)
```

For nested access via a path, use `Layout.get(mode)`:

```python
>>> A.get([1, 0])
Layout(2, 1)
>>> A.get([1])
Layout((2, 4), (1, 6))
>>> A.get([])
Layout((3, (2, 4)), (2, (1, 6)))      # the whole layout
```

The free function `get` from [`htuple.py`](../pycute/htuple.py)
dispatches to `Layout.get` when its argument is a `Layout`:

```python
>>> get[1](A)
Layout((2, 4), (1, 6))
>>> get[1, 0](A)
Layout(2, 1)
```

To pick out **multiple** modes at once and bundle them into a single new
layout, use `select` or `take` from
[`htuple.py`](../pycute/htuple.py) with `make_layout`:

```python
>>> A = Layout((2, 3, 5, 7), (1, 2, 6, 30))
>>> make_layout(select[1, 3](A))     # modes 1 and 3
Layout((3, 7), (2, 30))
>>> make_layout(select[0, 1, 3](A))  # modes 0, 1, and 3
Layout((2, 3, 7), (1, 2, 30))
>>> make_layout(take[1, 4](A))       # modes 1 through 3 inclusive
Layout((3, 5, 7), (2, 6, 30))
```

`select[i, j, ...]` and `take[i, j]` always return a *tuple* of
sub-layouts; wrapping with `make_layout` concatenates them into a single
`Layout`. This is the moral equivalent of `cute::select<I...>(A)` and
`cute::take<Begin, End>(A)` in C++ CuTe.

(See [`test_layout.py`](../test/test_layout.py).)

## Slicing a Layout

Whenever you partially evaluate a layout — supply a definite coordinate for
some modes and `None` for others — you get back a *sublayout* paired with
an *offset*. PyCuTe exposes this through `Layout._offset_and_slice(crd)`:

```python
>>> A = Layout((4, 8), (1, 4))
>>> offset, sub = A._offset_and_slice((2, None))
>>> offset
2
>>> sub
Layout((8,), (4,))      # the column starting at row 2
```

`Tensor.__getitem__` is built on top of this — it drives `_offset_and_slice`
to get an offset and a residual layout, accumulates the offset into its
accessor, and either dereferences (if there is no residual layout) or
returns a new tensor (if there is). See
[`tensor.py`](../pycute/tensor.py).

The slicing is implemented by passing the partial coordinate to both
`slice_` and `dice_` from [`htuple.py`](../pycute/htuple.py): `dice_`
extracts the parts that contribute to the offset, and `slice_` extracts
the parts that survive into the residual layout. (See
[Hierarchical Tuples — Slicing helpers](./01_htuple.md#slicing-helpers-slice_-dice_)
and [`test_layout.py::TestLayoutSlicing`](../test/test_layout.py).)

## Printing a Layout

`str(L)` returns the canonical `Shape:Stride` notation that you have seen
throughout this documentation.

```python
>>> A = Layout((3, (2, 4)), (2, (1, 6)))
>>> print(A)
(3, (2, 4)):(2, (1, 6))
>>> repr(A)
'Layout((3, (2, 4)), (2, (1, 6)))'
```

For visualizing a 2-D layout, use
[`print_tensor`](./05_tensor.md#print_tensor) on the layout itself
(it interprets a `Layout` as an implicit-coordinate tensor):

```python
>>> from pycute.util import print_tensor
>>> print_tensor(Layout((4, 8), (1, 4)))
(4, 8):(1, 4)
0     4     8     12    16    20    24    28
1     5     9     13    17    21    25    29
2     6     10    14    18    22    26    30
3     7     11    15    19    23    27    31
```

Or for a more graphical view, save an SVG with `draw_svg` (or a LaTeX/PDF
with `draw_latex`); see [Visualization](./07_visualization.md).

## Equality

Two layouts are equal iff their `shape` and `stride` are identical
hierarchical tuples:

```python
>>> Layout(3, 1) == Layout(3, 1)
True
>>> Layout((3, 4)) == Layout((3, 4), (1, 3))
True
>>> Layout((3, 4)) == Layout((4, 3))           # different shapes
False
```

This is *structural* equality, not functional equality. Two functionally
equivalent layouts (e.g. `Layout((4, 8), (1, 4))` and `Layout(32, 1)`) are
*not* `==` to each other. To compare functional behavior, run them through
`coalesce` first or evaluate them on a range of coordinates.

(See [`test_layout.py::TestLayoutEquality`](../test/test_layout.py).)

## Worked example: a hierarchical layout

The full example from
[`test_layout.py::TestLayout::test_layout`](../test/test_layout.py):

```python
>>> A = Layout((3, (2, 4)), (2, (1, 6)))
>>> print(A)
(3, (2, 4)):(2, (1, 6))

>>> size(A)
24
>>> A[0]
Layout(3, 2)
>>> A[1]
Layout((2, 4), (1, 6))
>>> A[1][0]
Layout(2, 1)

>>> A == make_layout([Layout(3, 2),
...                   make_layout([Layout(2, 1), Layout(4, 6)])])
True
```

The image of `A` for `i = 0..23`:

```
[0, 2, 4, 1, 3, 5, 6, 8,10, 7, 9,11,12,14,16,13,15,17,18,20,22,19,21,23]
```

If you call `A(i, j)` instead of `A(i + j * size(A[0]))`, you get the same
result for every flat coordinate `(i, j)`:

```python
>>> for i in range(size(A[0])):
...     for j in range(size(A[1])):
...         I = i + j * size(A[0])
...         assert A(I) == A(i, j)
```

## Three coordinate systems, one layout

A common point of confusion is "if I have a layout of shape `(3, (2, 4))`,
do I call it as `A(i)`, `A(i, j)`, `A(i, (j, k))`, or `A((i, (j, k)))`?".
The answer: **any of those work**, and they all produce the same offset
when `(i, j)` and `(i, (j, k))` are the equivalent flat and natural forms
of `i`. CuTe is designed so that the user can choose the coordinate form
most natural for their algorithm — they don't have to commit at layout
construction time.

This is why, in idiomatic PyCuTe code, you frequently see iterations like

```python
for i in range(size(A)):              # 1-D iteration (any layout)
    A(i)

for i in range(size[0](A)):           # 2-D iteration (rank-2 layout)
    for j in range(size[1](A)):
        A(i, j)
```

Both are valid, both produce the same sequence of offsets when iterated in
colexicographical order, and both are tested by
[`test_layout.py`](../test/test_layout.py).

## Source and tests

* Source: [`pycute/layout.py`](../pycute/layout.py)
* Tests: [`test/test_layout.py`](../test/test_layout.py)
  (construction, coordinate forms, slicing, equality, coshape),
  [`test/test_make_layout.py`](../test/test_make_layout.py)
  (`make_layout`, `tiler_to_layout`, by-mode composition invariants),
  [`test/test_atuple.py`](../test/test_atuple.py)
  (coordinate-stride layouts).

## Copyright

Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
