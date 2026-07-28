# Tensors and Accessors

A **`Tensor`** is the composition of an `Accessor` with a `Layout`:
`T = e ∘ L`. Evaluating `T(c)` runs `L(c)` to get an offset and dereferences
the accessor at that offset. In PyCuTe, use subscript syntax — `T[c]` —
not function call syntax:

> `T(c) = (e ∘ L)(c) = *(e + L(c)) = e[L(c)]`
>
> *(Whitepaper, §2.5.)*

This chapter walks through PyCuTe's [`Tensor`](../pycute/tensor.py),
the [`Accessor`](../pycute/accessor.py) hierarchy, and the
[`print_tensor`](../pycute/util/print_tensor.py) utility (in
[`pycute.util`](../pycute/util/)). `Tensor` and `Accessor` are part of the
stable public API (re-exported from `pycute` alongside the layout algebra).

All of the behaviors in this chapter are exercised by
[`test_tensor.py`](../test/test_tensor.py); each section below
links to the specific test class.

## Accessors

An accessor is the random-access pointer-like object that backs a tensor.
Two roles, two abstract base classes:

* **`Accessor`** — read-only random access. Defines `__add__` (offset) and
  `__getitem__` (dereference).
* **`MutableAccessor`** — read/write random access. Adds `__setitem__`.

PyCuTe ships three concrete accessors out of the box:

### `Array(size, dtype=ctypes.c_double)`

A heap-allocated, read/write contiguous array of `size` elements of `dtype`.
This is the production accessor for any code that needs a real backing
buffer.

```python
>>> from pycute import *
>>> import ctypes
>>> a = Array(16, dtype=ctypes.c_int)
>>> a[5] = 42
>>> a[5]
42
>>> a + 5             # ArrayView at offset 5
<...ArrayView...>
```

`Array` keeps the underlying `ctypes` storage alive (it stores a reference
in `self._raw_storage`), so it is safe to slice and re-offset without
worrying about garbage collection. Source:
[`accessor.py`](../pycute/accessor.py).

(See [`test_tensor.py::TestAccessor`](../test/test_tensor.py).)

### `ArrayView(base, offset)`

A "fat pointer" into an existing `Array` (or another `ArrayView`).
`ArrayView` uses `ctypes.cast` to compute a typed pointer at a byte offset
inside `base.ptr`. Adding to an `ArrayView` produces another `ArrayView`
at a deeper offset. PyCuTe creates these implicitly via `Array.__add__`
and `Tensor.__getitem__`.

### `ImplicitAccessor(base)`

A "no-op" accessor that does *no* dereference: dereferencing it returns
the offset itself. This is what gets used when you call
`identity_tensor` or pass a `Layout` to `print_tensor`:

```python
>>> ia = ImplicitAccessor(0)
>>> ia[5]                  # returns 0 + 5 = 5
5
>>> ia + 5                 # returns ImplicitAccessor(5)
{5}
```

`ImplicitAccessor` is what lets the same `print_tensor` rendering code
display either real tensor values *or* the offsets of a layout (which is
how the C++ `print_layout` function is implemented).

## Tensors

The `Tensor(accessor, layout)` constructor binds the two pieces together:

```python
>>> from pycute import *
>>> import ctypes
>>> T = Tensor(Array(16, ctypes.c_double), Layout((4, 4), (4, 1)))
>>> T
Tensor(<...Array...>, Layout((4, 4), (4, 1)))
```

The convenience constructor `make_tensor(layout, dtype=ctypes.c_double)`
allocates an `Array` of size equal to `coshape(layout)`:

```python
>>> T = make_tensor(Layout((4, 4), (4, 1)))
>>> T
Tensor(<Array>, Layout((4, 4), (4, 1)))
```

`make_tensor` also accepts a tiler (an `int` or hierarchical tuple) and
will promote it via `tiler_to_layout` first.

(See [`test_tensor.py::TestTensor::test_make_tensor`](../test/test_tensor.py).)

### Reading and writing

`Tensor.__getitem__` and `Tensor.__setitem__` are how you index a tensor:

```python
>>> T = make_tensor(Layout((4, 4), (4, 1)))
>>> T[1, 2] = 7.5
>>> T[1, 2]
7.5
>>> T[1, 2] += 1.0       # not supported as compound, do read-then-write
>>> T[1, 2] = T[1, 2] + 1
>>> T[1, 2]
8.5
```

When `__getitem__` is called with a coordinate that does *not* fully
specify a position, you get back a **sub-tensor** rather than a value:

```python
>>> T[1, None]               # row 1 as a 1-D tensor
Tensor(<...ArrayView at offset 4...>, Layout(4, 1))
>>> T[None, 2][1]            # column 2, then row 1
8.5
```

This is built on `Layout._offset_and_slice(crd)`: PyCuTe `dice_`s the
coordinate to compute the offset, accumulates that offset into the
accessor, and `slice_`s the layout to produce the residual layout.

Compatible `__setitem__` requires a complete coordinate:

```python
>>> T[1, None] = 0           # raises ValueError: Incomplete coordinate
Traceback (most recent call last):
  ...
ValueError: Tensor.__setitem__(...): Incomplete coordinate in setitem.
```

(See
[`test_tensor.py::TestTensor::test_tensor_setitem_and_getitem`](../test/test_tensor.py),
[`test_tensor.py::TestTensor::test_tensor_slicing_returns_subtensor`](../test/test_tensor.py),
and
[`test_tensor.py::TestTensor::test_tensor_setitem_requires_full_coord`](../test/test_tensor.py).)

### Equality

Two tensors are equal iff they share the same accessor and layout:

```python
>>> T1 = Tensor(Array(16), Layout((4, 4)))
>>> T2 = Tensor(T1.accessor, Layout((4, 4)))
>>> T1 == T2
True
```

This is an exact, structural equality. To compare element-wise values, use
the explicit loop or `print_tensor`.

## Algebraic operations on Tensors

Several layout-algebra operations dispatch to `Tensor` methods: the
operation is applied to the layout half, with the accessor passing through
unchanged. Supported today: `coalesce`, `coalesce_z`, `composition`,
`logical_divide`, and `zipped_divide` (which delegates to
`logical_divide`). Operations such as `complement`, `logical_product`,
`right_inverse`, `left_inverse`, and `nullspace` are layout-only.

```python
>>> T = make_tensor(Layout(24))
>>> coalesce(T, 1)                    # operates on T.layout
Tensor(<Array>, Layout(24, 1))
>>> composition(T, Layout((4, 6)))
Tensor(<Array>, Layout((4, 6), (1, 4)))
>>> logical_divide(T, Layout(4, 2))
Tensor(<Array>, Layout((4, (2, 3)), (2, (1, 8))))
```

Note that operations that *change the codomain* — composition with another
tensor's layout, for instance — make sense only when the resulting layout
remains a valid index into the same accessor. PyCuTe does not enforce
that bound; it is the user's responsibility.

(See
[`test_tensor.py::TestTensor::test_tensor_algebra_passes_through_to_layout`](../test/test_tensor.py).)

## `identity_tensor(shape)`

Returns a tensor that maps each coordinate to itself: a coordinate-strided
layout (`E(0), E(1), ...`) over an `ImplicitAccessor`:

```python
>>> I = identity_tensor((3, 4))
>>> I
Tensor(<...ImplicitAccessor((0, 0))...>, Layout((3, 4), (E(0), E(1))))
>>> I[1, 2]
(1, 2)
>>> I[None, 2]
Tensor(<...ImplicitAccessor((0, 2))...>, Layout(3, E(0)))
```

This is the workhorse for **predication**: build an identity tensor over
the iteration shape, slice it the same way you slice the data tensor,
and you get back the *coordinates* of every element. Compare those
coordinates to the actual matrix bounds and you have a predicate mask.

(See *(Whitepaper, §2.5.1 Slicing)* for the formal account, the C++
docs's
[0y_predication.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0y_predication.md)
for an extended example, and
[`test_tensor.py::TestIdentityTensor`](../test/test_tensor.py)
for the checked behavior.)

## `print_tensor`

[`print_tensor(t)`](../pycute/util/print_tensor.py) renders a tensor of
rank 1, 2, 3, or 4 as nested ASCII tables. It lives in
[`pycute.util`](../pycute/util/), so import it explicitly:

```python
>>> from pycute import *
>>> from pycute.util import print_tensor

>>> T = make_tensor(Layout((4, 4), (4, 1)))
>>> for i in range(size(T)): T[i] = i
>>> print_tensor(T)
<...Array...> o (4, 4):(4, 1)
0.0   4.0   8.0   12.0
1.0   5.0   9.0   13.0
2.0   6.0   10.0  14.0
3.0   7.0   11.0  15.0
```

The header shows the accessor (`<...Array...>` for a real backing array,
or `{(0, 0)}` for an `ImplicitAccessor`) followed by the layout in
canonical form. Higher-rank tensors are printed as 2-D slabs separated by
axis labels:

```python
>>> T = make_tensor(Layout((3, 3, 2)))
>>> print_tensor(T)
... 3x3 slab for k=0 ...
--------  k = 1  ---------
... 3x3 slab for k=1 ...
```

`print_tensor` also accepts a `Layout` argument directly. In that case it
prints the *offsets* (using an `ImplicitAccessor` internally):

```python
>>> print_tensor(Layout((4, 4), (4, 1)))
(4, 4):(4, 1)
0     4     8     12
1     5     9     13
2     6     10    14
3     7     11    15
```

When passed a `Layout`, `print_tensor` constructs an implicit-accessor
tensor with the supplied layout and prints it. This is exactly the C++
`print_layout` behavior.

For SVG visualization, see [Visualization](./07_visualization.md).

## Why this design?

The CuTe tensor is, by design, a *thin* abstraction. The layout encodes
all the indexing logic and the accessor encodes all the memory-management
logic. Keeping them separate has several benefits:

* **Replaceable accessors**. The same `Tensor` API works on a real
  `Array`, an `ArrayView` into someone else's storage, an
  `ImplicitAccessor` for offset/coordinate inspection, or a custom
  user-defined accessor.
* **Sliceable everywhere**. `Layout._offset_and_slice` is the single
  function that powers `Tensor.__getitem__`, `Tensor.__setitem__`, and
  `Tensor.get`. There is no separate "slice" path.
* **Composable with the algebra**. Every layout operation passes through
  to a tensor by way of `_method` dispatch in
  [`algebra.py`](../pycute/algebra.py).

## Source and tests

* Source: [`pycute/tensor.py`](../pycute/tensor.py),
  [`pycute/accessor.py`](../pycute/accessor.py),
  [`pycute/util/print_tensor.py`](../pycute/util/print_tensor.py).
* Tests: [`test/test_tensor.py`](../test/test_tensor.py)
  for `Tensor`, `Accessor`, `make_tensor`, `identity_tensor`, slicing,
  and tensor-algebra pass-through. Indirect coverage of `Tensor` also
  comes from the algebra tests
  ([`test_logical_divide.py`](../test/test_logical_divide.py),
  [`test_composition.py`](../test/test_composition.py)).

## Copyright

Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
