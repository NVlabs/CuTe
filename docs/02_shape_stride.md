# Shapes, Strides, and Integer-Modules

A `Layout` is built from a `Shape` and a `Stride` of the same hierarchical
profile. This chapter walks through both halves carefully and introduces the
*integer-modules* that PyCuTe's strides are allowed to live in.

> **Naming note.** The Whitepaper (§2.3.1) introduces the more general
> *integer-semimodule* abstraction, which doesn't require an additive
> identity. PyCuTe always has a unique additive identity `int 0`, so
> each stride leaf type is the stricter *integer-module* (an abelian
> group with a `Z` scalar action). We use "module" below. `ArithTuple`
> stores its children verbatim, so rank-bearing zeros like
> `ArithTuple(0, 0)` (a rank-2 element, distinct from rank-0 `int 0`)
> participate in congruence checks; equality applies implicit
> zero-extension along trailing positions. See
> [`atuple.py`](../pycute/atuple.py).

* The Whitepaper's source-of-truth is *(Whitepaper, §2.2 Shape, §2.3 Stride)*.
* PyCuTe source: [`shape.py`](../pycute/shape.py),
  [`stride.py`](../pycute/stride.py),
  [`atuple.py`](../pycute/atuple.py),
  [`typedefs.py`](../pycute/typedefs.py).

## Shapes

> **Shape.** A `Shape` is an `HTuple` of positive integers.
>
> *(Whitepaper, §2.2.)*

In PyCuTe, a shape is just a Python integer or a Python `tuple`/`list` of
shapes. There is no `Shape` *class* — the carrier is still `int` and `tuple` —
but PyCuTe exports a documentation-grade `Shape` type *alias* (an
`HTuple(Integer)`, defined in [`typedefs.py`](../pycute/typedefs.py)) for use in
annotations such as `Layout(shape: Shape, ...)` and `crd2idx(crd, shape: Shape)`.
It is a *hint*, not a runtime check: positivity of the leaves (`Z⁺`) and
congruence with a stride remain runtime invariants verified by
[`congruent` / `weakly_congruent`](../pycute/htuple.py) and
[`compatible`](../pycute/shape.py).

```python
>>> from pycute import *
>>> shape(7)
7
>>> shape((3, 4))
(3, 4)
>>> shape(((2, 3), 4))
((2, 3), 4)
```

`shape`, `size`, `rank`, and `depth` work on shapes as `HTuple`s and on
layouts/tensors that store a `.shape` attribute:

```python
>>> S = ((2, 3), 4)
>>> size(S)        # product of leaves
24
>>> rank(S)        # number of top-level entries
2
>>> depth(S)       # depth of the deepest tuple
2
```

(See [`shape.py`](../pycute/shape.py) and [`stride.py`](../pycute/stride.py)
for the implementations; [`test_htuple.py::TestHTuple`](../test/test_htuple.py)
covers `product` / `prefix_product` / `inner_product`, and
[`test_htuple.py::TestHTuple::test_idx2crd`](../test/test_htuple.py) covers
`idx2crd` / `crd2idx`.)

### Coordinate sets

Every shape `S` defines a finite set `Z(S)` of valid coordinates, but those
coordinates can be written in three equivalent forms:

* The **integral** coordinate `i ∈ {0, 1, ..., size(S)-1}` — a single integer.
* The **flat** R-D coordinate `(c_0, ..., c_{R-1})` — one integer per
  top-level mode.
* The **natural** (or *h-D*) coordinate, which has the same hierarchical
  profile as `S`.

All three forms are captured by the `Coord` type alias (defined in
[`typedefs.py`](../pycute/typedefs.py)). `Coord` also admits `None` at any
position as a *slice-marker* — the form produced by `Tensor.__getitem__` and
consumed by `Layout._offset_and_slice` (and `idx2crd(None, S)` maps a `None`
to zeros).

For shape `(3, (2, 3))` the three sets enumerate as:

| Integral | Flat | Natural |
|---|---|---|
| 0 | (0, 0) | (0, (0, 0)) |
| 1 | (1, 0) | (1, (0, 0)) |
| 2 | (2, 0) | (2, (0, 0)) |
| 3 | (0, 1) | (0, (1, 0)) |
| ... | ... | ... |
| 17 | (2, 5) | (2, (1, 2)) |

PyCuTe's `idx2crd(i, S)` and `crd2idx(c, S)` (defined in
[`shape.py`](../pycute/shape.py)) implement the bijection between any
two coordinate forms via the **colexicographical** ordering — reading from
right to left, the leftmost mode varies fastest.

```python
>>> idx2crd(7, (3, (2, 3)))
(1, (0, 1))
>>> crd2idx((1, (0, 1)), (3, (2, 3)))
7
```

This is the exact same convention as in C++ CuTe and the CuTe DSL.

### Compatibility, congruence, and weak congruence

Three relations between shapes appear constantly in PyCuTe's post-conditions
and error messages. **`congruent`** and **`weakly_congruent`** are implemented
in [`htuple.py`](../pycute/htuple.py) (they apply to any `HTuple` profile).
**`compatible`** is implemented in [`shape.py`](../pycute/shape.py) and applies
to shapes (and objects with a `.shape`):

* **`congruent(a, b)`** — `a` and `b` have *the same profile*.
* **`weakly_congruent(a, b)`** — `a` *coarsens* `b`; every leaf of `a` may
  fold a sub-tree of `b`.
* **`compatible(a, b)`** — *every coordinate of `a` is a valid coordinate
  of `b`*. Equivalently, `a` is a coarsened shape of `b`. This is a
  partial order on shapes.

```python
>>> congruent((4, 8), (5, 7))
True
>>> congruent((4, 8), (4, (2, 4)))
False
>>> weakly_congruent(30, (1, 1))
True                                  # int coarsens any shape
>>> compatible(24, (4, 6))
True                                  # |  (4, 6)  | == 24
>>> compatible((4, 6), ((2, 2), 6))
True
>>> compatible((4, 6), ((2, 3), 8))
False
```

(See *(Whitepaper, §2.2.1)* for the full definitions,
[`test_compatibility.py`](../test/test_compatibility.py) for the
checked PyCuTe semantics, and the C++ documentation's
[01_layout.md § Layout compatibility](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md)
for an extended list of examples.)

### `common_refinement` and `common_coarsening`

When two shapes share a common refinement or coarsening in the compatibility
partial order, PyCuTe exposes join/meet helpers in [`shape.py`](../pycute/shape.py):

* **`common_refinement(a, b)`** — minimal shape refining both `a` and `b`
  (least upper bound under `≼`). Raises `ValueError` when no such shape exists.
* **`common_coarsening(a, b)`** — maximal shape coarsening both `a` and `b`
  (greatest lower bound under `≼`). Raises `ValueError` when sizes disagree.

These are used internally by advanced layout analysis (for example,
`layout_add` and `greatest_common_domain` in [`algebra.py`](../pycute/algebra.py)).

### Coordinate conversion: `idx2crd` and `crd2idx`

See the worked examples under [Coordinate sets](#coordinate-sets) above.
`idx2crd` accepts out-of-bounds integral coordinates (the final mode absorbs
overflow); `crd2idx` inverts it on in-bounds inputs.

(See [`test_htuple.py::TestHTuple::test_idx2crd`](../test/test_htuple.py).)

### `coordinates(shape)`

Generator over all natural coordinates of `shape` in colexicographical order:

```python
>>> list(coordinates((3, 2)))
[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
```

## Strides

> **Stride.** A `Stride D` for a `Shape S` is an `HTuple` *congruent* with
> `S` whose leaves are elements of an *integer-semimodule*. The stride
> defines a map from natural coordinates to a codomain `D` via
> `inner_product`.
>
> *(Whitepaper, §2.3.)*

In annotations, a stride is spelled with the `Stride` type alias — an
`HTuple(StrideScalar)`, defined in [`typedefs.py`](../pycute/typedefs.py).

In *most* code, strides are integers and the codomain is just `Z`. PyCuTe
also supports more exotic strides, all of which are *integer-modules*:

| Stride leaf | Codomain | Used for |
|---|---|---|
| Python `int` | `Z` | Common data layouts |
| `sympy.Symbol`, etc. | symbolic `Z` | Symbolic shape/stride analysis |
| `ArithTuple` (incl. basis elements `E(...)`) | `Z^S` | Coordinate layouts (TMA, predication) and arithmetic on hierarchical coordinates |
| `F2` | `F_2 = (Z, XOR, ·)` | Bank-conflict-avoiding swizzles |

This generalization is one of the key contributions of the *(Whitepaper,
§2.3.1)* and is the most significant difference between PyCuTe and the
older C++ CuTe documentation.

### `Integer` and `is_int`

PyCuTe treats a value as an "integer" iff its type is *registered* with the
abstract base class [`Integer`](../pycute/typedefs.py) — `int` (and
subclasses) automatically, `numpy.integer` / `sympy.Expr` when importable, and
anything else via `register_integer_type(...)`. Membership is a deliberate
registration, not duck-typing. `bool` and `float` are explicitly excluded, but
`sympy` symbols are integers:

```python
>>> from pycute import *
>>> is_int(7)
True
>>> is_int(True)
False
>>> import sympy
>>> N = sympy.symbols('N')
>>> is_int(N)
True
```

(See [`test_typing.py`](../test/test_typing.py).)

### `StrideScalar` and `is_stride_scalar`

A *stride scalar* is anything that supports
elementwise addition (`+`) and scalar multiplication by an integer (`*`).
That is the operations needed by `inner_product`. The abstract base class
[`StrideScalar`](../pycute/typedefs.py) registers `int` and `Integer` by
default; `ArithTuple` and `F2` are subclasses (basis elements are just
canonical `ArithTuple`s).

```python
>>> is_stride_scalar(3)
True
>>> is_stride_scalar(ArithTuple(1, 2, 3))
True
>>> is_stride_scalar(E(0))                # also an ArithTuple
True
```

### `prefix_product`, `inner_product`, and the layout default stride

The default constructor `Layout(shape)` calls
`prefix_product(shape)` to generate strides, producing a "generalized
column-major" layout — that is, the leftmost mode varies fastest in memory:

```python
>>> Layout((4, 8))
Layout((4, 8), (1, 4))
>>> Layout((3, (2, 4)))
Layout((3, (2, 4)), (1, (3, 6)))
```

You can override the stride by passing it explicitly:

```python
>>> Layout((4, 8), (8, 1))    # row-major
Layout((4, 8), (8, 1))
>>> Layout((3, (2, 4)), (24, (1, 6)))
Layout((3, (2, 4)), (24, (1, 6)))
```

If the constructor is given a single integer in place of a stride, that is
the **base** stride and the default `prefix_product(shape, base)` is
applied:

```python
>>> Layout((4, 8), 2)
Layout((4, 8), (2, 8))        # column-major with base 2
```

If you want to skip the prefix-product expansion entirely (e.g. when you
have already-computed strides), use the private constructor
`Layout._set(shape, stride)`. This is what the algebra uses internally to
avoid recomputing strides.

## Coordinate strides: `ArithTuple`, `ScaledBasis`, and `E()`

The *(Whitepaper, §2.3.1 Integer-Semimodules)* generalizes "stride"
beyond integers, and the most useful generalization in CuTe is the
**coordinate stride**: a stride leaf that produces a *coordinate* (not
an offset) when an integer scales it.

PyCuTe encodes coordinate strides as canonical
[`ArithTuple`](../pycute/atuple.py) values. A *basis element* is an
`ArithTuple` with a single nonzero leaf; the shorthand `E(*seq)` creates
a unit basis element:

```python
>>> from pycute import *
>>> E(0)             # the basis vector e_0
1@0
>>> E(1)
1@1
>>> E(0, 0)          # nested basis: e_0 of the e_0 sub-mode
1@0@0
>>> 3 * E(0)         # scaled basis 3 e_0
3@0
```

The string form `value@m_n@...@m_0` reads "scale by `value` in mode-path
`(m_0, ..., m_n)`". Equivalently, the factory `ScaledBasis(value, seq)`
represents `value` placed at the path `seq` of an otherwise-zero
arithmetic tuple:

```python
>>> ScaledBasis(42, [])         # bare scalar — collapses to int 42
42
>>> ScaledBasis(42, [0])        # (42, 0, 0, 0, 0)
42@0
>>> ScaledBasis(42, [1])        # (0, 42, 0, 0, 0)
42@1
>>> ScaledBasis(42, [1, 0])     # (0, (42, 0, ...), 0, ...)
42@0@1
```

`ScaledBasis` is a factory function, not a class — it returns the
canonical `int` or `ArithTuple` representation. To detect a basis
element at a use site, call `is_basis(x)`; to read the underlying
`(coefficient, path)` decomposition of any `ArithTuple` (single basis
element or multi-term sum), call `basis_repr(x)`.

(See [`test_atuple.py::TestArithTuple::test_sbasis`](../test/test_atuple.py) for the
full table of equivalences and
[`test_atuple.py::TestBasisRepr`](../test/test_atuple.py) /
[`test_atuple.py::TestIsBasis`](../test/test_atuple.py) for the
decomposition and predicate.)

### Arithmetic tuples

When you scale `E(0)` by `3` and `E(1)` by `5` and add them, you get a rank-2
arithmetic tuple `(3, 5)`:

```python
>>> 3 * E(0) + 5 * E(1)
(3, 5)
>>> isinstance(3 * E(0) + 5 * E(1), ArithTuple)
True
```

`ArithTuple` is closed under elementwise addition and scalar multiplication
(but *not* element-by-element multiplication, because that is not how
strides compose):

```python
>>> ArithTuple(1, 2, 3) + ArithTuple(7, 8, 9)
(8, 10, 12)
>>> ArithTuple(1, 2, 3) * 4
(4, 8, 12)
>>> 4 * ArithTuple(1, 2, 3)
(4, 8, 12)
>>> ArithTuple(1, 2, (3, 4)) + (7, 8, (9, 10))
(8, 10, (12, 14))
```

Adding a non-`ArithTuple` scalar to an `ArithTuple` is an *incompatibility*
error and raises:

```python
>>> ArithTuple(1, 2, 3) + 7      # 7 has no profile
Traceback (most recent call last):
  ...
TypeError: ArithTuple Incompatibility: ...
```

(See [`test_atuple.py::TestArithTuple::test_atuple`](../test/test_atuple.py).)

### Why use coordinate strides?

A layout `Layout((4, 8), (E(0), E(1)))` is the *identity* on its
2-D coordinate space — that is, it maps `(i, j)` to itself rather than to
some integer offset. This is exactly what you want when you need the
coordinate of a position in a tile, not the offset.

```python
>>> A = Layout((4, 8), (E(0), E(1)))
>>> A(2, 3)
(2, 3)
>>> A = Layout((4, 8), (E(1), E(0)))
>>> A(2, 3)
(3, 2)            # row and col swapped, like a transpose
```

(See [`test_atuple.py::TestArithTuple::test_coord_layout`](../test/test_atuple.py)
for several variations.)

Coordinate strides are also the way you describe predication coordinates,
TMA tensor descriptors, and the layout of "logical coordinates within a
tile". They appear pervasively in
[`test_complement.py`](../test/test_complement.py),
[`test_composition.py`](../test/test_composition.py),
[`test_inverse_left.py`](../test/test_inverse_left.py), and
[`test_inverse_right.py`](../test/test_inverse_right.py).

> **Note on conformance.** Coordinate strides are first-class in PyCuTe and
> in the *Whitepaper*. They are *not* a built-in feature of the C++ CuTe
> headers. The C++ documentation's claim that "strides are integers" is a
> simplification that this documentation deliberately corrects.

### Comparison of `ArithTuple` values

`ArithTuple` defines `__eq__`, `__lt__`, and `__gt__` along the
colexicographical ordering. This is what lets `complement` sort its
strides without knowing whether they are integers or coordinates:

```python
>>> 0 < E(0)
True
>>> ArithTuple(1, 0, 0) < ArithTuple(0, 1, 0)
True
>>> ArithTuple(idx2crd(3, (3, 4))) < ArithTuple(idx2crd(7, (3, 4)))
True
```

(See [`test_atuple.py::TestArithTuple::test_atuple_lt`](../test/test_atuple.py).)

### `make_basis_like`, `proj`, `unit`, `as_tuple`

Convenience functions defined in [`atuple.py`](../pycute/atuple.py):

* `make_basis_like(profile)` — build the matching basis tuple for a
  profile. `make_basis_like((a, (b, c)))` returns
  `(E(0), (E(1, 0), E(1, 1)))`, which prints as `(1@0, (1@0@1, 1@1@1))`.
  Used by `complement` to construct a weakly-congruent extension.
* `proj(x, profile)` — extract from `x` the part that lives at the leaf
  position of `profile`. `profile` must be a single scaled basis vector.
* `unit(profile)` — the unit basis element at `profile`'s path
  (`E()` for an integer `profile`, `E(*seq)` otherwise). `profile` must
  be a single scaled basis vector. Used by `recast` to produce a stride
  of the same *type* as the input.
* `as_tuple(obj)` — convert an `ArithTuple` into a plain Python tuple of
  integers.
* `basis_repr(x)` — algebraic decomposition of `x` into `(value, path)`
  pairs; `is_basis(x)` is true iff that decomposition has length 1.

## F2 strides — XOR-based swizzles

> Field $F_2 = (\{0, 1\}, \mathrm{XOR}, \mathrm{AND})$ is also an
> integer-module (XOR is an abelian group on `Z` with every element
> self-inverse). Strides drawn from `F_2` produce *swizzled* layouts
> where what would normally be addition becomes XOR.
>
> *(Whitepaper, §2.3.1.)*

(See [`test_swizzle.py::TestF2`](../test/test_swizzle.py) and
[`test_swizzle.py::TestF2Layout`](../test/test_swizzle.py) for
checked F2 arithmetic and worked F2-layout examples.)

PyCuTe's [`F2`](../pycute/swizzle.py) class wraps an integer and
overloads `+` to be `^` (bitwise XOR) and `*` to remain integer
multiplication:

```python
>>> from pycute import *
>>> F2(0b1010) + F2(0b1100)
F6                       # 0b0110
>>> F2(0b1010) * 2
F20
```

`F2` strides describe XOR-based swizzles directly inside a regular
`Layout` — no special wrapper is needed. See [Swizzling](./06_swizzle.md)
for examples and the broader theory.

## Codomain helpers: `coshape` and `coprofile`

Many algorithms need to know the *codomain* of a layout (the smallest
bounding box the layout's image fits inside) or the *profile* of that
codomain.

* `coshape(L)` — for `Layout` objects, returns `as_tuple(L(size(L) - 1) + 1)`
  treating each codomain leaf as a `0..max+1` range.
* `coprofile(L)` — alias for `coshape`. Used in `complement`,
  `right_inverse`, and `left_inverse` to know whether they need to produce
  an integer-codomain or a coordinate-codomain layout.

```python
>>> coshape(Layout((4, 8), (1, 4)))
32
>>> coshape(Layout((4, 8), (E(0), E(1))))
(4, 8)
>>> coshape(Layout((4, 8), (8, 1)))
32
```

## Source and tests

* Source: [`shape.py`](../pycute/shape.py),
  [`stride.py`](../pycute/stride.py),
  [`atuple.py`](../pycute/atuple.py),
  [`typedefs.py`](../pycute/typedefs.py),
  [`swizzle.py`](../pycute/swizzle.py).
* Tests:
  [`test_htuple.py`](../test/test_htuple.py),
  [`test_atuple.py`](../test/test_atuple.py),
  [`test_typing.py`](../test/test_typing.py),
  [`test_compatibility.py`](../test/test_compatibility.py),
  [`test_swizzle.py::TestF2`](../test/test_swizzle.py).

## Copyright

Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
