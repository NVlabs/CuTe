# PyCuTe API Reference

This is a reference index of names exported in `pycute.__all__`, with a
one-line summary, a runnable example where practical, and links to the source
and the unit test that exercises it (when one exists).

The reference is organized by module:

* [`htuple`](#module-htuple) — hierarchical tuples (`congruent`, `weakly_congruent`, …)
* [`typedefs`](#module-typedefs) — the type vocabulary: `Integer` /
  `StrideScalar` ABCs, `is_int` / `is_stride_scalar`, and the `HTuple` /
  `Profile` / `IntTuple` / `Shape` / `Coord` / `Stride` aliases
* [`stride`](#module-stride) — `stride`, `coshape`, `coprofile`,
  `inner_product`, `prefix_product`
* [`shape`](#module-shape) — `shape`, `size`, `rank`, `depth`, `compatible`,
  `common_refinement`, `common_coarsening`, `idx2crd`, `crd2idx`, `coordinates`
* [`atuple`](#module-atuple) — `ArithTuple`, `ScaledBasis`, `E`, `V`,
  `basis_repr`, `is_basis`, `make_basis_like`, `proj`, `unit`,
  `as_tuple`
* [`layout`](#module-layout) — `Layout`, `Tiler` alias, `make_layout`,
  `make_layout_like`, `make_ordered_layout`, `tiler_to_layout`, `recast`
* [`algebra`](#module-algebra) — `coalesce`, `coalesce_z`, `composition`,
  `complement`, `logical_divide`, `zipped_divide`, `logical_product`,
  `blocked_product`, `raked_product`, `right_inverse`, `left_inverse`,
  `nullspace`, `layout_add`, `greatest_common_domain`
* [`swizzle`](#module-swizzle) — `Swizzle`, `F2`, `shiftr`, `shiftl`
* [`accessor`](#module-accessor) — `Accessor`, `MutableAccessor`,
  `Ptr`, `Array`, `ImplicitAccessor`, `TransformAccessor`
* [`tensor`](#module-tensor) — `Tensor`, `is_tensor`, `make_tensor`,
  `identity_tensor`
* [`util.print_tensor`](#module-utilprint_tensor) — `print_tensor`
* [`util.print_table`](#module-utilprint_table) — `print_table`
* [`util.draw_svg`](#module-utildraw_svg) — `draw_svg`, `draw_svg_tv`
* [`util.draw_latex`](#module-utildraw_latex) — `draw_latex`,
  `draw_latex_tv`
* [`util.draw_colors`](#module-utildraw_colors) — `index_grey_8x`,
  `bank_color_8x`, `bank_color_16x`, `bank_color_32x`, `thread_color_8x`,
  `value_color_8x`, `warp_color_8x`, `white`, `constant`

For each function, click the **Test** link to see the post-condition that
PyCuTe verifies as part of its test suite.

---

## Module: `htuple`

Source: [`pycute/htuple.py`](../pycute/htuple.py)
Tests: [`test/test_htuple.py`](../test/test_htuple.py)

The `HTuple` / `Profile` type aliases (and the rest of the vocabulary) live in
[`typedefs`](#module-typedefs).

### `is_tuple(x)`

Return `True` if `x` is a `tuple` or `list`. (PyCuTe treats both as
hierarchical-tuple containers.)

```python
>>> is_tuple((1, 2)), is_tuple([1, 2]), is_tuple(7)
(True, True, False)
```

### `wrap(x)` / `unwrap(x)`

Wrap a non-tuple in a 1-tuple; recursively unwrap 1-tuples.

```python
>>> wrap(7), wrap((7,)), unwrap(((((42,))),))
((7,), (7,), 42)
```

### `front(x)` / `back(x)` / `replace_front(obj, value)` / `replace_back(obj, value)`

First / last element if `x` is a tuple; mutators that replace the first /
last element.

```python
>>> front((1, 2, 3)), back((1, 2, 3)), replace_front((1, 2, 3), 9), replace_back((1, 2, 3), 9)
(1, 3, (9, 2, 3), (1, 2, 9))
```

### `get(obj, mode=())`

Index by path (the path must be a tuple). Subscript form
`get[i, j, ...](obj)` is equivalent. Test:
[`test_htuple.py`](../test/test_htuple.py),
[`test_layout.py`](../test/test_layout.py).

```python
>>> get(((0, 0, (0, 0, 0, 42)),), (0, 2, 3))
42
```

### `lift(obj, mode=())`

Insert `obj` at a given path inside a zero-padded structure.

```python
>>> lift(42, (0, 2, 3))
((0, 0, (0, 0, 0, 42)),)
```

### `transform_apply_leaf(g, f, htuple, *tuples)` / `transform_leaf(f, *tuples)`

Apply `f` to corresponding leaves and rebuild with `g` (default `tuple`).

### `leaves(t)` / `flatten(t, g=tuple)` / `unflatten(iter, profile, g=tuple)`

Generator over leaves; tuple of leaves; inverse of flatten.

### `repeat_like(x, profile)`

Replicate `x` to match `profile`'s structure.

```python
>>> repeat_like(0, ((1, (2, 3)), 4))
((0, (0, 0)), 0)
```

### `product(a)`

Product of all leaves of `a`. Test:
[`test_htuple.py::TestHTuple::test_product`](../test/test_htuple.py).

```python
>>> product(((2, 3), 4))
24
```

### `product_each(a)`

Product of each top-level mode — returns a flat tuple. Test:
[`test_htuple.py::TestHTuple::test_product_each`](../test/test_htuple.py).

```python
>>> product_each(((2, 3), 4, (5, 6)))
(6, 4, 30)
```

### `select(obj, mode=())` / `take(obj, mode=())`

Pick out sub-objects by index. `select` takes an arbitrary sequence of
indices (`select[1, 3](A) == (A[1], A[3])`); `take` takes a `[begin, end)`
range (`take[1, 3](A) == (A[1], A[2])`). Both always return a tuple, so
combine with `make_layout(...)` to recover a single-Layout result.
Subscript form `select[i, j, ...]` / `take[i, j]` works as well. Tests:
[`test_htuple.py::TestSelectTake`](../test/test_htuple.py).

```python
>>> A = Layout((2, 3, 5, 7), (1, 2, 6, 30))
>>> select[1, 3](A)
(Layout(3, 2), Layout(7, 30))
>>> take[1, 4](A)
(Layout(3, 2), Layout(5, 6), Layout(7, 30))
>>> make_layout(select[1, 3](A))
Layout((3, 7), (2, 30))
```

### `slice_(htuple, B)` / `dice_(htuple, B)`

Extract leaves of `B` corresponding to `None`/non-`None` leaves of
`htuple`. Test:
[`test_htuple.py::TestHTuple::test_slice_dice`](../test/test_htuple.py).

```python
>>> slice_((None, 1), ((2, 3), (5, 7, 9)))
((2, 3),)
>>> dice_((None, 1), ((2, 3), (5, 7, 9)))
((5, 7, 9),)
```

### `zip_leaves(*tuples)` / `fold_leaf(fn, v, *tuples)`

Internal leaf iterators used by `inner_product` and related helpers. Exported
for advanced callers mirroring the C++ CuTe utilities.

### `congruent(a, b)` / `weakly_congruent(a, b)`

Profile relations on any `HTuple`. Tests:
[`test_compatibility.py::TestCongruent`](../test/test_compatibility.py),
[`test_compatibility.py::TestWeaklyCongruent`](../test/test_compatibility.py).

---

## Module: `typedefs`

Source: [`pycute/typedefs.py`](../pycute/typedefs.py)
Tests: [`test/test_typing.py`](../test/test_typing.py)

The home of PyCuTe's type language: the two scalar ABCs, their predicates, and
the `HTuple` type-alias vocabulary. (Only `Tiler`, whose leaf is a `Layout`,
lives elsewhere — in [`layout`](#module-layout).)

### `class Integer`

Abstract base class for *integer-shaped* scalar types. Membership is decided
by **registration**, not duck-typing: `int` (and subclasses) are recognized
automatically, `bool` and `float` are explicitly excluded, and any other type
is included only via `register_integer_type(...)` / `Integer.register(...)`
(`numpy.integer` and `sympy.Expr` are auto-registered when importable). So
`int` and `sympy` symbols are `Integer`s, but `F2` — whose `+` is XOR — is
deliberately not.

```python
>>> isinstance(7, Integer), isinstance(True, Integer), isinstance(1.0, Integer)
(True, False, False)
>>> import sympy; isinstance(sympy.symbols('x'), Integer)
True
```

### `is_int(x)`

Equivalent to `isinstance(x, Integer)`.

### `is_static(x)`

True iff `x` is an `Integer` whose value is known at "compile time" (plain
`int`, or a registered integer type with a fixed value).

### `divmod(a, b)`

Quotient and remainder `(a // b, a % b)`. Dispatches through the built-in `divmod`
protocol (honoring a fused `__divmod__` / `__rdivmod__`) and otherwise falls back to `//` and `%`.

### `register_integer_type(*types)`

Declare one or more types as integer-shaped (`Integer.register(...)` for each),
so that `is_int` and PyCuTe's shape/coordinate code treat their instances as
integers. Idempotent.

### `class StrideScalar`

Abstract base class for stride leaf types — the integer-semimodule elements a
`Layout`'s stride may hold. Requires `__add__`, `__radd__`, `__mul__`,
`__rmul__`. `int` and `Integer` are registered; `ArithTuple` and `F2` are
subclasses.

### `is_stride_scalar(value)`

True iff `value` is a `StrideScalar`.

### Type aliases: `HTuple`, `Profile`, `IntTuple`, `Shape`, `Coord`, `Stride`

Documentation-grade aliases (3.10-compatible `TypeAlias` + string forward-refs)
— *hints*, not runtime checks. The structural contracts (congruence, positivity,
`None`-handling) are enforced at runtime by `congruent` / `weakly_congruent` /
`compatible` and the `is_*` predicates. Because both scalar ABCs are defined in
this module, every alias below is a direct reference (no forward-refs).

* `HTuple` — `Union[Any, tuple["HTuple", ...], list["HTuple"]]`: a leaf, or a
  tuple/list of `HTuple`s.
* `Profile` — a synonym for `HTuple`, used where only the tree structure matters.
* `IntTuple` — `Union[Integer, tuple["IntTuple", ...], list["IntTuple"]]`: the
  shared carrier of shapes and coordinates.
* `Shape` — an `IntTuple` whose leaves are positive (`Z⁺`).
* `Coord` — `Union[Integer, None, tuple["Coord", ...], list["Coord"]]`: an
  `IntTuple` that additionally admits `None` slice-markers at any position.
* `Stride` — `Union[StrideScalar, tuple["Stride", ...], list["Stride"]]`: an
  `HTuple(StrideScalar)`, congruent (at runtime) with a layout's `Shape`.

The `Tiler` alias (`HTuple(Integer | Layout)`) lives in
[`layout`](#module-layout), beside the `Layout` class it references.

---

## Module: `stride`

Source: [`pycute/stride.py`](../pycute/stride.py)
Tests: throughout (`test_atuple.py`, `test_htuple.py`, etc.).

The `Stride` type alias and the `StrideScalar` ABC now live in
[`typedefs`](#module-typedefs).

### `stride(obj, mode=())`

Get the stride of a layout/tensor or pass through a stride literal.
Subscript form `stride[i](obj)` works the same way.

### `inner_product(a, b)`

Sum-of-products at every leaf. The fundamental layout-evaluation operator.
Test: [`test_htuple.py::TestHTuple::test_inner_product`](../test/test_htuple.py).

```python
>>> inner_product(((2, 3), 4), ((2, 1), 2))
15
```

### `prefix_product(a, init=1)`

Exclusive prefix product. Used to expand a base stride into a full stride.
Test: [`test_htuple.py::TestHTuple::test_prefix_product`](../test/test_htuple.py).

```python
>>> prefix_product((3, 2, 4))
(1, 3, 6)
>>> prefix_product(((2, 3), (2, 1, 2), (5, 2, 1)))
((1, 2), (6, 12, 12), (24, 120, 240))
```

### `coshape(obj, mode=())` / `coprofile(obj, mode=())`

Codomain shape and codomain profile of a layout. `coshape` computes the
codomain's extents; `coprofile` fixes only its tuple/leaf structure and is
congruent to `coshape`, not equal to it. Taken straight from the strides,
`coprofile` needs no extent arithmetic and never interprets a codomain leaf, so
it stays defined for codomains whose extents `coshape` cannot bound.

```python
>>> coshape(Layout((4, 8), (1, 4)))
32
>>> coshape(Layout((4, 8), (E(0), E(1))))
(4, 8)
```

---

## Module: `shape`

Source: [`pycute/shape.py`](../pycute/shape.py)
Tests: [`test/test_htuple.py`](../test/test_htuple.py),
[`test/test_compatibility.py`](../test/test_compatibility.py).

The `IntTuple` / `Shape` / `Coord` type aliases live in
[`typedefs`](#module-typedefs).

### `shape(obj, mode=())`

Shape of a layout/tensor; pass-through for tuples and integers.

### `size(obj, mode=())`

Product of leaves of `shape(obj, mode)`.

### `rank(obj, mode=())`

Number of top-level entries of `shape(obj, mode)`.

### `depth(obj, mode=())`

Depth of the deepest tuple in `shape(obj, mode)`.

### `compatible(a, b)`

True iff `a` coarsens `b` and `size(a) == size(b)`. (See
*(Whitepaper, §2.2.1)*.) Tests:
[`test_compatibility.py::TestCompatible`](../test/test_compatibility.py).

### `common_refinement(a, b)` / `common_coarsening(a, b)`

Join/meet helpers on the compatibility partial order of shapes. Each raises
`ValueError` when no common refinement/coarsening exists. Used by
`layout_add` and `greatest_common_domain`.

### `idx2crd(idx, shape)`

Map an index (or arbitrary coordinate) to the natural coordinate of
`shape`. Test:
[`test_htuple.py::TestHTuple::test_idx2crd`](../test/test_htuple.py).

```python
>>> idx2crd(7, (3, (2, 4)))
(1, (0, 1))
```

### `crd2idx(crd, shape)`

Inverse of `idx2crd` on in-bounds inputs.

```python
>>> crd2idx((1, (0, 1)), (3, (2, 4)))
7
```

### `coordinates(shape)`

Generator over natural coordinates of `shape` in colexicographical order.

---

## Module: `atuple`

Source: [`pycute/atuple.py`](../pycute/atuple.py)
Tests: [`test/test_atuple.py`](../test/test_atuple.py)

### `class ArithTuple`

Hierarchical tuple-of-integers under elementwise addition and scalar
multiplication, with implicit zero-extension along trailing positions.
The single carrier for both *coordinate strides* (one-term sums like
`E(0)`) and *coordinate sums* (multi-term sums like `3*E(0) + 5*E(1)`).

`ArithTuple` stores its children verbatim, preserving rank/profile
information: `ArithTuple(0, 0)` is a rank-2 zero, distinct from
rank-0 `int 0`. Equality (`__eq__`) applies implicit zero-extension
along trailing positions, so three syntactically-different
constructions of "the unit basis at position 0" compare equal:

```python
>>> E(0) == ArithTuple(1, 0) == ArithTuple((1,))
True
```

`ArithTuple` is **unhashable**: equivalent representations have
distinct `data`, so a structural hash would violate
`a == b => hash(a) == hash(b)`.

Closed under `+`, `-`, and scalar `*`. Addition forms an abelian *group* —
`int 0` is the unique identity and every element has a negation — so `-` is
elementwise too; unlike `+` it does not commute, so `0 - x` negates `x`.
Adding or subtracting a nonzero scalar is an incompatibility error. Tests:
[`test_atuple.py::TestArithTuple::test_atuple`](../test/test_atuple.py),
[`test_atuple_sub`](../test/test_atuple.py).

```python
>>> ArithTuple(1, 2, 3) + (7, 8, 9)
(8, 10, 12)
>>> ArithTuple(1, 2, 3) * 4
(4, 8, 12)
```

#### `ArithTuple._set(data)`

Internal trust-the-caller constructor analogous to `Layout._set`.
Stores `data` as `self.data` verbatim, given that every element is
already an `int` or an `ArithTuple` (no nested raw tuples/lists to
lift). Used by every internal builder to skip the recursive
sub-tuple lift performed by `__new__`.

### `ScaledBasis(value, seq=())`

A scaled basis vector at path `seq`, built as a factory: returns the
`int` `value` when `seq` is empty, otherwise an `ArithTuple` whose
innermost child is `value` at the path described by `seq`. Test:
[`test_atuple.py::TestArithTuple::test_sbasis`](../test/test_atuple.py).

```python
>>> ScaledBasis(42, [])         # empty seq -> the int value itself
42
>>> ScaledBasis(42, [0])        # 42 at path (0,)
42@0
>>> ScaledBasis(42, [1, 0])     # 42 at path (1, 0)
42@0@1
```

### `class V(value)` and `V(value) @ i`

Shortcut: `V(v) @ i` builds `ScaledBasis(v, (i,))`.

### `E(*seq)`

Unit basis: `ScaledBasis(1, seq)`. The most common way to write a
coordinate stride.

```python
>>> E(0)        # (1, 0, 0, ...)
1@0
>>> E(1)        # (0, 1, 0, ...)
1@1
>>> E(0, 1)     # ((0, 1, 0, ...), 0, ...)
1@1@0
```

### `basis_repr(x)`, `is_basis(x)`

`basis_repr(x)` returns the algebraic decomposition of `x` as a
non-empty list of `(value, seq)` pairs satisfying

```
x == sum(v * E(*s) for v, s in basis_repr(x))
```

Each entry corresponds to one nonzero leaf of `x` with its path. When
every leaf of `x` is zero (an `int 0`, or an `ArithTuple` whose every
leaf is zero), the decomposition collapses to the single rank-zero
term `[(0, ())]` — matching the rank-zero decomposition of `int 0`.

`is_basis(x)` is a thin predicate over that decomposition: it is true
iff `x` is a single scaled basis vector `v * E(*s)` (including every
Python `int`, which is the rank-zero basis element). Tests:
[`test_atuple.py::TestBasisRepr`](../test/test_atuple.py),
[`test_atuple.py::TestIsBasis`](../test/test_atuple.py).

```python
>>> b = 5 * E(1, 2)
>>> basis_repr(b)
[(5, (1, 2))]
>>> is_basis(b)
True
>>> basis_repr(0), is_basis(0)
([(0, ())], True)
>>> is_basis(3 * E(0) + 5 * E(1))     # multi-term sum, not a basis element
False
>>> basis_repr(3 * E(0) + 5 * E(1))   # still decomposes
[(3, (0,)), (5, (1,))]
```

### `make_basis_like(profile, seq=())`

Build a basis tuple matching `profile`'s shape:

```python
>>> make_basis_like((10, (20, 30)))
(1@0, (1@0@1, 1@1@1))                # = (E(0), (E(1, 0), E(1, 1)))
```

### `proj(x, profile)`

Extract from `x` the leaf at the path implied by `profile`. `profile`
must be a single scaled basis vector (any `int`, or an `ArithTuple`
with exactly one nonzero leaf); a multi-term sum raises `TypeError`.
Tests: [`test_atuple.py::TestProjAndUnit`](../test/test_atuple.py).

### `unit(profile)`

The multiplicative unit of `profile`'s algebra, at `profile`'s basis path:
`E()` (which collapses to `int 1`) when `profile` is an integer, `E(*seq)`
when `profile` is a single basis element with path `seq`, and `F2(1)` for an
`F2`. Multi-term sums raise `TypeError`.

Dropping the magnitude while keeping the algebra and axis is what lets
`unit(d) * n` rebuild a stride of magnitude `n` in the same place, as `recast`
does to produce a stride of the same *type* as its input. A scalar type
supplies a `_unit` hook when its identity is not an `int` / `ArithTuple` basis
element; `F2` does, since scaling `int 1` would stay in `Z` and multiply with
carries. Tests: [`test_atuple.py::TestProjAndUnit`](../test/test_atuple.py),
[`test_swizzle.py::TestF2`](../test/test_swizzle.py).

### `as_tuple(obj)`

Convert an `ArithTuple` to a plain nested Python tuple.

```python
>>> as_tuple(ArithTuple(1, 2, (3, 4)))
(1, 2, (3, 4))
```

---

## Module: `layout`

Source: [`pycute/layout.py`](../pycute/layout.py)
Tests: [`test/test_layout.py`](../test/test_layout.py),
[`test/test_atuple.py`](../test/test_atuple.py).

### `class LayoutBase`

Marker class for any layout. `is_layout(x)` checks `isinstance(x, LayoutBase)`.

### `class Layout(shape, stride=1)`

A `Shape:Stride` pair. Calling `L(crd)` evaluates the layout. The class
also exposes `__getitem__`, `get`, `__call__`, and the methods used by
the dispatcher in `algebra.py` (`_coalesce`, `_coalesce_z`,
`_composition`, `_right_inverse`, `_left_inverse`, `_complement`,
`_logical_divide`, `_logical_product`, `_nullspace`).

Test: [`test_layout.py::TestLayout::test_layout`](../test/test_layout.py).

```python
>>> A = Layout((3, (2, 4)), (2, (1, 6)))
>>> A(17), A(2, 5), A(2, (1, 2))
(17, 17, 17)
>>> A[1][0]
Layout(2, 1)
```

### `Layout._set(_shape, _stride)` (classmethod)

Skip `prefix_product` expansion. Used by the algebra to build layouts
without recomputing strides.

### `is_layout(x)`

True iff `isinstance(x, LayoutBase)`.

### `make_layout(layouts)`

Concatenate a sequence of layouts into one. Each input layout becomes one
mode of the result. Tests:
[`test_make_layout.py::TestMakeLayout`](../test/test_make_layout.py).

```python
>>> make_layout([Layout(3, 1), Layout((5, 1), (7, 2)), Layout(2, 42)])
Layout((3, (5, 1), 2), (1, (7, 2), 42))
```

### `make_layout_like(layout)` / `make_ordered_layout(shape, order)`

Build a layout with the same shape/stride structure as an existing layout, or
with modes permuted according to `order`. See [`layout.py`](../pycute/layout.py).

### `Tiler` (type alias)

`Union[Integer, "Layout", tuple["Tiler", ...], list["Tiler"]]` — an
`HTuple(Integer | Layout)`. The general by-mode right-hand side accepted by
`composition`, `logical_divide`, and `logical_product`; `tiler_to_layout`
converts it to a single equivalent `Layout` (Whitepaper, §3.3.5).

### `tiler_to_layout(tiler, e=1)`

Convert a tiler (`int`, tuple of tilers, or `Layout`) to a `Layout`.
A `Shape` is interpreted as "tilers with stride 1 in each mode" via
`E(*seq)`. Tests:
[`test_make_layout.py::TestTilerToLayout`](../test/test_make_layout.py).

```python
>>> tiler_to_layout(3)
Layout(3, 1)
>>> tiler_to_layout((4, 5))
Layout((4, 5), (1@0, 1@1))             # = (E(0), E(1))
>>> tiler_to_layout((Layout(4, 2), Layout(5, 3)))
Layout((4, 5), (2@0, 3@1))             # = (2*E(0), 3*E(1))
```

### `recast(layout, scale)`

Rescale a layout to a different element size. `scale > 1` packs (shrinks
the shape, may divide the stride); `scale < 1` (`Fraction`) unpacks
(grows the shape, multiplies the stride). Test:
[`test_recast.py`](../test/test_recast.py).

```python
>>> recast(Layout(24, 1), 8)
Layout(3, 1)
>>> recast(Layout(24, 1), Fraction(1, 2))
Layout(48, 1)
>>> recast(Layout((4, 4), (4, 1)), 4)
Layout((4, 1), (1, 1))
```

---

## Module: `algebra`

Source: [`pycute/algebra.py`](../pycute/algebra.py)

Each function in this module is a thin dispatcher: if the argument has a
matching `_method`, call it; if `None`, return `None`; if it is an
integer or tuple, promote via `tiler_to_layout`. This is what allows
`coalesce(layout)`, `coalesce(tensor)`, and `coalesce((4, 8))` to all
work.

### `coalesce(A, profile=1)`

Simplify `A` as a function from `int` to `int`. By-mode coalesce when
`profile` is a tuple. Tests:
[`test_coalesce.py`](../test/test_coalesce.py).

> *Post-conditions:* `depth(R) ≤ 1` (or `≤ depth(profile)`),
> `R(i) == A(i)` for `i ∈ [0, |A|)`.

```python
>>> coalesce(Layout((2, (1, 6)), (1, (6, 2))))
Layout(12, 1)
>>> coalesce(Layout((2, (1, 6)), (1, (6, 2))), (1, 1))
Layout((2, 6), (1, 2))                # by-mode: each mode coalesced independently
```

### `coalesce_z(A, profile=1)`

Like `coalesce`, but preserves trailing size-1 modes so that *every*
integer (not just in-bounds integers) maps consistently. Tests:
[`test_coalesce_z.py`](../test/test_coalesce_z.py).

### `composition(A, B)`

Group composition `R = A ∘ B`. Tests:
[`test_composition.py`](../test/test_composition.py).

> *Post-conditions:* `compatible(B, R)`, `R(i) == A(B(i))` for every
> `i ∈ Z(B)`.

```python
>>> composition(Layout((6, 2), (8, 2)), Layout((4, 3), (3, 1)))
Layout(((2, 2), 3), ((24, 2), 8))
```

### `complement(A, extend=None)`

Smallest layout with weakly-congruent codomain to `A`, ordered, disjoint
image from `A`. Pass `extend=` (a shape) for an extended complement up to a
target size. Tests:
[`test_complement.py`](../test/test_complement.py).

> *Post-conditions:* `weakly_congruent(coprofile(A), R)`,
> `R(i-1) < R(i)`, `R(i) ≠ A(j)` for every `j ∈ Z(A)` and `i ≥ 1`.

```python
>>> complement(Layout(4, 2))
Layout((2, 1), (1, 8))               # minimal: just plugs the size-1 gap
>>> complement(Layout(4, 2), extend=20)
Layout((2, 3), (1, 8))
```

### `logical_divide(A, B)`

Split `A` into a tile of shape `B` (mode 0) and a layout of those tiles
(mode 1). Tests:
[`test_logical_divide.py`](../test/test_logical_divide.py).

> *Post-conditions:* `rank(R) == 2`, `compatible(B, R[0])`,
> `R(i, 0) == A(B(i))`, every offset of `A` appears in the image of `R`.

```python
>>> logical_divide(Layout(24, 1), Layout(4, 2))
Layout((4, (2, 3)), (2, (1, 8)))
```

### `zipped_divide(A, B)`

`logical_divide(A, tiler_to_layout(B))`. Convenience for
"interpret `B` as a tiler".

### `logical_product(A, B)`

Reproduce `A` as a tile across the layout `B`. Tests:
[`test_logical_product.py`](../test/test_logical_product.py).

> *Post-conditions:* `rank(R) == 2`, `R[0] == A`, `compatible(B, R[1])`.

```python
>>> logical_product(Layout((2, 2), (4, 1)), Layout(6, 1))
Layout(((2, 2), (2, 3)), ((4, 1), (2, 8)))
```

### `blocked_product(A, B)`

Rank-sensitive product: each mode of `A` combined with the corresponding
mode of `B`, tile-first. Tests:
[`test_blocked_raked.py::TestBlockedProduct`](../test/test_blocked_raked.py).

```python
>>> blocked_product(Layout((2, 5), (5, 1)), Layout((3, 4)))
... # 2x5 row-major tiles arranged in a 3x4 column-major grid
```

### `raked_product(A, B)`

Rank-sensitive product, grid-first ("cyclic distribution"). Tests:
[`test_blocked_raked.py::TestRakedProduct`](../test/test_blocked_raked.py).

### `right_inverse(A)`

The largest right-inverse of `A`. Tests:
[`test_inverse_right.py`](../test/test_inverse_right.py).

> *Post-condition:* `R(A(R(k))) == R(k)` for every `k`. When `A` is a
> bijection on its codomain, equivalently `A(R(k)) == k`.

```python
>>> right_inverse(Layout((4, 8), (1, 4)))
Layout(32, 1)
>>> right_inverse(Layout((4, 8), (1, 5)))
Layout(4, 1)
```

### `left_inverse(A)`

A left-inverse of `A`. Tests:
[`test_inverse_left.py`](../test/test_inverse_left.py).

> *Pre-condition:* `A`'s nonzero strides form an ordered chain — sorted as
> `d_0 < d_1 < ...` (sizes `s_k`), each `d_{k-1} | d_k` and
> `d_k >= d_{k-1} * s_{k-1}`. Otherwise a `ValueError` is raised; this covers
> both non-injective `A` and injective-but-non-divisible strides such as
> `Layout((2, 2), (2, 3))`.
>
> *Post-condition:* `A(R(A(k))) == A(k)` for every `k ∈ Z_|A|`. When
> `A` is injective, equivalently `R(A(k)) == k`.

```python
>>> left_inverse(Layout((4, 8), (1, 5)))
Layout((5, 8), (1, 4))
```

### `nullspace(A)`

The layout of coordinates that map to offset `0`. Tests:
[`test_nullspace.py`](../test/test_nullspace.py).

> *Post-condition:* `A(R(i)) == 0` for every `i ∈ Z(R)`.

```python
>>> nullspace(Layout((2, 4, 6), (1, 2, 0)))
Layout(6, 8)
```

### `layout_add(A, B)`

Coordinate-wise sum of two layouts of equal size:
`R(i) == A(i) + B(i)`. Tests:
[`test_layout_add.py`](../test/test_layout_add.py).

### `greatest_common_domain(A, B)`

Layout selecting the greatest common factorization shared by `shape(A)` and
`shape(B)` in leaf order. Tests:
[`test_greatest_common_domain.py`](../test/test_greatest_common_domain.py).

---

## Module: `swizzle`

Source: [`pycute/swizzle.py`](../pycute/swizzle.py)
Tests: [`test/test_swizzle.py`](../test/test_swizzle.py)

### `class F2(value)`

Stride scalar for binary-field strides: an element of
$F_2^m = (\mathbb{Z}_{2^m}, \mathrm{XOR}, \cdot)$, whose bits are the
coefficients of a polynomial over the two-element field. `+` is `^` (XOR)
and `*` is the *carry-less* (polynomial) product, in both its `F2 * F2` and
`F2 * int` forms — an integer operand acts through its bits, not its value.
The two products coincide wherever the schoolbook multiplication carries
nowhere, as it does for a power-of-two operand.

```python
>>> F2(0b1010) + F2(0b1100)
F6
>>> 3 * F2(0b1010)
F30
>>> F2(0b11) * 0b11
F5                       # 0b101 carry-lessly, where 3 * 3 == 9 in Z
```

`divmod(F2(a), b)` is Euclidean division in $GF(2)[x]$: the unique `(q, r)`
with `a == q * b + r` and `deg(r) < deg(b)`, both in `F2`. For a
power-of-two `b` it is exactly the bit split
`(F2(a >> k), F2(a & (2**k - 1)))`. This is what lets
[`idx2crd`](#idx2crdidx-shape) decompose an `F2` codomain value into the
natural coordinates of a shape, via `F2`'s `_idx2crd` hook; a shape whose
colexicographical prefix products disagree in $\mathbb{Z}$ and $F_2$ raises
`ValueError`. Tests:
[`test_swizzle.py::TestF2Divmod`](../test/test_swizzle.py),
[`TestF2Idx2Crd`](../test/test_swizzle.py).

### `class Swizzle(bits, base, shift)`

XOR-based offset transformation:
`offset → offset ^ ((offset & YYY_mask) >> shift)`, where
`YYY_mask = ((1 << bits) - 1) << (base + max(0, shift))`. A function on
integers, useful for inspecting the bank-conflict pattern produced by an
`F2`-stride layout. Tests:
[`test_swizzle.py::TestSwizzle`](../test/test_swizzle.py).

### `shiftr(a, s)` / `shiftl(a, s)`

Right shift / left shift, supporting negative `s` (which inverts the
direction). Used inside `Swizzle.__call__`.

---

## Module: `accessor`

Source: [`pycute/accessor.py`](../pycute/accessor.py)

### `class Accessor`

ABC for read-only random-access pointers. Defines `__add__` and
`__getitem__`.

### `class MutableAccessor`

ABC for read/write random-access pointers. Adds `__setitem__`.

### `class Ptr(source, dtype=None, owner=None)`

A typed pointer into memory owned elsewhere — the general-purpose accessor.
`source` is an integer address (then `dtype` is required) or an object that
can provide one (`array.array`, `numpy.ndarray`, a ctypes array,
`bytearray`, …), in which case `dtype` and the storage `owner` are inferred.
Offsetting yields another `Ptr` anchored to the same owner. `MutableAccessor`.

```python
>>> import array, ctypes
>>> data = array.array('d', [1.0, 2.0, 3.0, 4.0])
>>> T = Tensor(Ptr(data), Layout((2, 2), (2, 1)))   # dtype inferred
>>> T[1, 0] = 9.0
>>> data[2]
9.0
>>> Ptr(data.buffer_info()[0], ctypes.c_double)[0]  # raw address
1.0
```

### `class Array(size, dtype=ctypes.c_double)`

Heap-allocated contiguous storage: a `Ptr` that owns its allocation.

```python
>>> a = Array(16, dtype=ctypes.c_int)
>>> a[5] = 42; a[5]
42
```

### `class ImplicitAccessor(base)`

A "no-op" accessor: dereferencing returns `base + offset` rather than
loading from memory. Used by `print_tensor` and `identity_tensor`.

### `class TransformAccessor(accessor, transform)`

Wraps an accessor and applies a callable `transform` on every
dereference: `TransformAccessor(e, f)[i] = f(e[i])`. Offsetting
propagates to the inner accessor, leaving `transform` unchanged.

```python
>>> a = Array(4, dtype=ctypes.c_int)
>>> for i in range(4): a[i] = i
>>> t = TransformAccessor(a, lambda x: x * x)
>>> t[3]
9
```

---

## Module: `tensor`

Source: [`pycute/tensor.py`](../pycute/tensor.py)
Tests: [`test/test_tensor.py`](../test/test_tensor.py)

### `class Tensor(accessor, layout)`

A tensor: an `Accessor` paired with a `Layout`. Supports `__getitem__`,
`__setitem__`, `get(mode)`, equality, `coalesce`, `composition`,
`logical_divide`, and `zipped_divide`. Slicing returns a sub-tensor. Tests:
[`test_tensor.py::TestTensor`](../test/test_tensor.py).

```python
>>> T = make_tensor(Layout((4, 4), (4, 1)))
>>> T[1, 2] = 42.0
>>> T[1, 2]
42.0
```

### `is_tensor(x)`

True iff `isinstance(x, Tensor)`.

### `make_tensor(layout, dtype=ctypes.c_double)`

Allocate an `Array` of size `coshape(layout)` and bind it to `layout`. To bind
a layout to existing data, pass a `Ptr` to `Tensor` directly.

### `identity_tensor(shape)`

`Tensor(ImplicitAccessor(0), Layout(shape, make_basis_like(shape)))`.
The "every coordinate maps to itself" tensor — used for predication.

```python
>>> identity_tensor((3, 4))[1, 2]
(1, 2)
```

---

## Module: `util.print_tensor`

Source: [`pycute/util/print_tensor.py`](../pycute/util/print_tensor.py).
Imported as `from pycute.util import print_tensor`.

### `print_tensor(tensor, print_type=True)`

Print a layout or tensor of rank 1 through 4 as nested ASCII tables. When
called on a `Layout`, prints the offsets via an `ImplicitAccessor`. See
[Visualization](./07_visualization.md) for examples.

---

## Module: `util.print_table`

Source: [`pycute/util/print_table.py`](../pycute/util/print_table.py).
Imported as `from pycute.util import print_table`. Requires
[`tabulate`](https://pypi.org/project/tabulate/).

### `print_table(tensor, print_type=True)`

Render a rank-2 layout or tensor as a single bordered grid (one cell per
coordinate). Like `print_tensor`, it accepts a `Layout` (rendered through
an `ImplicitAccessor`) or a `Tensor`, and prints the `Shape:Stride` header
unless `print_type=False`. For non-rank-2 inputs, falls back to a plain
`print`.

---

## Module: `util.draw_svg`

Source: [`pycute/util/draw_svg.py`](../pycute/util/draw_svg.py). Requires
[`svgwrite`](https://pypi.org/project/svgwrite/).

### `draw_svg(tensor, filename="layout.svg", color=white)`

Save a rank-2 `Tensor` or `Layout` as an SVG table (with row/column index
labels); a rank-1 input is drawn as a single row. Like `print_tensor`, either
input is accepted: a `Layout` labels each cell with its offset, a `Tensor` with
the element stored there. `color` is a functor `color(idx) -> (r, g, b)`
(RGB-255 tuple) keyed on the cell offset; it defaults to `white`.

### `draw_svg_tv(layout, tile_mn=None, filename="tvlayout.svg", color=thread_color_8x)`

Save a rank-2 thread-value layout as a colored SVG table with `T`/`V`
annotations. The layout's codomain may be 2-D `(m, n)` coordinates, or a
linear offset — in which case a rank-2 `tile_mn` is required and the
offsets are folded into the tile via `composition(tile_mn, layout)`.
`color` is a functor `color(tid, vid) -> (r, g, b)` (RGB-255 tuple); it
defaults to `thread_color_8x` (coloring by `tid % 8`).

---

## Module: `util.draw_latex`

Source: [`pycute/util/draw_latex.py`](../pycute/util/draw_latex.py). Needs
no Python packages; PDF output requires a LaTeX install (`pdflatex`).

### `draw_latex(tensor, filename="layout.tex", compile_pdf=True, color=white)`

LaTeX/PDF analogue of `draw_svg`: write a standalone TikZ document for a
rank-2 `Tensor` or `Layout` (rank-1 becomes a single row) and (by default)
compile it to a cropped PDF with `pdflatex`.
Mirrors `cute::print_latex`. `color` is a functor `color(idx) -> (r, g, b)`
(same contract as `draw_svg`), defaulting to `white`.

### `draw_latex_tv(layout, tile_mn=None, filename="tvlayout.tex", compile_pdf=True, color=thread_color_8x)`

LaTeX/PDF analogue of `draw_svg_tv`: same thread-value layouts (2-D
`(m, n)` or linear-offset codomain via a rank-2 `tile_mn`) and the same
errors, rendered as TikZ/PDF with `T`/`V` annotations. `color` is a functor
`color(tid, vid) -> (r, g, b)` (same contract as `draw_svg_tv`), defaulting
to `thread_color_8x` (coloring by `tid % 8`).

---

## Module: `util.draw_colors`

Source: [`pycute/util/draw_colors.py`](../pycute/util/draw_colors.py). Pure
Python; no dependencies. A catalog of `color` functors shared by the SVG and
LaTeX drawers. Every functor returns an `(r, g, b)` tuple with each component
in `[0, 255]`. Offset functors take `color(idx)`; thread-value functors take
`color(tid, vid)`.

### `index_grey_8x(idx)`

Offset coloring: greyscale shade by `idx % 8`. Offset functors coerce `idx`
with `int()`, so `F2`-strided layouts color correctly.

### `bank_color_8x(idx)` / `bank_color_16x(idx)` / `bank_color_32x(idx)`

Offset coloring: color by shared-memory bank `idx % N` (N = 8, 16, 32) from a
shared 32-color light spectrum (the 8- and 16-entry palettes are evenly-spaced
subsamples). Equal-bank cells share a color — useful for spotting bank
conflicts; pick the modulus matching your bank count.

### `thread_color_8x(tid, vid)`

Thread-value coloring: color by `tid % 8` (`vid` ignored); matches
`cute::print_latex`. The default `color` of `draw_svg_tv` / `draw_latex_tv`.

### `value_color_8x(tid, vid)`

Thread-value coloring: color by value index `vid % 8` (`tid` ignored).

### `warp_color_8x(tid, vid)`

Thread-value coloring: color by warp `(tid // 32) % 8` (32 threads per warp).

### `white(*args)`

Constant white `(255, 255, 255)`; ignores its key, so it is valid for either
functor signature. The default `color` of `draw_svg` / `draw_latex`.

### `constant(rgb)`

Factory returning a functor that ignores its key and always yields `rgb`
(e.g. `constant((255, 225, 180))`). Generalizes `white`.

---

## Per-test cross reference

When you want to learn the post-condition of an operation, the unit test
is the authoritative source. Each PyCuTe test defines a
`postcondition_*` helper that asserts the formal contract.

| Test | What it checks |
|---|---|
| [`test_htuple.py`](../test/test_htuple.py) | `is_tuple`/`wrap`/`unwrap`/`front`/`back`, `product`, `product_each`, `inner_product`, `prefix_product`, `idx2crd`/`crd2idx` round-trip, `slice_`/`dice_`, `get`/`lift` round-trip, `select`/`take`, `transform_leaf`, `flatten`/`unflatten`/`repeat_like`, mode-indexed `shape`/`size`/`rank`/`depth` |
| [`test_atuple.py`](../test/test_atuple.py) | `ArithTuple` arithmetic, `ScaledBasis` semantics, colex ordering, coordinate-strided layouts |
| [`test_typing.py`](../test/test_typing.py) | `Integer` / `is_int`, `sympy` symbol compatibility, layout algebra with symbolic shapes |
| [`test_layout_add.py`](../test/test_layout_add.py) | `layout_add` post-conditions |
| [`test_greatest_common_domain.py`](../test/test_greatest_common_domain.py) | `greatest_common_domain` post-conditions |
| [`test_compatibility.py`](../test/test_compatibility.py) | `congruent`, `weakly_congruent`, `compatible` (with the Whitepaper's §2.2.1 examples) |
| [`test_layout.py`](../test/test_layout.py) | `Layout` construction, `[]` indexing, three-coordinate-form equivalence, slicing via `_offset_and_slice`, structural equality, `coshape`/`coprofile` |
| [`test_make_layout.py`](../test/test_make_layout.py) | `make_layout` concatenation, `tiler_to_layout` for int/Shape/Layout/tuple, the `composition(A, T) == composition(A, tiler_to_layout(T))` invariant, `zipped_divide` ↔ `logical_divide` |
| [`test_coalesce.py`](../test/test_coalesce.py) | `coalesce` post-conditions: `depth ≤ 1`, `size`, `R(i) == A(i)` |
| [`test_coalesce_z.py`](../test/test_coalesce_z.py) | `coalesce_z` post-conditions: same as `coalesce` but on the extended domain |
| [`test_composition.py`](../test/test_composition.py) | `composition` post-conditions: `compatible(B, R)`, `R(i) == A(B(i))` |
| [`test_complement.py`](../test/test_complement.py) | `complement` post-conditions: weakly congruent codomain, ordered, disjoint; strong inverse condition |
| [`test_logical_divide.py`](../test/test_logical_divide.py) | `logical_divide` post-conditions and the SM70 MMA associativity test |
| [`test_logical_product.py`](../test/test_logical_product.py) | `logical_product` post-conditions: `rank == 2`, `R[0] == A`, `compatible(B, R[1])` |
| [`test_blocked_raked.py`](../test/test_blocked_raked.py) | `blocked_product` and `raked_product`: rank-sensitive products, post-conditions, blocked-vs-raked image equivalence |
| [`test_inverse_right.py`](../test/test_inverse_right.py) | `right_inverse` post-condition |
| [`test_inverse_left.py`](../test/test_inverse_left.py) | `left_inverse` post-condition + cotiling application |
| [`test_nullspace.py`](../test/test_nullspace.py) | `nullspace` post-condition |
| [`test_recast.py`](../test/test_recast.py) | `recast` for integer and `Fraction` scales, scalar/vector/matrix/nested layouts |
| [`test_swizzle.py`](../test/test_swizzle.py) | `F2` arithmetic (XOR-as-addition, carry-less multiplication, self-inverse), `F2` Euclidean division and the `idx2crd` decomposition of an `F2` value, `Swizzle` (XOR pattern, involution, constructor validation), F2-stride layouts evaluated at integer and `F2` coordinates |
| [`test_tensor.py`](../test/test_tensor.py) | `Tensor` construction, three-coordinate-form indexing, `__getitem__`/`__setitem__`, slicing returning sub-tensors, tensor algebra pass-through, `Accessor`, `make_tensor`, `identity_tensor` (predication) |

Run them all with:

```sh
pytest --log-cli-level DEBUG
```

## Copyright

Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
