# PyCuTe API Reference

Every name in `pycute.__all__`, with its signature, its documentation, and
links to the source and to the tests that exercise it.

> **Generated file — do not edit.** Every word below comes from a docstring
> in `pycute/`. Edit the docstring, then run
> `python scripts/gen_api_reference.py`.

The reference is organized by module:

* [`htuple`](#module-htuple) — `is_tuple`, `profile`, `congruent`, `weakly_congruent`, `wrap`, `unwrap`, `ModeOpDecorator`, `get`, `lift`, `replace`, `select`, `take`, `transform_apply_leaf`, `transform_leaf`, `leaves`, `zip_leaves`, `fold_leaf`, `flatten`, `unflatten`, `repeat_like`, `product`, `product_each`, `slice_`, `dice_`
* [`typedefs`](#module-typedefs) — `Integer`, `register_integer_type`, `is_int`, `is_static`, `divmod`, `StrideScalar`, `is_stride_scalar`, `HTuple`, `Profile`, `IntTuple`, `Shape`, `Coord`, `Stride`
* [`stride`](#module-stride) — `stride`, `inner_product`, `prefix_product`, `coshape`, `coprofile`
* [`shape`](#module-shape) — `shape`, `size`, `rank`, `depth`, `compatible`, `common_refinement`, `common_coarsening`, `idx2crd`, `crd2idx`, `coordinates`
* [`atuple`](#module-atuple) — `ArithTuple`, `ScaledBasis`, `E`, `V`, `basis_repr`, `is_basis`, `make_basis_like`, `proj`, `unit`, `as_tuple`
* [`layout`](#module-layout) — `LayoutBase`, `is_layout`, `Layout`, `make_layout`, `make_layout_like`, `make_ordered_layout`, `Tiler`, `tiler_to_layout`, `recast`
* [`algebra`](#module-algebra) — `coalesce_z`, `coalesce`, `composition`, `right_inverse`, `left_inverse`, `complement`, `logical_product`, `logical_divide`, `zipped_divide`, `blocked_product`, `raked_product`, `nullspace`, `layout_add`, `greatest_common_domain`
* [`swizzle`](#module-swizzle) — `F2`, `shiftr`, `shiftl`, `Swizzle`
* [`accessor`](#module-accessor) — `Accessor`, `MutableAccessor`, `Ptr`, `Array`, `ImplicitAccessor`, `TransformAccessor`
* [`tensor`](#module-tensor) — `Tensor`, `is_tensor`, `identity_tensor`, `make_tensor`
* [`util.print_tensor`](#module-utilprint_tensor) — `print_tensor`
* [`util.print_table`](#module-utilprint_table) — `print_table`
* [`util.draw_svg`](#module-utildraw_svg) — `draw_svg`, `draw_svg_tv`
* [`util.draw_latex`](#module-utildraw_latex) — `draw_latex`, `draw_latex_tv`
* [`util.draw_colors`](#module-utildraw_colors) — `index_grey_8x`, `bank_color_8x`, `bank_color_16x`, `bank_color_32x`, `thread_color_8x`, `value_color_8x`, `warp_color_8x`, `constant`, `white`

Each entry's *Pre-conditions* and *Post-conditions* are the formal contract
its unit test asserts, and every *Examples* block is evaluated by
`test/test_docstring_examples.py`, so both are true of the code as it is.

---

## Module: `htuple`

Source: [`pycute/htuple.py`](../pycute/htuple.py)
Tests: [`test_compatibility.py`](../test/test_compatibility.py), [`test_htuple.py`](../test/test_htuple.py)

Functions for manipulating Hierarchical Tuples

An *HTuple* is the container at the base of CuTe: a leaf, or a tuple/list of
HTuples. `Shape`, `Stride`, `Coord` and `Tiler` are all HTuples that differ only
in what they admit at a leaf, so the combinators here serve all of them.

Two ideas recur. A *profile* is an HTuple read for its tree alone, with the
leaves ignored; `congruent` and `weakly_congruent` compare profiles. A *mode* is
a path of indices into that tree, and every function taking one is
subscriptable, so `get[0, 2](x)` is `get(x, mode=(0, 2))`.

### `is_tuple(x)`

Test whether `x` is an HTuple internal node rather than a leaf.

*Examples:*

```python
is_tuple((1, 2))          == True
is_tuple([1, 2])          == True
is_tuple(7)               == False
is_tuple(Layout((2, 3)))  == False
```

### `profile(obj)`

Get an object's *profile*: its HTuple tree with the leaves left as they are.

Congruence reads a tuple's tree and ignores whatever sits at its leaves, so
anything that is not a `Layout` or `Tensor` already *is* its own profile.

*Notable consequences:*

* `profile(obj) == shape(obj)` for a `Layout` or a `Tensor`.
* `profile(obj) is obj` for every other `HTuple`: it is already a profile.
* `profile` is idempotent and total: it accepts any object and rejects none.

*Examples:*

```python
profile((2, (3, 4)))            == (2, (3, 4))
profile(42)                     == 42
profile(Layout((2, (3, 4))))    == (2, (3, 4))
profile((F2(1), F2(2)))         == (F2(1), F2(2))    # a Stride has no shape
profile((Layout(2), Layout(3))) == (Layout(2), Layout(3))    # a Tiler's leaves
```

### `congruent(a, b)`

Test whether `a` and `b` have the same hierarchical profile (Whitepaper, §2.1).

*Congruence* is an equivalence relation on `HTuple`s: `a ~ b` iff `a` and `b`
have matching tuple/leaf structure at every level, whatever their leaves hold.

*Examples:*

```python
congruent((4, 8), (5, 7))                 == True
congruent(31, 42)                         == True
congruent((4, 8), (4, (2, 4)))            == False    # different profile
congruent(31, (4, 8))                     == False    # leaf vs tuple
congruent((1, 1, 1), (1, 1))              == False    # different rank
congruent((4, 8), (E(0), E(1)))           == True     # a coordinate stride
congruent((4, 8), (F2(1), F2(8)))         == True     # an F2 stride
```

### `weakly_congruent(a, b)`

Test whether `a` *coarsens the profile* of `b` (Whitepaper, §2.1).

*Weak congruence* is a partial order on `HTuple`s: `a ≲ b` iff `a`'s structure
can be obtained from `b`'s by collapsing zero or more sub-trees into leaves.

*Notable consequences:*

* A leaf is weakly congruent to any profile (it coarsens everything).
* A tuple is never weakly congruent to a leaf.
* `a ~ b`  implies  `a ≲ b`  (congruence implies weak congruence).

*Examples:*

```python
weakly_congruent(30, (3, 4))              == True     # a leaf coarsens any shape
weakly_congruent(30, ((3, 4), 5))         == True
weakly_congruent((3, 4), 30)              == False    # tuple does not coarsen leaf
weakly_congruent((3, 4), (5, (6, 7)))     == True     # rank-2 vs rank-2, recurse
weakly_congruent((3, (4, 5)), (5, 6))     == False    # (4,5) does not coarsen 6
weakly_congruent((1, 2, 3), (1, 2))       == False    # top-level rank mismatch
weakly_congruent(E(0), 8)                 == True     # a stride scalar leaf
weakly_congruent(E(0, 0), (8, 8))         == True     # ... coarsens a shape too
```

### `wrap(x)`

Wrap `x` in a 1-tuple unless it is already a tuple.

*Examples:*

```python
wrap(7)     == (7,)
wrap((7,))  == (7,)
wrap(())    == ()
```

### `unwrap(x)`

Strip enclosing 1-tuples from `x`, recursively.

*Post-conditions:*

```
unwrap(wrap(x)) == x   for a non-tuple x; a 1-tuple is unwrapped, not restored
```

*Examples:*

```python
unwrap((7,))          == 7
unwrap(((((42,))),))  == 42
unwrap((1, 2))        == (1, 2)
unwrap(wrap((3,)))    == 3            # not (3,): wrap had nothing to add
```

### `ModeOpDecorator(func)`

Expose the keyword-only `mode` parameter of `func` as a subscript.

Subscripted modes are prepended to the `mode` given at the call site, and
every other argument passes through untouched:

```
op(A)                <==>  op(A, mode=())        # no mode filtering
op[0](A)             <==>  op(A, mode=(0,))      # mode 0 of A
op[0,1](A)           <==>  op(A, mode=(0,1))     # mode (0,1) of A
op[0][1](A)          <==>  op(A, mode=(0,1))     # subscripts accumulate
op[0](A, B)          <==>  op(A, B, mode=(0,))   # any number of arguments
op[0](A, B, mode=1)  <==>  op(A, B, mode=(0,1))
```

`mode` is keyword-only, so a mode is never mistaken for an argument of `op`.

*Examples:*

```python
shape[1](Layout((3, (2, 4))))     == shape(Layout((3, (2, 4))), mode=(1,))
shape[1][0](Layout((3, (2, 4))))  == 2
size.__name__                     == 'size'
```

### `get(obj, *, mode=())`

Get the `mode[0]`th mode, then the `mode[1]`th mode, etc of `obj`.

*Post-conditions:*

```
get[mode](lift[mode](x)) == x
get(obj) is obj
```

*Examples:*

```python
get[0, 2, 3](((0, 0, (0, 0, 0, 42)),))        == 42
get(((0, 0, (0, 0, 0, 42)),), mode=(0, 2, 3)) == 42
get[1](Layout((3, (2, 4)), (2, (1, 6))))      == Layout((2, 4), (1, 6))
get[1, 0]((1, (2, 3)))                        == 2
```

### `lift(obj, *, pad=0, make=tuple, mode=())`

Create an object with `obj` as the `mode`-th element.

*Args:*

obj: The object to place at `mode`
pad: The value filling the modes that `mode` does not name
make: Builds each mode created, from the sequence of its elements
mode: Sequence of indices to apply in order

*Post-conditions:*

```
get[mode](lift[mode](x)) == x
lift(x) is x
```

*Examples:*

```python
lift[0, 2, 3](42)                                         == ((0, 0, (0, 0, 0, 42)),)
lift[1](42, pad=None)                                     == (None, 42)
lift[1](Layout(4, 2), pad=Layout(1, 0), make=make_layout)  == Layout((1, 4), (0, 2))
```

### `replace(obj, x, *, mode=())`

Create a copy of `obj` with its `mode`-th element replaced by `x`.

*Pre-conditions:*

```
`mode` names an existing element of `obj`; otherwise a ValueError is raised
```

*Post-conditions:*

```
get[mode](replace[mode](obj, x)) == x
replace(obj, x) == x
```

*Examples:*

```python
replace[1]((1, 2, 3), 42)                  == (1, 42, 3)
replace[0, 2](((1, 2, 3), 4), 42)          == ((1, 2, 42), 4)
replace[1](repeat_like(None, (3, 4)), 42)  == (None, 42)
replace[3]((1, 2, 3), 42)                  -> ValueError
```

### `select(obj, *, mode=())`

Select the modes of `obj` named by `mode`, in the order given, as a tuple.

*Post-conditions:*

```
len(result) == len(mode)
result[i] == get[mode[i]](obj)
```

*Examples:*

```python
A = Layout((2, 3, 5, 7), (1, 2, 6, 30))
select[1, 3](A)               == (Layout(3, 2), Layout(7, 30))
select[3, 1](A)               == (Layout(7, 30), Layout(3, 2))
select[2](A)                  == (Layout(5, 6),)
make_layout(select[1, 3](A))  == Layout((3, 7), (2, 30))
select[0, 1]((2, (3, 4), 5))  == (2, (3, 4))
```

### `take(obj, *, mode=())`

Select the modes of `obj` in the half-open range `[mode[0], mode[1])`.

*Pre-conditions:*

```
len(mode) == 2 and mode[0] <= mode[1]; otherwise a ValueError is raised
```

*Post-conditions:*

```
take[i, j](obj) == select[tuple(range(i, j))](obj)
```

*Examples:*

```python
A = Layout((2, 3, 5, 7), (1, 2, 6, 30))
take[1, 4](A)     == (Layout(3, 2), Layout(5, 6), Layout(7, 30))
take[1, 2](A)     == (Layout(3, 2),)
take[2, 2](A)     == ()
take[2, 1](A)     -> ValueError
take[1, 2, 3](A)  -> ValueError
```

### `transform_apply_leaf(make, fn, htuple, *tuples)`

Rebuild `htuple` with `fn` applied at every leaf and `make` at every node:
`transform_apply_leaf(make, fn, t...) == make(fn(t)...)`.

*Args:*

make: Builds one node of the result from an iterable of its children
fn: Maps the corresponding leaves of every input to one leaf of the result
htuple: The HTuple whose tree the result follows
*tuples: Further HTuples walked alongside `htuple`

*Pre-conditions:*

```
weakly_congruent(htuple, t) for every t in tuples
```

*Examples:*

```python
transform_apply_leaf(tuple, lambda x: x * 2, (1, (2, 3)))  == (2, (4, 6))
transform_apply_leaf(sum, lambda x: x, (1, (2, 3)))        == 6
transform_apply_leaf(make_layout, Layout, (2, 3), (1, 4))  == Layout((2, 3), (1, 4))
```

### `transform_leaf(fn, *tuples)`

Apply `fn` at every leaf, rebuilding the tree with plain tuples.

`transform_apply_leaf` with `make=tuple`, which is the common case.

*Post-conditions:*

```
congruent(result, tuples[0])
```

*Examples:*

```python
transform_leaf(lambda x: x + 1, (1, (2, 3)))       == (2, (3, 4))
transform_leaf(lambda x, y: x * y, (2, 3), (5, 7)) == (10, 21)
```

### `leaves(htuple)`

Generate the leaves of `htuple`, left to right.

*Examples:*

```python
tuple(leaves(((2, 3), 4)))  == (2, 3, 4)
tuple(leaves(42))           == (42,)
tuple(leaves(()))           == ()
```

### `zip_leaves(htuple, *tuples)`

Generate the corresponding leaves of every input, as tuples.

`htuple` drives the walk, so where it has a leaf the matching sub-trees of
`*tuples` are yielded whole rather than descended into.

*Pre-conditions:*

```
weakly_congruent(htuple, t) for every t in tuples
```

*Examples:*

```python
list(zip_leaves((1, 2), (3, 4)))            == [(1, 3), (2, 4)]
list(zip_leaves((1, (2, 3)), (4, (5, 6))))  == [(1, 4), (2, 5), (3, 6)]
list(zip_leaves((1, 2), (3, 4), (5, 6)))    == [(1, 3, 5), (2, 4, 6)]
list(zip_leaves(1, (2, 3)))                 == [(1, (2, 3))]
```

### `fold_leaf(fn, init, *tuples)`

Left-fold `fn` over the corresponding leaves of `*tuples`, starting from `init`.

*Pre-conditions:*

```
weakly_congruent(tuples[0], t) for every t in tuples
```

*Examples:*

```python
fold_leaf(lambda acc, x: acc + x, 0, (1, (2, 3)))           == 6
fold_leaf(lambda acc, x, y: acc + x * y, 0, (2, 3), (5, 7)) == 31
```

### `flatten(htuple, make=tuple)`

Collect the leaves of `htuple` into one flat `make`, discarding the tree.

*Post-conditions:*

```
depth(result) <= 1
unflatten(iter(flatten(htuple)), htuple) == htuple
```

*Examples:*

```python
flatten(((2, 3), 4))                          == (2, 3, 4)
flatten(42)                                   == (42,)
flatten((Layout(2), Layout(3)), make_layout)  == Layout((2, 3), (1, 1))
```

### `unflatten(values, profile, make=tuple)`

Rebuild `profile`'s tree from a flat iterator of leaves; inverse of `flatten`.

*Pre-conditions:*

```
`values` yields at least as many items as `profile` has leaves
```

*Post-conditions:*

```
congruent(result, profile)
unflatten(iter(flatten(htuple)), htuple) == htuple
```

*Examples:*

```python
unflatten(iter([2, 3, 4]), ((0, 0), 0))  == ((2, 3), 4)
unflatten(iter([1, 2, 3]), 0)            == 1
unflatten(iter([2, 3]), (0, 0), list)    == [2, 3]
```

### `repeat_like(x, profile)`

Replicate `x` at every leaf of `profile`'s tree.

*Post-conditions:*

```
congruent(result, profile)
```

*Examples:*

```python
repeat_like(0, ((1, (2, 3)), 4))  == ((0, (0, 0)), 0)
repeat_like(None, (3, 4))         == (None, None)
repeat_like(0, 42)                == 0
```

### `product(s)`

Multiply every leaf of `s` together.

*Examples:*

```python
product(((2, 3), 4))  == 24
product(42)           == 42
product(())           == 1
```

### `product_each(s)`

The `product` of each top-level mode of `s`, as a flat tuple.

*Post-conditions:*

```
len(result) == len(s)
product(result) == product(s)
```

*Examples:*

```python
product_each(((2, 3), 4, (5, 6)))  == (6, 4, 30)
product_each((2, 3))               == (2, 3)
product_each(())                   == ()
```

### `slice_(htuple, B, make=tuple)`

Collect the leaves of `B` whose counterpart in `htuple` is `None`.

*Pre-conditions:*

```
weakly_congruent(htuple, B)
```

*Examples:*

```python
slice_((None, 1), ((2, 3), (5, 7, 9)))                 == ((2, 3),)
slice_((None, None), (2, 3))                           == (2, 3)
slice_((1, 2), (5, 7))                                 == ()
slice_((None, 1), (Layout(4), Layout(8)), make_layout) == Layout((4,), (1,))
```

### `dice_(htuple, B, make=tuple)`

Collect the leaves of `B` whose counterpart in `htuple` is not `None`.

*Pre-conditions:*

```
weakly_congruent(htuple, B)
```

*Examples:*

```python
dice_((None, 1), ((2, 3), (5, 7, 9)))  == ((5, 7, 9),)
dice_((None, None), (2, 3))            == ()
dice_((1, 2), (5, 7))                  == (5, 7)
```

---

## Module: `typedefs`

Source: [`pycute/typedefs.py`](../pycute/typedefs.py)
Tests: [`test_typing.py`](../test/test_typing.py)

Type definitions for PyCuTe: the scalar ABCs, their predicates, and the
`HTuple` type-alias vocabulary.

The single home of PyCuTe's type language (Whitepaper, §2). Three
non-overlapping predicates discriminate a leaf, each answering one question:

```
`is_tuple(x)`          (in `htuple.py`) the structural boundary between an
                       HTuple node and an HTuple leaf. Purely syntactic, so
                       `int`, `ArithTuple`, `F2` and `Layout` are all leaves.
`is_int(x)`            "acts as an ordinary integer scalar": shapes,
                       coordinates, sizes, divisors.
`is_stride_scalar(x)`  "may sit at a Layout's stride leaf", an algebraic
                       contract of `+` and `*` by an integer.
```

The latter two are decided by ABC registration rather than by duck-typing, so
that membership is a deliberate semantic claim: `F2` defines `__add__` and
`__mul__` yet is not an `Integer`, because its `+` is XOR.

The `Tiler` alias, whose leaf is a `Layout`, lives in `layout.py` beside that
class; every other alias is here.

### `class Integer`

Abstract base class for *integer-shaped* scalar types in PyCuTe.

Membership is by **registration**, not duck-typing, so that it states a
semantic property rather than detecting it syntactically -- a type
advertising `__add__` and `__mul__` need not add like an integer.

* `int`, and any class inheriting from it, is recognized automatically.
* `bool` and `float` are excluded, even though `bool` subclasses `int`.
* Every other type must be registered, by `register_integer_type` or
  `Integer.register`. `numpy.integer` and `sympy.Expr` are registered on
  import, when importable.

*Examples:*

```python
isinstance(7, Integer)      == True
isinstance(True, Integer)   == False
isinstance(1.0, Integer)    == False
isinstance(F2(1), Integer)  == False    # its + is XOR
```

### `register_integer_type(*types)`

Declare one or more types as integer-shaped for PyCuTe purposes.

After registration `is_int` is true of their instances, and PyCuTe's shape,
coordinate and divisor code treats them as ordinary integers. Idempotent:
registering an already-registered type is a no-op.

```
import pycute
pycute.register_integer_type(mylib.MyIntegerType)
```

### `is_int(x)`

True iff `x` is an instance of an `Integer`-registered type.

### `is_static(x)`

True iff `x` is an `Integer` whose value is known at "compile time".

Grounded in the standard `int()` protocol rather than in any
library-specific attribute: `x` is static iff it is an integer equal to its
own `int()` coercion. A `sympy.Symbol`, or any expression carrying a free
symbol, is therefore dynamic, because its `int()` raises.

*Examples:*

```python
is_static(7)        == True
is_static(F2(3))    == True
is_static(1.0)      == False
```

### `divmod(a, b)`

Quotient and remainder `(a // b, a % b)`, with an overridable fast path.

Dispatches through the built-in `divmod` protocol first, so a type may supply
a fused `__divmod__` / `__rdivmod__`, and otherwise falls back to `//` and `%`.

*Examples:*

```python
divmod(7, 3)             == (2, 1)
divmod(F2(0b1011), 0b11) == (F2(0b110), F2(0b1))   # F2 supplies its own
```

### `class StrideScalar`

Abstract base class for the leaf types a `Layout`'s stride may hold.

The contract is algebraic -- `__add__`, `__radd__`, `__mul__`, `__rmul__`,
making an integer-semimodule (Whitepaper, §2.3.1) -- so `int` and every
`Integer` qualify and are registered below, while `ArithTuple` and `F2`
subclass it.

*Examples:*

```python
is_stride_scalar(7)      == True
is_stride_scalar(E(0))   == True
is_stride_scalar(F2(1))  == True
is_stride_scalar(1.0)    == False
```

### `is_stride_scalar(value)`

True iff `value` is an instance of a `StrideScalar` type.

### `HTuple`

```python
HTuple: TypeAlias = Union[Any, tuple['HTuple', ...], list['HTuple']]
```

An HTuple with unconstrained leaves; used for generic, profile-shaped args.

### `Profile`

```python
Profile: TypeAlias = HTuple
```

A congruence *profile*: an HTuple whose leaf values are irrelevant -- only its tuple/leaf tree structure matters (see `congruent`).

### `IntTuple`

```python
IntTuple: TypeAlias = Union[Integer, tuple['IntTuple', ...], list['IntTuple']]
```

An HTuple(Integer): the shared carrier of shapes and coordinates.

### `Shape`

```python
Shape: TypeAlias = IntTuple
```

A shape: an HTuple of (positive) integers describing per-mode extents (Z+).

### `Coord`

```python
Coord: TypeAlias = Union[Integer, None, tuple['Coord', ...], list['Coord']]
```

A coordinate into a shape: an HTuple of integers -- natural / integral / flat / admissible -- optionally carrying `None` slice-markers at any level.

### `Stride`

```python
Stride: TypeAlias = Union[StrideScalar, tuple['Stride', ...], list['Stride']]
```

An HTuple(StrideScalar): the stride half of a Layout, congruent with its Shape.

---

## Module: `stride`

Source: [`pycute/stride.py`](../pycute/stride.py)
Tests: [`test_htuple.py`](../test/test_htuple.py)

Functions for CuTe Strides

A `Stride` is an HTuple of stride scalars, congruent with a layout's `Shape`,
giving the step each mode takes through the codomain (Whitepaper, §2.3). A leaf
need only be an integer-semimodule element -- an `int`, an `ArithTuple`, an `F2`
-- so the same machinery addresses linear memory, multidimensional coordinates
and swizzled offsets alike.

This module holds the stride-side operations: reading a stride (`stride`),
evaluating one against a coordinate (`inner_product`), building a compact one
from a shape (`prefix_product`), and describing the codomain a stride reaches
(`coshape`, `coprofile`).

### `stride(obj, *, mode=())`

Get an object's stride.

*Examples:*

```python
stride(Layout((4, 8), (1, 4)))          == (1, 4)
stride((1, (4, 8)))                     == (1, (4, 8))
stride[1](Layout((3, (2, 4))))          == (3, 6)
stride(Layout((4, 8), (F2(1), F2(8))))  == (F2(1), F2(8))
```

### `inner_product(a, b)`

Sum of the leaf-wise products of two congruent HTuples: `sum(x*y)`.

*Pre-conditions:*

```
congruent(a, b)
```

*Examples:*

```python
inner_product((1, 0, 1),    (1, 3, 6))       == 7
inner_product((2, 3),       (1, 4))          == 14
inner_product((1, (2, 3)),  (1, (10, 100)))  == 321
```

### `prefix_product(a, init=1)`

Exclusive prefix product of the leaves of `a`, congruent with `a`.

`init` seeds the running product and may be:
  -- a stride scalar (e.g. `int`; the default `1`), or
  -- a tuple of stride scalars weakly congruent with `a`; 
     each mode is prefix-producted independently.

*Pre-conditions:*

```
weakly_congruent(init, a)
```

*Examples:*

```python
prefix_product((3, 2, 4))           == (1, 3, 6)
prefix_product((3, (2, 4)))         == (1, (3, 6))
prefix_product((4, 8), 2)           == (2, 8)               # base 2
prefix_product(((2, 3), (4, 5)), (1, 100)) == ((1, 2), (100, 400))   # per-mode base
```

### `coshape(obj, *, mode=())`

Shape of the codomain: an extent large enough to hold every value `obj` produces.

Each mode contributes its maximal offset `(s-1) * d`, and where the codomain's
addition is monotone -- `Z` and `Z^S` -- those contributions add. A codomain
whose addition is not monotone supplies its own bound instead; `F2`, whose `+`
is XOR, is bounded by bit-span rather than by sum.

*Examples:*

```python
coshape(Layout((4, 8), (1, 4)))        == 32
coshape(Layout((4, 8), (E(0), E(1))))  == (4, 8)
coshape(Layout(4, F2(3)))              == 8
```

### `coprofile(obj, *, mode=())`

Profile of the codomain: an HTuple congruent to `coshape(obj)` whose leaf values
carry no meaning.

Read straight off the strides, so unlike `coshape` it stays defined for
codomains whose extents cannot be bounded.

*Post-conditions:*

```
congruent(coprofile(obj), coshape(obj))   wherever coshape is defined
```

*Examples:*

```python
congruent(coprofile(Layout((4, 8), (1, 4))), 0)            == True
congruent(coprofile(Layout((4, 8), (E(0), E(1)))), (0, 0)) == True
```

---

## Module: `shape`

Source: [`pycute/shape.py`](../pycute/shape.py)
Tests: [`test_compatibility.py`](../test/test_compatibility.py)

Functions for CuTe Shapes

A `Shape` is an `IntTuple` of positive extents describing a layout's domain
(Whitepaper, §2.2). Its leaves and its tree together fix a coordinate space,
and this module holds the operations on that space: reading its structure
(`shape`, `size`, `rank`, `depth`), the *compatibility* partial order relating
one shape's coordinates to another's (`compatible`, `common_refinement`,
`common_coarsening`), and the maps between a coordinate's forms (`idx2crd`,
`crd2idx`, `coordinates`).

### `shape(obj, *, mode=())`

Get an object's shape.

*Examples:*

```python
shape(Layout((4, 8), (1, 4)))     == (4, 8)
shape((3, (2, 4)))                == (3, (2, 4))
shape(42)                         == 42
shape[1](Layout((3, (2, 4))))     == (2, 4)
shape[1, 0](Layout((3, (2, 4))))  == 2
```

### `size(obj, *, mode=())`

Get an object's size: the number of integral coordinates in its domain.

*Post-conditions:*

```
size(obj) == product(shape(obj))
```

*Examples:*

```python
size(Layout((4, 8), (1, 4)))   == 32
size((3, (2, 4)))              == 24
size(42)                       == 42
size[1](Layout((3, (2, 4))))   == 8
size(())                       == 1
```

### `rank(obj, *, mode=())`

Get an object's rank: the number of top-level modes of its shape.

*Examples:*

```python
rank(Layout((4, 8), (1, 4)))  == 2
rank((3, (2, 4), 5))          == 3
rank(42)                      == 1
rank(())                      == 0
rank[1](Layout((3, (2, 4))))  == 2
```

### `depth(obj, *, mode=())`

Get an object's depth: how deeply its shape nests.

*Examples:*

```python
depth(42)                        == 0
depth((3, 4))                    == 1
depth((3, (2, 4)))               == 2
depth(Layout((3, (2, (4, 5)))))  == 3
depth[1](Layout((3, (2, 4))))    == 1
```

### `compatible(a, b)`

Test whether `a` *coarsens* `b`.

*Compatibility*, `a ≼ b`, is a partial order on shapes: weak congruence that
additionally requires sizes to agree, so every coordinate of `a` is also a
coordinate of `b`, i.e. `Z(a) ⊆ Z(b)`. We say `a` *coarsens* `b`, and `b`
*refines* `a`.

Accepts any object that has a CuTe shape (e.g. `Layout`, `Tensor`).

*Notable consequences:*

* `a ≼ b`  implies  `a ≲ b`  (compatibility implies weak congruence).
* The least element below any shape `b` is the integer `size(b)`.

*Examples:*

```python
compatible(30, (2, 15))                   == True     # 30 ≼ (2, 15)
compatible((2, 15), (2, (3, 5)))          == True
compatible(30, (2, (3, 5)))               == True     # transitivity
compatible(24, ((2, 2), (3, 2)))          == True
compatible(24, 32)                        == False    # size mismatch
compatible((4, 6), ((2, 3), 8))           == False    # mode 0: 4 != 2*3
compatible((2, (3, 5)), ((3, 2), 5))      == False    # same size, but incompatible
compatible(24, (24,))                     == True     # int ≼ (int,)
compatible((24,), 24)                     == False    # but not the reverse
```

### `common_refinement(a, b)`

Find the minimal shape `c` that *refines* both `a` and `b` (Whitepaper, §2.2.1).

Equivalently, `c` is the *join* (least upper bound) of `a` and `b` in the
compatibility partial order on shapes:

```
a ≼ c,  b ≼ c,  and c is minimal under ≼.
```

Raises `ValueError` if no such shape exists.

*Notable consequences:*

* Symmetric: `common_refinement(a, b) == common_refinement(b, a)`.
* Reflexive: `common_refinement(a, a) == a`.
* `common_refinement` exists iff `a` and `b` share at least one common
  refinement, which requires `size(a) == size(b)` and compatible profiles.
  Accepts any object that has a CuTe shape (e.g. `Layout`, `Tensor`) via `shape(...)`.

*Examples:*

```python
common_refinement(30, (2, 15))              == (2, 15)
common_refinement((2, 15), (2, (3, 5)))     == (2, (3, 5))
common_refinement(10, (10,))                == (10,)
common_refinement(((2, 3), 20), (6, (4, 5))) == ((2, 3), (4, 5))
common_refinement((6, 5), (2, 15))          -> ValueError    # 6 != 2 at mode 0
common_refinement((2, (3, 5)), ((3, 2), 5)) -> ValueError    # same size, but incompatible
```

### `common_coarsening(a, b)`

Find the maximal shape `c` that *coarsens* both `a` and `b` (Whitepaper, §2.2.1).

Equivalently, `c` is the *meet* (greatest lower bound) of `a` and `b` in the
compatibility partial order on shapes:

```
c ≼ a,  c ≼ b,  and c is maximal under ≼.
```

Raises `ValueError` if no such shape exists.

*Notable consequences:*

* Symmetric: `common_coarsening(a, b) == common_coarsening(b, a)`.
* Reflexive: `common_coarsening(a, a) == a`.
* If `size(a) == size(b)`, a common coarsening always exists -- in the worst
  case, the integer `size(a)` itself.
* `common_coarsening` exists iff `size(a) == size(b)`.
  Accepts any object that has a CuTe shape (e.g. `Layout`, `Tensor`) via `shape(...)`.

*Examples:*

```python
common_coarsening((2, 15), (2, (3, 5)))     == (2, 15)
common_coarsening((4, (3, 5)), ((2, 2), 15)) == (4, 15)
common_coarsening(30, (2, 15))              == 30
common_coarsening((2, (3, 5)), ((3, 2), 5)) == 30
common_coarsening((6, 5), (2, 15))          == 30           # mode 0 mismatch -> int
common_coarsening((2, 3), (2, 3, 1))        == 6            # rank mismatch -> int
common_coarsening(3, 4)                     -> ValueError   # size mismatch
common_coarsening(7, (2, 3))                -> ValueError   # size mismatch
```

### `idx2crd(idx, shape)`

Map any coordinate to a *natural* coordinate of `shape`.

Input is decomposed in *colexicographical* order (leftmost mode varies fastest).
The final mode keeps the full quotient (its `mod` is skipped), so an
out-of-bounds `idx` does not wrap -- the excess accumulates in the last leaf.

A scalar `idx` may also be a non-`Integer` stride scalar that supplies an
`_idx2crd` hook -- `ArithTuple` and `F2` both do -- which is what lets a value
drawn from a layout's codomain be fed back in as a coordinate.

*Pre-conditions:*

```
weakly_congruent(idx, shape)
```

*Post-conditions:*

```
congruent(result, shape)
right-inverse of `crd2idx` on in-bounds inputs:
  crd2idx(idx2crd(i, S), S) == i   for i in range(size(S))
```

*Examples:*

```python
idx2crd(7,    14)          == 7
idx2crd(7,    (3, 2, 4))   == (1, 0, 1)
idx2crd(7,    (3, (2, 4))) == (1, (0, 1))
idx2crd(7,    ((3, 2), 4)) == ((1, 0), 1)
idx2crd(42,   (3, 7, 2))   == (0, 0, 2)      # out of bounds: last leaf absorbs excess
idx2crd(None, (3, (2, 4))) == (0, (0, 0))
idx2crd(F2(0b10110), (4, 8)) == (F2(0b10), F2(0b101))    # carry-less bit split
```

### `crd2idx(crd, shape)`

Map any coordinate of `shape` to an integral coordinate.

Input is recomposed in *colexicographical* order (leftmost mode varies fastest).

*Pre-conditions:*

```
weakly_congruent(crd, shape)
```

*Post-conditions:*

```
congruent(result, 0)
inverse of `idx2crd` on in-bounds inputs:
  idx2crd(crd2idx(c, S), S) == c   for c in coordinates(S)
  crd2idx(idx2crd(i, S), S) == i   for i in range(size(S))
```

*Examples:*

```python
crd2idx((1, 0, 1),   (3, 2, 4))   == 7
crd2idx((1, (0, 1)), (3, (2, 4))) == 7
crd2idx(7,           (3, (2, 4))) == 7      # integral coordinate passes through
crd2idx((2, 5),      (3, (2, 3))) == 17     # flat coordinate of a hierarchical shape
```

### `coordinates(shape)`

Generate all natural coordinates of `shape`, in *colexicographical* order.

*Post-conditions:*

```
list(coordinates(s)) == [idx2crd(i, s) for i in range(size(s))]
[crd2idx(c, s) for c in coordinates(s)] == list(range(size(s)))
```

*Examples:*

```python
list(coordinates(6))           == [0, 1, 2, 3, 4, 5]
list(coordinates((3, 2)))      == [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)]
list(coordinates((2, (2, 2)))) == [(0,(0,0)), (1,(0,0)), (0,(1,0)), (1,(1,0)),
                                   (0,(0,1)), (1,(0,1)), (0,(1,1)), (1,(1,1))]
```

---

## Module: `atuple`

Source: [`pycute/atuple.py`](../pycute/atuple.py)
Tests: [`test_atuple.py`](../test/test_atuple.py)

Arithmetic Tuples and related utilities.

An `ArithTuple` is an element of `Z^S`: a hierarchical tuple of stride scalars
under elementwise addition and scalar multiplication, with implicit
zero-extension along trailing positions. It is the single carrier for both
*coordinate strides* (one-term sums like `E(0)`) and *coordinate sums*
(multi-term sums like `3*E(0) + 5*E(1)`).

A leaf is ordinarily an integer, hence `Z^S`, but may be any `StrideScalar`, and
every operation defers to the leaf's own algebra. An `F2` leaf therefore adds by
XOR while its siblings keep integer addition, which is what lets one coordinate
axis carry a swizzled offset while another stays an ordinary index.

An instance carries one field, `self.data`, holding the children verbatim as
given. The same algebraic element therefore admits several representations --
`ArithTuple(1, 0)`, `ArithTuple((1,))` and `E(0)` all denote `1*e_0` while
holding different `data` -- and equality, which extends trailing positions by
zero, identifies them.

A scalar and an `ArithTuple` always differ in depth, and the single-scalar
passthrough in `__new__` means `ArithTuple(1)` is the depth-0 `int 1`; use
`ArithTuple((1,))`, `ArithTuple(1, 0)` or `E(0)` for the depth-1 element.

Pretty printing is hybrid: a single nonzero leaf renders in the basis form
`value@p_n@...@p_0`, everything else as a Python tuple.

### `class ArithTuple(*args)`

An element of the hierarchical module `Z^S`: a hierarchical tuple of stride
scalars under elementwise addition and scalar multiplication, with implicit
zero-extension along trailing positions.

Closed under `+`, `-` and scalar `*`, elementwise and to any depth:

```
ArithTuple(A,B,ArithTuple(C,D)) + ArithTuple(W,X,ArithTuple(Y,Z))
  := ArithTuple(A+W,B+X,ArithTuple(C+Y,D+Z))
X * ArithTuple(A,B,ArithTuple(C,D))
  := ArithTuple(X*A,X*B,ArithTuple(X*C,X*D))
```

Addition forms an abelian *group*: `int 0` is the unique identity and every
element has a negation, so `-` is elementwise too. Unlike `+` it does not
commute, so `0 - x` negates `x`. Adding or subtracting a nonzero scalar is an
incompatibility error.

Unhashable, because equivalent representations hold different `data` and so a
structural hash would violate `a == b => hash(a) == hash(b)`.

*Examples:*

```python
ArithTuple(1, 2, 3) + (7, 8, 9)               == (8, 10, 12)
ArithTuple(1, 2, 3) * 4                       == (4, 8, 12)
0 - ArithTuple(1, 2)                          == (-1, -2)
E(0) == ArithTuple(1, 0) == ArithTuple((1,))
ArithTuple(0, 0) == 0 == ArithTuple((0,))
ArithTuple((5,)) != 5                                     # different depths
ArithTuple(1, 2) + 1                          -> TypeError
```

#### `ArithTuple.__matmul__(other)`

`x @ i` wraps `x` at outer index `i`: `i` leading zeros, then `x`.

*Examples:*

```python
E(0) @ 1 == E(1, 0)
```

### `ScaledBasis(value, mode=())`

A scaled basis vector at path `mode`, with `value` kept verbatim at the leaf.

Returns the canonical scalar / `ArithTuple` representation, so an empty `mode`
collapses to `value` itself and `ScaledBasis(F2(1), (0,))` is an `F2`-valued
axis.

```
ScaledBasis(A,[])    := A
ScaledBasis(A,[0])   := (A,0,0,...)
ScaledBasis(A,[1])   := (0,A,0,...)
ScaledBasis(A,[0,0]) := ((A,0,0,...),0,0,...)
ScaledBasis(A,[0,1]) := ((0,A,0,...),0,0,...)
ScaledBasis(A,[1,0]) := (0,(A,0,0,...),0,...)
ScaledBasis(A,[1,1]) := (0,(0,A,0,...),0,...)
```

*Examples:*

```python
ScaledBasis(42, [])      == 42
ScaledBasis(42, [0])     == 42 * E(0)
ScaledBasis(42, [1, 0])  == 42 * E(1, 0)
```

### `E(*mode)`

Unit basis element: `E(*mode) == ScaledBasis(1, mode)`.

The usual way to write a coordinate stride.

```
E()    := 1
E(0)   := (1,0,0,...)
E(1)   := (0,1,0,...)
E(0,0) := ((1,0,0,...),0,0,...)
E(0,1) := ((0,1,0,...),0,0,...)
E(1,0) := (0,(1,0,0,...),0,...)
E(1,1) := (0,(0,1,0,...),0,...)
```

*Examples:*

```python
E()                  == 1
E(0)                 == ArithTuple(1, 0)
E(1)                 == ArithTuple(0, 1)
E(0, 1)              == ArithTuple(ArithTuple(0, 1), 0)
E(1, 0)              == ArithTuple(0, ArithTuple(1, 0))
Layout((4, 5), (E(0), E(1)))(2, 3) == ArithTuple(2, 3)
```

### `class V(value)`

Basis-scalar shortcut: `V(value) @ i` is sugar for `ScaledBasis(value, (i,))`.

```
V(1)     := 1
V(1)@0   := (1,0,0,...)
V(1)@1   := (0,1,0,...)
V(1)@0@0 := ((1,0,0,...),0,0,...)
V(1)@1@0 := ((0,1,0,...),0,0,...)
V(1)@0@1 := (0,(1,0,0,...),0,...)
V(1)@1@1 := (0,(0,1,0,...),0,...)
```

*Examples:*

```python
V(1) @ 0    == E(0)
V(42) @ 1   == 42 * E(1)
V(F2(1)) @ 0 == ScaledBasis(F2(1), (0,))
```

### `basis_repr(x)`

Algebraic decomposition of `x` into scaled basis vectors.

Each entry is one nonzero leaf of `x` with its path, and its value is that
leaf verbatim, of whatever algebra it belongs to. When every leaf of `x` is
zero the decomposition collapses to the single rank-zero term `[(0, ())]`,
matching the decomposition of `int 0`.

*Post-conditions:*

```
x == sum(v * E(*s) for v, s in basis_repr(x))
len(basis_repr(x)) >= 1
```

*Examples:*

```python
basis_repr(5 * E(1, 2))          == [(5, (1, 2))]
basis_repr(3 * E(0) + 5 * E(1))  == [(3, (0,)), (5, (1,))]
basis_repr(0)                    == [(0, ())]
basis_repr(ArithTuple(0, 0))     == [(0, ())]
```

### `is_basis(x)`

True iff `x` is a single scaled basis vector `v * E(*s)`.

Every Python `int` qualifies, being the rank-zero basis element.

*Examples:*

```python
is_basis(5 * E(1, 2))          == True
is_basis(0)                    == True
is_basis(3 * E(0) + 5 * E(1))  == False
```

### `make_basis_like(profile, mode=())`

Build a `profile`-shaped tuple of unit basis elements, one per leaf position.

*Post-conditions:*

```
congruent(result, profile)
get[path](result) == E(*path)   for every leaf path of profile
```

*Examples:*

```python
make_basis_like((10, 20))         == (E(0), E(1))
make_basis_like((10, (20, 30)))   == (E(0), (E(1, 0), E(1, 1)))
make_basis_like(10)               == E()
congruent(make_basis_like((10, (20, 30))), (10, (20, 30))) == True
```

### `proj(x, basis)`

Extract from `x` the part at the position implied by `basis`.

*Pre-conditions:*

```
is_basis(basis); a multi-term sum raises a TypeError
```

*Examples:*

```python
proj((7, 9), E(0))                == 7
proj((7, 9), 2 * E(1))            == 9
proj((7, 9), 42)                  == (7, 9)          # rank-zero path
proj((7, 9), E(0) + E(1))         -> TypeError
```

### `unit(basis)`

The multiplicative unit of `basis`'s algebra, at `basis`'s basis path.

Drops a stride scalar's magnitude while keeping the algebra and the axis it
lives on, so that `unit(d) * n` rebuilds a stride of magnitude `n` in the same
place -- which is how `recast` produces a stride of the same *type* as its
input. A scalar type supplies `_unit` when its algebra's identity is not
`int 1`; `F2` does, since `int 1` would scale by ordinary rather than
carry-less multiplication.

*Pre-conditions:*

```
is_basis(basis); a multi-term sum raises a TypeError
```

*Examples:*

```python
unit(5)                        == 1                          # Z
unit(2 * E(1))                 == E(1)                       # Z^S: axis kept
unit(F2(9))                    == F2(1)                      # F2, via _unit
unit(ScaledBasis(F2(9), (0,))) == ScaledBasis(F2(1), (0,))    # an F2 axis
unit(E(0) + E(1))              -> TypeError
```

### `as_tuple(atuple)`

Materialize an `ArithTuple`, or a tuple/list of them, as a plain nested tuple.

*Examples:*

```python
as_tuple(ArithTuple(1, 2, (3, 4)))  == (1, 2, (3, 4))
as_tuple(42)                        == 42
```

---

## Module: `layout`

Source: [`pycute/layout.py`](../pycute/layout.py)
Tests: [`test_layout.py`](../test/test_layout.py), [`test_make_layout.py`](../test/test_make_layout.py), [`test_recast.py`](../test/test_recast.py)

Definition of CuTe Layouts and functions to manipulate them

### `class LayoutBase`

Marker base class for every layout.

Subclassing it is what makes `is_layout` true, which lets the algebra and the
visualizers recognize a layout without importing `Layout` itself.

### `is_layout(x)`

True iff `x` is a layout, i.e. any `LayoutBase`.

*Examples:*

```python
is_layout(Layout((4, 8)))  == True
is_layout((4, 8))          == False
is_layout(42)              == False
```

### `class Layout(shape, stride=1)`

A CuTe Layout: a map from a coordinate domain to a codomain, defined by a
`shape` (an HTuple of Integers) and a congruent `stride` (an HTuple of stride
scalars).

Evaluates as `L(c) == inner_product(idx2crd(c, shape), stride)`, so every
coordinate form -- integral, flat, natural -- reaches the same value. The
default `stride` is the compact column-major `prefix_product(shape)`; pass one
explicitly, or as a single integer base, to override it.

The algebra is exposed as free functions in `algebra.py`; the `_`-prefixed
methods here implement its core operations.

*Examples:*

```python
Layout((4, 8))               == Layout((4, 8), (1, 4))   # default compact column-major
Layout((4, 8), (8, 1))(2, 3) == 19                       # evaluate a coordinate
A = Layout((3, (2, 4)), (2, (1, 6)))
A(17) == A(2, 5) == A(2, (1, 2)) == 17                    # the three coordinate forms
A[1][0] == Layout(2, 1)                                  # index into the modes
```

#### `Layout.__call__(*crd)`

Map a coordinate to the layout's codomain:
`L(c) == inner_product(idx2crd(c, shape), stride)`.

Accepts a coordinate in any form (integral, flat, or natural), passed either
as a single argument or as separate per-mode arguments.

*Examples:*

```python
L = Layout((4, 8), (8, 1))
L(14) == 19
L(2, 3) == 19
L((2, 3)) == 19
```

#### `Layout.__getitem__(i)`

Get mode `i` of the layout as a sublayout (tuple-like indexing over modes).

#### `Layout.get(mode=())`

Get the sublayout at the given (possibly nested) `mode` path.

#### `Layout.__eq__(other)`

Two Layouts are equal iff their shapes and strides are equal.

#### `Layout.__str__()`

Compact `shape:stride` form, e.g. `(4, 8):(1, 4)`.

#### `Layout.__repr__()`

Constructor form, e.g. `Layout((4, 8), (1, 4))`.

### `make_layout(layouts)`

Concatenate multiple Layouts; each input becomes one mode of the result.

*Post-conditions:*

```
rank(result) == len(layouts)
result[i] == layouts[i]   for i in range(rank(result))
```

*Examples:*

```python
make_layout([Layout(3, 1), Layout((5, 1), (7, 2)), Layout(2, 42)])
    == Layout((3, (5, 1), 2), (1, (7, 2), 42))
make_layout([]) == Layout((), ())
```

### `make_layout_like(layout)`

Construct a compact Layout with the same shape as `layout` whose strides
follow the ordering induced by `layout`'s strides.

The mode with the smallest non-zero source stride receives stride 1, and the
remaining non-zero modes receive compact (prefix-product) strides in stable
ascending order of the source stride magnitudes. Modes that carry no positional
information -- a size-1 shape or a static stride of 0 -- are pinned to stride 0.

Only static strides can be ordered by magnitude; symbolic (non-static) strides
are considered larger than every static stride.

*Post-conditions:*

```
shape(result) == shape(layout)
the non-zero modes of result form a compact (densely-packed) layout
idempotent: make_layout_like(make_layout_like(A)) == make_layout_like(A)
```

*Examples:*

```python
make_layout_like(Layout((4, 8), (1, 4)))              == Layout((4, 8), (1, 4))
make_layout_like(Layout((4, 8), (100, 1)))            == Layout((4, 8), (8, 1))
make_layout_like(Layout((2, 3, 4, 5), (0, 42, 4, 0))) == Layout((2, 3, 4, 5), (0, 4, 1, 0))
```

### `make_ordered_layout(_shape, _order)`

Construct a compact Layout with the same shape as `_shape` whose strides
follow the ordering induced by `_order`.

The mode with the smallest `_order` receives stride 1, and the remaining
modes receive compact (prefix-product) strides in ascending order of
`_order`. Only the relative ordering of the `_order` values matters, not
their magnitudes, so they need not be a contiguous `0..rank-1` permutation.

Only static orders can be ordered by magnitude; symbolic (non-static) orders
are considered larger than every static order and, being mutually
incomparable, retain their left-to-right order.

*Pre-conditions:*

```
congruent(_shape, _order)
```

*Post-conditions:*

```
shape(result) == _shape
the modes of result form a compact (densely-packed) layout
```

*Examples:*

```python
make_ordered_layout((4, 8), (0, 1))              == Layout((4, 8), (1, 4))
make_ordered_layout((4, 8), (1, 0))              == Layout((4, 8), (8, 1))
make_ordered_layout((2, 3, 4, 2), (0, 2, 3, 0))  == Layout((2, 3, 4, 2), (1, 4, 12, 2))
```

### `Tiler`

```python
Tiler: TypeAlias = Union[Integer, 'Layout', tuple['Tiler', ...], list['Tiler']]
```

An HTuple(Integer | Layout): the by-mode tiler argument to the algebra.

### `tiler_to_layout(tiler, e=1)`

Transform a "Tiler" (an HTuple of Layout|Integer) into a Layout that acts
identically under composition.

*Post-conditions:*

```
shape(result) == shape(tiler)
composition(A, result) == composition(A, tiler)        for all admissible Layouts A
logical_divide(A, result) == zipped_divide(A, tiler)   for all admissible Layouts A
```

*Examples:*

```python
tiler_to_layout(3)                          == Layout(3, 1)
tiler_to_layout(Layout((7, 2), (3, 1)))     == Layout((7, 2), (3, 1))
tiler_to_layout((4, 5))                     == Layout((4, 5), (E(0), E(1)))       # prints as (1@0, 1@1)
tiler_to_layout((Layout(4, 2), Layout(5, 3))) == Layout((4, 5), (2*E(0), 3*E(1))) # prints as (2@0, 3@1)
```

### `recast(layout, scale)`

Recast a Layout to a new element scale.

Rewrites both shape and stride so the layout addresses a differently-sized
element: `scale = 8` packs 8 source elements per new element, while
`scale = Fraction(1, 2)` unpacks 2 new elements per source element. Each leaf
`s:d` is rescaled by the ratio between `d` and `scale` -- shrinking the shape
when packing, growing it when unpacking.

*Pre-conditions:*

```
at each leaf the stride and scale divide cleanly (one is a multiple of the
other); otherwise a ValueError is raised.
```

*Examples:*

```python
from fractions import Fraction
recast(Layout(24, 1), 8)          == Layout(3, 1)
recast(Layout(24, 2), 4)          == Layout(12, 1)
recast(Layout((4, 4), (4, 1)), 4) == Layout((4, 1), (1, 1))
recast(Layout((4, 4), (4, 1)), Fraction(1,2)) == Layout((4, 8), (8, 1))
```

---

## Module: `algebra`

Source: [`pycute/algebra.py`](../pycute/algebra.py)
Tests: [`test_blocked_raked.py`](../test/test_blocked_raked.py), [`test_coalesce.py`](../test/test_coalesce.py), [`test_coalesce_z.py`](../test/test_coalesce_z.py), [`test_complement.py`](../test/test_complement.py), [`test_composition.py`](../test/test_composition.py), [`test_greatest_common_domain.py`](../test/test_greatest_common_domain.py), [`test_inverse_left.py`](../test/test_inverse_left.py), [`test_inverse_right.py`](../test/test_inverse_right.py), [`test_layout_add.py`](../test/test_layout_add.py), [`test_logical_divide.py`](../test/test_logical_divide.py), [`test_logical_product.py`](../test/test_logical_product.py), [`test_nullspace.py`](../test/test_nullspace.py)

Generic algebraic operations that dispatch to Layout or Tensor methods.

### `coalesce_z(A, profile=1, *, mode=())`

Coalesce a Layout or Tensor into a maximally-merged, equivalent form while
preserving trailing size-1 modes.

A non-empty `mode` coalesces only that mode of `A` and leaves every other mode
unchanged.

*Post-conditions:*

```
size(result) == size(A)
depth(result) <= 1   at each leaf of profile, within mode
result(i) == A(i)   for all integers i
```

*Examples:*

```python
coalesce_z(Layout((2, 1, 6, 1), (1, 7, 8, 0)))     == Layout((2, 6, 1), (1, 8, 0))
coalesce_z[1](Layout((3, (2, 6)), (1, (3, 6))))    == Layout((3, 12), (1, 3))
```

### `coalesce(A, profile=1, *, mode=())`

Coalesce a Layout or Tensor into a simpler, equivalent form.

Like `coalesce_z`, but additionally drops a trailing size-1 mode, matching C++
`cute:coalesce`. The integral evaluation `A(i)` is preserved for every
in-bounds `i`; the natural-coordinate evaluation generally is not, since the
coordinate space changes. `profile` selects whole-layout (`1`) vs by-mode
(tuple) coalescing, and `None` is a no-op. Integers and tuples are promoted via
`tiler_to_layout`.

A non-empty `mode` coalesces only that mode of `A` and leaves every other mode
unchanged: `coalesce[1](A)` is `coalesce(A, (None, 1))`.

*Post-conditions:*

```
size(result) == size(A)
depth(result) <= 1   at each leaf of profile, within mode
result(i) == A(i)   for i in range(size(A))
```

*Examples:*

```python
coalesce(Layout((2, (1, 6)), (1, (6, 2))))         == Layout(12, 1)
coalesce(Layout((2, 4, 6), (24, 6, 1)))            == Layout((2, 4, 6), (24, 6, 1))
coalesce(Layout((2, 1, 6, 1), (1, 7, 8, 0)))       == Layout((2, 6), (1, 8))
coalesce(Layout((2, (1, 6)), (1, (6, 2))), (1, 1)) == Layout((2, 6), (1, 2))
coalesce[1](Layout((3, (2, 6)), (1, (3, 6))))      == Layout((3, 12), (1, 3))
```

### `composition(A, B, *, mode=())`

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

*Pre-conditions:*

```
A and B satisfy the shape- and stride-divisibility conditions
(Whitepaper, Eqs. (20)-(21)); otherwise a ValueError is raised.
mode names a mode of A: rank[mode[:-1]](A) > mode[-1]
```

*Post-conditions:*

```
compatible(B, get[mode](result))  -- B refines result's domain
get[mode](result)(i) == get[mode](A)(B(i))   for i in range(size(B))
```

*Examples:*

```python
composition(Layout((6, 2), (8, 2)), Layout((4, 3), (3, 1))) == Layout(((2, 2), 3), ((24, 2), 8))
composition(Layout(20, 2), Layout((5, 4), (4, 1)))          == Layout((5, 4), (8, 2))
composition(Layout(12), Layout((4, 3)))                     == Layout((4, 3), (1, 4))
composition[1](Layout((4, 6), (1, 4)), Layout(3, 2))        == Layout((4, 3), (1, 8))
composition(None, (4, 3))                                   == Layout((4, 3), (E(0), E(1)))
```

### `right_inverse(A)`

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

*Post-conditions:*

```
result(A(result(i))) == result(i)  for i in range(size(result))
```

*Examples:*

```python
right_inverse(Layout((4, 8), (1, 4)))         == Layout(32, 1)
right_inverse(Layout((4, 8), (8, 1)))         == Layout((8, 4), (4, 1))
right_inverse(Layout((4, 8), (1, 5)))         == Layout(4, 1)
right_inverse(Layout((8, 8), (F2(1), F2(9)))) == Layout((8, 8), (F2(1), F2(9)))
right_inverse(Layout((8, 8), (F2(9), F2(1)))) == Layout((8, 8), (F2(8), F2(9)))
```

### `left_inverse(A)`

Left inverse of a Layout.

Returns a layout `R` with `A(R(A(k))) == A(k)`; when `A` is injective this is a
true inverse on the domain, `R(A(k)) == k`. Unlike `right_inverse`, when `A`'s
image is non-contiguous the left inverse extends its codomain to recover the
original coordinate.

*Pre-conditions:*

```
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
```

*Post-conditions:*

```
A(result(A(i))) == A(i)  for i in range(size(A))
```

*Examples:*

```python
left_inverse(Layout((4, 8), (1, 4))) == Layout(32, 1)
left_inverse(Layout((4, 8), (1, 5))) == Layout((5, 8), (1, 4))
```

### `complement(A, extend=None)`

Complement of a Layout, optionally extended to cover `extend`.

Returns a layout whose image fills the codomain "holes" of `A`: it is weakly
congruent to `A`'s codomain, strictly ordered, and disjoint from `A`. The free
`complement(A)` is the *minimal* complement; pass `extend` (a shape) to grow it
to a target size.

There are two regimes, governed by whether `A`'s sorted stride chain is
*divisible* -- i.e. each running extent `d_{k-1} * s_{k-1}` divides the next
stride `d_k` (modes ordered by stride as `d_0 <= d_1 <= ...` with sizes `s_k`):

* Divisible: the complement tiles, i.e. `make_layout([A, complement(A)])` is
  a bijection onto a contiguous range. This is the strong/typical case.
* Not divisible: the result is the *largest* ordered, disjoint layout that
  fits; it still satisfies the post-conditions below but does NOT tile (it
  under-fills the codomain).

*Pre-conditions:*

```
A's nonzero strides are non-overlapping (injective): each
`d_k >= d_{k-1} * s_{k-1}`. Enforced where statically decidable; otherwise a
ValueError is raised.
```

*Post-conditions:*

```
weakly_congruent(coprofile(A), result)
result(i-1) < result(i)   for i in range(1, size(result))  -- ordered
result(i) != A(j)                                          -- disjoint
```

*Examples:*

```python
complement(Layout(4, 2))                      == Layout((2, 1), (1, 8))
complement(Layout((2, 2), (1, 6)))            == Layout((3, 1), (2, 12))
complement(Layout(4, 2), Layout(20, 1).shape) == Layout((2, 3), (1, 8))
```

### `logical_product(A, B, *, mode=())`

Reproduce layout `A` over the layout of tiles `B`: `A x B = (A, A* o B)`.

The rank-2 result places a copy of `A` (mode-0) at each position of `B`
(mode-1), where mode-1 is `A`'s complement composed with `B`. A tuple `B`
applies by-mode and `B=None` is a no-op. An integer or tuple `A` is promoted
via `tiler_to_layout` before `B` is applied, so a by-mode `B` sees the
promoted Layout's modes.

A non-empty `mode` reproduces only that mode of `A` over `B` and leaves every
other mode unchanged.

*Post-conditions:*

```
rank(get[mode](result)) == 2  when is_layout(B)
size(result) == size(A) * size(B)  when is_layout(B)
get[mode](result)[0] == get[mode](A)
compatible(B, get[mode](result)[1])
```

*Examples:*

```python
logical_product(Layout((2, 2), (4, 1)), Layout(6, 1)) == Layout(((2, 2), (2, 3)), ((4, 1), (2, 8)))
logical_product(Layout(3, 1), Layout(4, 1))           == Layout((3, 4), (1, 3))
logical_product[0](Layout((3, 5), (1, 20)), Layout(4, 1))
    == Layout(((3, 4), 5), ((1, 3), 20))
```

### `logical_divide(A, B, *, mode=())`

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

*Pre-conditions:*

```
B divides A (the underlying composition's divisibility conditions hold);
otherwise a ValueError is raised.
mode names a mode of A: rank[mode[:-1]](A) > mode[-1]
```

*Post-conditions:*

```
rank(get[mode](result)) == 2  when is_layout(B)
compatible(B, get[mode](result)[0])
get[mode](result)[0] == composition(get[mode](A), B)
```

*Examples:*

```python
logical_divide(Layout(24), Layout(4, 2))         == Layout((4, (2, 3)), (2, (1, 8)))
logical_divide[1](Layout((3, 8)), Layout(4, 2))  == Layout((3, (4, 2)), (1, (6, 3)))
logical_divide(None, Layout(4, 2))               == Layout((4, (2, 1)), (2, (1, 8)))
```

### `zipped_divide(A, B, *, mode=())`

Logical divide of `A` by the tiler `B`, with `B` promoted to a Layout first.

Equivalent to `logical_divide(A, tiler_to_layout(B))`. Promoting the tiler `B`
to a single Layout zips the tile modes together and the remainder modes
together, so the result is `((tile...), (rest...))` rather than
`logical_divide`'s per-mode interleaving.

A non-empty `mode` divides only that mode of `A` and leaves every other mode
unchanged.

*Post-conditions:*

```
rank(get[mode](result)) == 2
compatible(B, get[mode](result)[0])
get[mode](result)[0] == composition(get[mode](A), B)
```

*Examples:*

```python
zipped_divide(Layout((9, 32)), (Layout(3, 3), Layout((2, 4), (1, 8))))
    == Layout(((3, (2, 4)), (3, 4)), ((3, (9, 72)), (1, 18)))
zipped_divide[1](Layout((5, 24)), Layout(4, 2)) == Layout((5, (4, (2, 3))), (1, (10, (5, 40))))
```

### `blocked_product(A, B)`

Rank-sensitive product that lays out copies of tile `A` in a *blocked*
arrangement over `B`.

Computes `logical_product(A, B)` and then interleaves modes so that, per mode,
the tile factor precedes the grid factor -- each tile stays contiguous before
the grid steps. Contrast `raked_product`, which swaps that order.

*Pre-conditions:*

```
rank(A) == rank(B); otherwise a ValueError is raised.
```

*Post-conditions:*

```
rank(result) == rank(A) == rank(B)
```

*Examples:*

```python
blocked_product(Layout((2, 5), (5, 1)), Layout((3, 4), (1, 3)))
    == Layout(((2, 3), (5, 4)), ((5, 10), (1, 30)))
```

### `raked_product(A, B)`

Rank-sensitive product that distributes copies of tile `A` *raked* (cyclically
interleaved) through `B`.

Computes `logical_product(A, B)` and then interleaves modes so that, per mode,
the grid factor precedes the tile factor -- the tile elements are spread across
grid positions instead of being contiguous. Contrast `blocked_product`.

*Pre-conditions:*

```
rank(A) == rank(B); otherwise a ValueError is raised.
```

*Post-conditions:*

```
rank(result) == rank(A) == rank(B)
```

*Examples:*

```python
raked_product(Layout((2, 5), (5, 1)), Layout((3, 4), (1, 3)))
    == Layout(((3, 2), (4, 5)), ((10, 5), (30, 1)))
```

### `nullspace(A)`

Nullspace of a Layout: the layout of coordinates that `A` maps to 0.

Collects the stride-0 modes of `A` and returns a compact layout enumerating
exactly the coordinates `c in Z(A)` with `A(c) == 0`. `None` returns `None`;
integers and tuples are promoted via `tiler_to_layout`.

*Post-conditions:*

```
A(result(i)) == 0   for i in range(size(result))
```

*Examples:*

```python
nullspace(Layout((4, 5), (E(0), E(1)))) == Layout(1, 0)
nullspace(Layout((4, 5), (0, E(1))))    == Layout(4, 1)
nullspace(Layout((2, 4, 6), (1, 2, 0))) == Layout(6, 8)
```

### `layout_add(A, B)`

Add two Layouts coordinate-wise.

Given Layouts `A` and `B` with `size(A) == size(B)`, return Layout `R` with

```
size(R) == size(A) == size(B)
R(i)    == A(i) + B(i)        for i in range(size(R))
```

`A` and `B` need not be compatible.
The result `R` is also not required to be compatible with `A` or `B`.

When no such Layout 'R' exists, a `ValueError` is raised.

*Pre-conditions:*

```
size(A) == size(B)
```

*Post-conditions:*

```
size(result) == size(A)
result(i) == A(i) + B(i)   for i in range(size(result))
symmetric:  layout_add(A, B) == layout_add(B, A)
```

*Examples:*

```python
layout_add(Layout(12, 1),          Layout(12, 1))          == Layout(12, 2)
layout_add(Layout(5, 0),           Layout(5, 1))           == Layout(5, 1)
layout_add(Layout((4, 3), (1, 4)), Layout((4, 3), (3, 1))) == Layout((4, 3), (4, 5))
layout_add(Layout((5, 3, 4), (1, 5, 15)),
           Layout((10, 6),     (1, 10)))                   == Layout(60, 2)
```

### `greatest_common_domain(A, B)`

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

*Post-conditions:*

```
symmetric: greatest_common_domain(A, B) == greatest_common_domain(B, A)
depth(result) == 1
size(result) divides math.gcd(size(A), size(B))
composition(A, result) and composition(B, result) are always admissible
greatest_common_domain(logical_divide(shape(A), result)[1],
                       logical_divide(shape(B), result)[1]) == Layout((1,), (0,))
```

*Examples:*

```python
greatest_common_domain((10,), (10,))        == Layout((10,), (1,))
greatest_common_domain((16, 3), (16, 3))    == Layout((16, 3), (1, 16))
greatest_common_domain((5, 3, 4), (10, 6))  == Layout((5, 2), (1, 30))
greatest_common_domain((6, 35), (15, 14))   == Layout((3, 7), (1, 30))
greatest_common_domain((5, 3), (3, 5))      == Layout((1,), (0,))
```

---

## Module: `swizzle`

Source: [`pycute/swizzle.py`](../pycute/swizzle.py)
Tests: [`test_swizzle.py`](../test/test_swizzle.py)

Methods for layout swizzling

A swizzle permutes the offsets a layout produces, to spread accesses across
shared-memory banks. `F2` expresses one as a stride: because its `+` is XOR, an
`F2`-strided `Layout` is an ordinary layout that happens to XOR its coordinates
together, and so passes through the whole algebra. `Swizzle` is the same
transformation written as a plain function on integers, for inspecting an
existing offset pattern.

### `class F2(value)`

A stride scalar over the binary field: `+` is XOR and `*` is carry-less.

An element of `F_2^m = (Z_{2^m}, XOR, .)`, whose bits are the coefficients of
a polynomial over the two-element field. Both the `F2 * F2` and `F2 * int`
products are carry-less, so an integer operand acts through its bits rather
than its value; the two agree wherever schoolbook multiplication carries
nowhere, as it does for a power-of-two operand.

Adding a nonzero `int` is an error rather than a coercion, since mixing `Z`'s
carrying addition into XOR is the hazard the separate type exists to prevent.

*Examples:*

```python
F2(0b1010) + F2(0b1100) == F2(0b0110)
3 * F2(0b1010)          == F2(0b11110)
F2(0b11) * 0b11         == F2(0b101)     # carry-less, where 3 * 3 == 9 in Z
F2(1) + 0               == F2(1)         # 0 is still the identity
F2(1) + 1               -> TypeError
Layout((4, 8), (F2(1), F2(8)))(2, 1) == F2(2 ^ 8)
```

#### `F2.__divmod__(other)`

Euclidean division in `F2`: the unique `(q, r)` satisfying

```
`self == q * other + r`   with   `deg(r) < deg(other)`,
```

where an `F2` value's bits are the coefficients of a polynomial over the
two-element field and `deg(F2(v)) == v.bit_length() - 1`.

`F2`'s `*` is a carry-less product, so this is polynomial long division
rather than integer division: `divmod(F2(0b1011), 0b11) == (F2(0b110),
F2(0b1))`, whereas `divmod(11, 3) == (3, 2)` in `Z`.

Both results are `F2`. Keeping the quotient in `F2` is what lets `idx2crd`
chain `divmod` across the modes of a shape and `crd2idx` recompose the
result, with every step staying carry-less.

Since `deg(r) < deg(other)` implies `r < other`, the remainder is always an
in-bounds coordinate of an extent-`other` mode. When `other` is a power of
two -- the only case in which the modes of an `F2` layout occupy disjoint
bit-fields -- the split is exactly the pair of bit-fields, and the
quotient's value is the ordinary shift:

```
`divmod(F2(a), 2**k) == (F2(a >> k), F2(a & (2**k - 1)))`.
```

*Examples:*

```python
divmod(F2(0b10110), 4)  == (F2(0b101), F2(0b10))
divmod(F2(0b1011), 3)   == (F2(0b110), F2(0b1))
divmod(F2(42), 1)       == (F2(42), F2(0))
```

### `shiftr(a, s)`

Shift `a` right by `s`, or left by `-s` when `s` is negative.

*Examples:*

```python
shiftr(0b1000, 2)   == 0b10
shiftr(0b1000, -2)  == 0b100000
```

### `shiftl(a, s)`

Shift `a` left by `s`, or right by `-s` when `s` is negative.

*Examples:*

```python
shiftl(0b1000, 2)   == 0b100000
shiftl(0b1000, -2)  == 0b10
```

### `class Swizzle(bits, base, shift)`

A generic Swizzle functor that can be applied to indices or addresses
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                              ^--^  Base is the number of least-sig bits to keep constant
                 ^-^       ^-^      Bits is the number of bits in the mask
                    ^---------^     Shift is the distance to shift the YYY mask
                                      (pos shifts YYY to the right, neg shifts YYY to the left)

e.g. Given
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
the result is
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY

*Pre-conditions:*

```
bits >= 0, base >= 0, and either shift >= 0 or abs(shift) >= bits;
otherwise a ValueError is raised
```

*Post-conditions:*

```
involution, when the YYY and ZZZ fields do not overlap (abs(shift) >= bits):
  S(S(offset)) == offset
```

*Examples:*

```python
Swizzle(1, 0, 1)(0b00)                    == 0b00
Swizzle(1, 0, 1)(0b10)                    == 0b11
Swizzle(1, 0, 1)(Swizzle(1, 0, 1)(0b10))  == 0b10
Swizzle(-1, 0, 1)                         -> ValueError
```

#### `Swizzle.__call__(offset)`

Apply the swizzle to an integer offset.

---

## Module: `accessor`

Source: [`pycute/accessor.py`](../pycute/accessor.py)
Tests: [`test_tensor.py`](../test/test_tensor.py)

Accessors: the memory of a CuTe Tensor

A `Layout` maps a coordinate to an offset; an accessor maps that offset to a
value. Keeping the two apart is what lets one layout address ordinary memory, a
lazily-computed function, or a transformed view without the layout algebra
knowing the difference.

Two ABCs fix the contract -- `Accessor` for read-only access and
`MutableAccessor` for read/write -- and both require `__add__`, so that an
accessor can be *offset* to yield another accessor.

### `class Accessor`

ABC for a read-only random-access handle.

### `class MutableAccessor`

ABC for a read/write random-access handle.

### `class Ptr(source, dtype=None, owner=None)`

A typed pointer into memory owned elsewhere: the general-purpose accessor.

`source` is either an integer address -- e.g. `array.buffer_info()[0]` or
`ndarray.__array_interface__['data'][0]` -- in which case `dtype` is
required, or an object that can provide one (`array.array`, `numpy.ndarray`,
a ctypes array, `bytearray`, ...), in which case `dtype` and the storage
`owner` are inferred from it.

Element `i` is read and written at `address + i * sizeof(dtype)`. Nothing is
allocated, copied, or bounds-checked, so a `Ptr` is exactly as valid as the
address it is handed. Prefer passing the object over a bare address: the
reference kept in `owner` stops the storage being collected while in use.

Offsetting yields another `Ptr` anchored to the same owner, so a chain of
offsets never accumulates wrappers.

*Examples:*

```python
data = array.array('d', [1.0, 2.0, 3.0, 4.0])
T = Tensor(Ptr(data), Layout((2, 2), (2, 1)))         # dtype inferred
T[1, 0] = 9.0
data.tolist() == [1.0, 2.0, 9.0, 4.0]
Ptr(data.buffer_info()[0], ctypes.c_double)[0] == 1.0 # a raw address
Ptr(data.buffer_info()[0]) -> ValueError              # ... needs a dtype
```

#### `Ptr.base()`

The accessor or object owning the storage; `self` when it is unowned.

### `class Array(size, dtype=c_double)`

A `Ptr` that allocates and owns `size` elements of `dtype`.

The allocation is kept alive by the `Array` itself, so unlike a bare `Ptr` it
stays valid for as long as it is reachable.

*Examples:*

```python
a = Array(16, dtype=ctypes.c_int)
a[5] = 42
a[5] == 42
```

### `class ImplicitAccessor(base)`

An accessor with no memory behind it: reading offset `i` returns `base + i`.

*Examples:*

```python
ImplicitAccessor(0)[7]          == 7
ImplicitAccessor(100)[7]        == 107
(ImplicitAccessor(0) + 100)[7]  == 107
```

### `class TransformAccessor(accessor, transform)`

Wraps an accessor and applies `transform` on every read:
`TransformAccessor(a, f)[i] == f(a[i])`.

Offsetting propagates to the inner accessor. Read-only: there is
no inverse with which to push a write back through `transform`.

*Examples:*

```python
a = Array(4, dtype=ctypes.c_int)
a[3] = 3
TransformAccessor(a, lambda x: x * x)[3] == 9
```

---

## Module: `tensor`

Source: [`pycute/tensor.py`](../pycute/tensor.py)
Tests: [`test_tensor.py`](../test/test_tensor.py)

CuTe Tensor

A `Tensor` is an `Accessor` composed with a `Layout`: the layout turns a
coordinate into an offset, the accessor turns that offset into a value.

### `class Tensor(accessor, layout)`

A CuTe Tensor: an `Accessor` paired with a `Layout`.

Indexing accepts a coordinate in any of the forms a `Layout` does. A
coordinate that leaves modes unnamed, marking them with `None` or `:`, returns
a sub-tensor view over those modes instead of an element.

The layout algebra applies through the tensor -- `coalesce`, `composition`,
`logical_divide` and `zipped_divide` rewrap the same accessor -- so reshaping
or tiling a tensor never touches its data.

*Examples:*

```python
T = make_tensor(Layout((4, 4), (4, 1)))
T[1, 2] = 42.0
T[1, 2] == 42.0
shape(T) == (4, 4)
T[1, None] == T[1, :]                          # a row, as a sub-tensor
shape(T[1, None]) == (4,)
T[1, None][2] == 42.0                          # sharing T's storage
coalesce(make_tensor(Layout((4, 4), (1, 4)))).layout == Layout(16, 1)
```

#### `Tensor.shape()`

Shape of the layout domain

#### `Tensor.__getitem__(i)`

Read the element at `i`, or return the sub-tensor `i` leaves unnamed.

*Examples:*

```python
T = make_tensor(Layout((4, 4), (4, 1)))
T[2, 3]            == 0.0
shape(T[None, 2])  == (4,)
```

#### `Tensor.__setitem__(i, value)`

Write `value` at `i`.

*Pre-conditions:*

```
`i` names every mode; a slicing coordinate raises a ValueError
```

#### `Tensor.get(mode=())`

Get the sub-tensor at the given (possibly nested) `mode` path.

#### `Tensor.__eq__(other)`

Two Tensors are equal iff their accessors and layouts are equal.

### `is_tensor(x)`

True iff `x` is a `Tensor`.

*Examples:*

```python
is_tensor(make_tensor(Layout((2, 2))))  == True
is_tensor(Layout((2, 2)))               == False
```

### `identity_tensor(shape)`

The tensor mapping every coordinate to itself.

Coordinate strides over an `ImplicitAccessor`, so nothing is allocated and
reading a position yields the coordinate that reaches it. Tiling it the same
way as a data tensor is how a tile recovers its global coordinates, which is
what predication needs.

*Examples:*

```python
identity_tensor((3, 4))[1, 2]   == (1, 2)
shape(identity_tensor((3, 4)))  == (3, 4)
```

### `make_tensor(layout, dtype=c_double)`

Allocate an `Array` of size `coshape(layout)` and bind it to `layout`.

To bind a layout to data you already have, wrap it in a `Ptr` instead.

*Pre-conditions:*

```
coshape(layout) is an Integer; a coordinate codomain raises a ValueError
```

*Examples:*

```python
shape(make_tensor(Layout((4, 4), (4, 1))))  == (4, 4)
make_tensor((2, 3)).layout                  == Layout((2, 3), (1, 2))
make_tensor(Layout((2, 2), (E(0), E(1))))   -> ValueError
```

---

## Module: `util.print_tensor`

Source: [`pycute/util/print_tensor.py`](../pycute/util/print_tensor.py)

Utilities for printing CuTe Layouts and Tensors as ASCII tables

### `print_tensor(tensor, print_type=True)`

Print a layout or tensor of rank 1 through 4 as nested ASCII tables.

A `Layout` is rendered through an `ImplicitAccessor`, so each cell shows its
offset; a `Tensor` shows the element stored there. Ranks 3 and 4 are printed
as a series of rank-2 slices. `print_type` prepends the `Shape:Stride` header.

*Pre-conditions:*

```
rank(tensor) <= 4; otherwise a ValueError is raised
```

---

## Module: `util.print_table`

Source: [`pycute/util/print_table.py`](../pycute/util/print_table.py)

Utilities for printing CuTe Layouts as bordered grids

### `print_table(tensor, print_type=True)`

Render a rank-2 layout or tensor as a single bordered grid, one cell per
coordinate.

Like `print_tensor`, a `Layout` is rendered through an `ImplicitAccessor` and
a `Tensor` shows its elements, with `print_type` prepending the
`Shape:Stride` header. A non-rank-2 input falls back to a plain `print`.

Requires the optional `tabulate` package (`pip install pycute[viz]`).

---

## Module: `util.draw_svg`

Source: [`pycute/util/draw_svg.py`](../pycute/util/draw_svg.py)

Utilities for SVG generation of CuTe Layouts

### `draw_svg(tensor, filename='layout.svg', color=white)`

Save a rank-2 `Tensor` or `Layout` as an SVG table with row/column labels.

A `Layout` labels each cell with its offset, a `Tensor` with the element
stored there. A rank-1 input is drawn as a single row. `color` is a functor
`color(idx) -> (r, g, b)` keyed on the cell's offset; `draw_colors` holds a
catalog of them.

Requires the optional `svgwrite` package (`pip install pycute[viz]`).

*Pre-conditions:*

```
rank(tensor) is 1 or 2; otherwise a ValueError is raised
```

### `draw_svg_tv(layout, tile_mn=None, filename='tvlayout.svg', color=thread_color_8x)`

Save a rank-2 thread-value layout as a colored SVG table with `T`/`V` labels.

The layout maps `(tid, vid)` to a position in the tile. Its codomain may be
2-D `(m, n)` coordinates, or a linear offset -- in which case a rank-2
`tile_mn` is required and the offsets are folded into the tile via
`composition(tile_mn, layout)`. `color` is a functor
`color(tid, vid) -> (r, g, b)`; where several `(tid, vid)` pairs land on one
cell, the first drawn wins.

Requires the optional `svgwrite` package (`pip install pycute[viz]`).

*Pre-conditions:*

```
rank(layout) == 2 and rank(tile_mn) == 2
the codomain is 2-D once folded; otherwise a ValueError is raised
```

---

## Module: `util.draw_latex`

Source: [`pycute/util/draw_latex.py`](../pycute/util/draw_latex.py)

Utilities for LaTeX/PDF generation of CuTe Layouts

### `draw_latex(tensor, filename='layout.tex', compile_pdf=True, color=white)`

The LaTeX/PDF analogue of `draw_svg`, mirroring `cute:print_latex`.

Writes a standalone TikZ document for a rank-2 `Tensor` or `Layout` (rank-1
becomes a single row) and, by default, compiles it to a cropped PDF with
`pdflatex`. A missing or failing `pdflatex` leaves the `.tex` in place rather
than raising. `color` has the same contract as in `draw_svg`.

*Pre-conditions:*

```
rank(tensor) is 1 or 2; otherwise a ValueError is raised
```

### `draw_latex_tv(layout, tile_mn=None, filename='tvlayout.tex', compile_pdf=True, color=thread_color_8x)`

The LaTeX/PDF analogue of `draw_svg_tv`.

Takes the same thread-value layouts -- a 2-D `(m, n)` codomain, or a linear
offset folded through a rank-2 `tile_mn` -- and renders them as TikZ with
`T`/`V` annotations, compiling to PDF by default. `color` has the same
contract as in `draw_svg_tv`.

*Pre-conditions:*

```
rank(layout) == 2 and rank(tile_mn) == 2
the codomain is 2-D once folded; otherwise a ValueError is raised
```

---

## Module: `util.draw_colors`

Source: [`pycute/util/draw_colors.py`](../pycute/util/draw_colors.py)

Coloring functors shared by the CuTe Layout visualizers (draw_svg / draw_latex)

A `color` functor maps a cell key to an `(r, g, b)` tuple with each component in
`[0, 255]`. The drawers use two key shapes, and this module holds a catalog of
each:

* offset functors take one integer offset, `color(idx) -> (r, g, b)`:
  `index_grey_8x`, `bank_color_8x`, `bank_color_16x`, `bank_color_32x`
* thread-value functors take a `(tid, vid)` pair,
  `color(tid, vid) -> (r, g, b)`: `thread_color_8x`, `value_color_8x`,
  `warp_color_8x`
* `white` and `constant(rgb)` ignore their key, so they serve either shape

`white` is the default of `draw_svg` / `draw_latex` and `thread_color_8x` that of
`draw_svg_tv` / `draw_latex_tv`; pass any functor of the matching signature to
override them.

### `index_grey_8x(idx)`

Greyscale shade by `idx % 8` -> `(r, g, b)`; default offset coloring.

### `bank_color_8x(idx)`

Color by `idx % 8` -> `(r, g, b)` from the light spectrum.

Like `bank_color_32x` but cycling every 8 -- handy for 8-bank groupings
or when fewer, more distinct colors read better.

### `bank_color_16x(idx)`

Color by `idx % 16` -> `(r, g, b)` from the light spectrum.

Like `bank_color_32x` but cycling every 16.

### `bank_color_32x(idx)`

Color by shared-memory bank `idx % 32` -> `(r, g, b)`.

Spreads the 32 banks around a light spectrum so equal-bank cells share a
color -- handy for spotting shared-memory bank conflicts.

### `thread_color_8x(tid, vid)`

Color by `tid % 8` -> `(r, g, b)` (`vid` ignored); default TV coloring.

### `value_color_8x(tid, vid)`

Color by value index `vid % 8` -> `(r, g, b)` (`tid` ignored).

### `warp_color_8x(tid, vid)`

Color by warp `(tid // 32) % 8` -> `(r, g, b)` (32 threads per warp).

### `constant(rgb)`

Return a coloring functor that ignores its key and always yields `rgb`.

### `white(*args)`

Constant white `(255, 255, 255)`; valid for either functor signature.

---

## Tests

Each test module states what it covers in its own docstring; this
table is generated from those.

| Test | What it checks |
|---|---|
| [`test_alg_copy.py`](../test/test_alg_copy.py) | Unit tests for pycute.alg.copy and pycute.alg.ref.copy |
| [`test_atuple.py`](../test/test_atuple.py) | Unit tests for pycute.atuple |
| [`test_blocked_raked.py`](../test/test_blocked_raked.py) | Unit tests for pycute.blocked_product and pycute.raked_product |
| [`test_coalesce.py`](../test/test_coalesce.py) | Unit tests for pycute.coalesce |
| [`test_coalesce_z.py`](../test/test_coalesce_z.py) | Unit tests for pycute.coalesce_z |
| [`test_compatibility.py`](../test/test_compatibility.py) | Unit tests for pycute.congruent, pycute.weakly_congruent and pycute.compatible. |
| [`test_complement.py`](../test/test_complement.py) | Unit tests for pycute.complement |
| [`test_composition.py`](../test/test_composition.py) | Unit tests for pycute.composition |
| [`test_docs_exports.py`](../test/test_docs_exports.py) | Ensure the docs keep describing pycute's public API accurately. |
| [`test_docstring_examples.py`](../test/test_docstring_examples.py) | Evaluate every example in every pycute docstring. |
| [`test_greatest_common_domain.py`](../test/test_greatest_common_domain.py) | Unit tests for pycute.greatest_common_domain |
| [`test_htuple.py`](../test/test_htuple.py) | Unit tests for pycute.htuple, and for pycute.stride's leaf arithmetic |
| [`test_inverse_left.py`](../test/test_inverse_left.py) | Unit tests for pycute.left_inverse |
| [`test_inverse_right.py`](../test/test_inverse_right.py) | Unit tests for pycute.right_inverse |
| [`test_layout.py`](../test/test_layout.py) | Unit tests for pycute.layout |
| [`test_layout_add.py`](../test/test_layout_add.py) | Unit tests for `pycute.layout_add`. |
| [`test_logical_divide.py`](../test/test_logical_divide.py) | Unit tests for pycute.logical_divide |
| [`test_logical_product.py`](../test/test_logical_product.py) | Unit tests for pycute.logical_product |
| [`test_make_layout.py`](../test/test_make_layout.py) | Unit tests for pycute.make_layout, pycute.tiler_to_layout, pycute.make_layout_like, and pycute.make_ordered_layout |
| [`test_nullspace.py`](../test/test_nullspace.py) | Unit tests for pycute.nullspace |
| [`test_recast.py`](../test/test_recast.py) | Unit tests for pycute.layout recast |
| [`test_swizzle.py`](../test/test_swizzle.py) | Unit tests for pycute.swizzle |
| [`test_tensor.py`](../test/test_tensor.py) | Unit tests for pycute.tensor and pycute.accessor |
| [`test_typing.py`](../test/test_typing.py) | Unit tests for pycute.typedefs |

Run them all with:

```sh
pytest
```

## Copyright

Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
