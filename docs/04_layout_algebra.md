# Layout Algebra

The CuTe **layout algebra** is the small set of operations that combine
layouts to produce new layouts. Every higher-level CuTe operation — tiling,
partitioning, predication, vectorization, swizzling — is built out of these
primitives.

This chapter covers the layout-algebra operations exported from
[`algebra.py`](../pycute/algebra.py) (plus `recast` in [`layout.py`](../pycute/layout.py)):

| Operation | Function | Module | Tests |
|---|---|---|---|
| Coalesce | `coalesce` | [`algebra.py`](../pycute/algebra.py) | [`test_coalesce.py`](../test/test_coalesce.py) |
| Coalesce-Z | `coalesce_z` | same | [`test_coalesce_z.py`](../test/test_coalesce_z.py) |
| Composition | `composition` | same | [`test_composition.py`](../test/test_composition.py) |
| Complement | `complement` | same | [`test_complement.py`](../test/test_complement.py) |
| Logical divide | `logical_divide`, `zipped_divide` | same | [`test_logical_divide.py`](../test/test_logical_divide.py) |
| Logical product | `logical_product`, `blocked_product`, `raked_product` | same | [`test_logical_product.py`](../test/test_logical_product.py), [`test_blocked_raked.py`](../test/test_blocked_raked.py) |
| Right inverse | `right_inverse` | same | [`test_inverse_right.py`](../test/test_inverse_right.py) |
| Left inverse | `left_inverse` | same | [`test_inverse_left.py`](../test/test_inverse_left.py) |
| Nullspace | `nullspace` | same | [`test_nullspace.py`](../test/test_nullspace.py) |
| Layout add | `layout_add` | same | [`test_layout_add.py`](../test/test_layout_add.py) |
| Greatest common domain | `greatest_common_domain` | same | [`test_greatest_common_domain.py`](../test/test_greatest_common_domain.py) |
| Recast | `recast` | [`layout.py`](../pycute/layout.py) | [`test_recast.py`](../test/test_recast.py) |

The Whitepaper's *(§3 Layout Algebra)* is the source of truth for every
post-condition you will see below. The C++ documentation's
[02_layout_algebra.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md)
is also a useful pedagogical companion (with diagrams).

## A guiding principle

Every operation in this algebra is defined first as a function on
"1-D layouts" — i.e. layouts treated as functions from `int` to `int` —
and then *lifted* to higher-rank layouts via two patterns:

1. **By-mode application**: if the second argument is a tuple of layouts
   (a *Tiler*), apply the operation to corresponding modes pairwise. PyCuTe
   uses the combinator
   `(A_0, A_1) ★ ⟨B, C⟩ = (A_0 ★ B, A_1 ★ C)` *(Whitepaper, Eq. (12))*.
2. **None-as-noop**: if the second argument is `None`, the operation is the
   identity. This is a uniform way to skip a mode.

In PyCuTe's code, you will see this pattern repeated for `coalesce`,
`composition`, `complement`, `logical_divide`, and `logical_product`:

```python
if profile is None:
    return self
if is_tuple(profile):
    if rank(self) < len(profile): raise ValueError(...)
    return make_layout(a._operation(p) for a, p in zip_longest(self, profile))
# ... 1-D base case below ...
```

## Coalesce

> A *coalesced* layout `R` of `A` satisfies
> `|R| = |A|`, `depth(R) <= 1`, and `R(i) == A(i)` for every integer `i`
> in `[0, |A|)`.
>
> *(Whitepaper, §3.2.)*

`coalesce` simplifies a layout *as a function from `int` to `int`*. It can
remove rank, flatten hierarchy, and merge consecutive modes whose strides
align. The integral evaluation `A(i)` is preserved; the natural-coordinate
evaluation `A(c)` may not be (since the natural coordinate space changed).

```python
>>> from pycute import *
>>> coalesce(Layout((2, (1, 6)), (1, (6, 2))))
Layout(12, 1)                           # collapses to a single mode
>>> coalesce(Layout((2, 4, 6), (24, 6, 1)))
Layout((2, 4, 6), (24, 6, 1))           # already minimal
>>> coalesce(Layout(((2, 2), (2, 2)), ((1, 4), (8, 32))))
Layout((2, 4, 2), (1, 4, 32))           # one pair merges, the other does not
```

The fundamental rule for two adjacent integer modes `s0:d0 ++ s1:d1` is:

| Pattern | Coalesces to | Reason |
|---|---|---|
| `s0:d0  ++  1:d1` | `s0:d0` | Size-1 mode is irrelevant |
| `1:d0  ++  s1:d1` | `s1:d1` | Same |
| `s0:d0  ++  s1:(s0*d0)` | `(s0*s1):d0` | Strides line up: collapse |
| `s0:d0  ++  s1:d1` (otherwise) | `(s0,s1):(d0,d1)` | Cannot coalesce |

PyCuTe's core fold lives in [`stride.py`](../pycute/stride.py) as `_coalesce_z`;
[`layout.py`](../pycute/layout.py) `Layout._coalesce` / `_coalesce_z` call it and
apply the trailing size-1 policy for `coalesce`. The implementation flattens the
layout and folds adjacent modes left-to-right; whenever a size-1 mode is removed,
algorithm recurses so that the now-newly-adjacent neighbors get a fresh
chance to merge. As a result, layouts like `(2, (1, 6)):(1, (6, 2))`
collapse all the way down to `12:1` even though the size-1 mode sits in
the middle.

### `coalesce` vs `coalesce_z`

There are two flavors of coalescing:

* **`coalesce(A, profile=1)`** — the conventional version. After folding,
  any *trailing* mode of size 1 is dropped. This makes `coalesce` agree
  with the C++ `cute::coalesce`.
* **`coalesce_z(A, profile=1)`** — preserves trailing size-1 modes. This is
  useful when you need the full "extended" function (where calling at an
  out-of-bounds integer also matters): `R(i) == A(i)` for *every* integer
  `i`, not just integers in `[0, |A|)`.

```python
>>> coalesce(  Layout((2, 1, 6, 1), (1, 7, 8, 0)))
Layout((2, 6), (1, 8))                # 1's dropped, last 1:0 also dropped
>>> coalesce_z(Layout((2, 1, 6, 1), (1, 7, 8, 0)))
Layout((2, 6, 1), (1, 8, 0))          # last 1:0 retained
```

(See [`test_coalesce.py`](../test/test_coalesce.py) and
[`test_coalesce_z.py`](../test/test_coalesce_z.py).)

### By-mode coalesce

`coalesce(A, profile)` applies `coalesce` mode-by-mode at the leaves of
`profile`. Use `1` as a leaf flag to "coalesce here" and a sub-tuple to
"coalesce within this mode":

```python
>>> A = Layout((2, (1, 6)), (1, (6, 2)))
>>> coalesce(A, 1)              # whole-layout coalesce
Layout(12, 1)
>>> coalesce(A, (1, 1))         # by-mode: each mode coalesced independently
Layout((2, 6), (1, 2))
>>> coalesce(A, None)           # noop
Layout((2, (1, 6)), (1, (6, 2)))
```

## Composition

> The *group composition* `R = A ∘ B` is the layout that satisfies
> `B ⪯ R` and `R(i) == A(B(i))` for every `i ∈ Z(B)`.
>
> *(Whitepaper, §3.3.)*

PyCuTe writes composition as `composition(A, B)`. `B` defines the domain
of the result; `A` defines the codomain.

```python
>>> composition(Layout((6, 2), (8, 2)), Layout((4, 3), (3, 1)))
Layout(((2, 2), 3), ((24, 2), 8))

>>> composition(Layout(20, 2), Layout((5, 4), (4, 1)))
Layout((5, 4), (8, 2))                   # reshape 20 elements as 5x4 row-major

>>> composition(Layout((10, 2), (16, 4)),
...             Layout((5, 4), (1, 5)))
Layout((5, (2, 2)), (16, (80, 4)))       # column-major reshape
```

(See [`test_composition.py`](../test/test_composition.py) for many
worked examples and the post-condition checker.)

### Distributivity and the base case

Composition is **left-distributive over concatenation of `B`**:

`A ∘ (B_0, B_1, ...) = (A ∘ B_0, A ∘ B_1, ...)`

so we may always reduce the problem to "compose `A` (a coalesced layout)
with `B = s : d` (a single integer mode)". *(Whitepaper, Eq. (19).)*

The *base case* `A ∘ (s : d)` produces the layout whose first `s` elements
are `A` strided by `d`. The implementation uses two divisibility checks:

* **Stride divisibility** (Eq. (20)): for each prefix product `S̄_r` of
  `A`'s shape, either `S̄_r | d` or `d | S̄_r`. This ensures the strides
  divide cleanly.
* **Shape divisibility** (Eq. (21)): for each prefix product, the result
  must have size `s`.

Examples that satisfy both:

```python
>>> composition(Layout(12), Layout((4, 3)))
Layout((4, 3), (1, 4))
>>> composition(Layout((4, 3)), Layout(12))
Layout(12, 1)
>>> composition(Layout((4, 3), (3, 1)), Layout((6, 2), (2, 1)))
Layout((6, 2), (2, 1))
```

Examples that fail divisibility raise `ValueError`:

```python
>>> composition(Layout((5, 3), (7, 1)), Layout(2, 3))
Traceback (most recent call last):
  ...
ValueError: Stride divisibility condition violated: ...
>>> composition(Layout((5, 3), (7, 1)), Layout(7, 1))
Traceback (most recent call last):
  ...
ValueError: Shape divisibility condition violated: ...
```

### Right-tilers and by-mode composition

When `B` is a tuple-of-layouts (a *tiler*), composition applies
mode-by-mode:

```python
>>> A = Layout((12, (4, 8)), (59, (13, 1)))
>>> tiler = (Layout(3, 4), Layout(8, 2))
>>> composition(A, tiler)
Layout((3, (2, 4)), (236, (26, 1)))
```

PyCuTe interprets a *Shape* as "tilers with stride 1 in each mode":

```python
>>> composition(A, (3, 8))               # equivalent to (3:1, 8:1)
Layout((3, (4, 2)), (59, (13, 1)))
```

(See *(Whitepaper, §3.3.5)* and the C++
[02_layout_algebra.md § By-mode Composition](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md)
for the diagrams.)

### Composition with coordinate strides

When `B`'s stride is a basis element `a · e_i` (i.e. `is_basis(B.stride)`
is true), composition with `B` first selects mode `i` of `A` and then
composes with `s : a`:

```python
>>> composition(Layout((4, 5), (5, 1)), Layout(30, E(0)))
Layout((4, 5), (5, 1))                   # E(0) selects mode-0 path
>>> composition(Layout((4, 5), (5, 1)), Layout(12, E(1)))
Layout((4, 3), (1, 4))                   # E(1) selects mode-1 path
```

This recovers the *Reductive Case: Coordinate* from
*(Whitepaper, §3.3.2)*. (See
[`test_composition.py::TestComposition::test_composition_coords`](../test/test_composition.py).)

## Complement

> The *complement* `L*` of a layout `L` is the unique layout that:
> 1. is *weakly congruent* to the codomain of `L`
> 2. produces an *ordered* sequence of offsets, and
> 3. has *disjoint image* from `L`.
>
> *(Whitepaper, §3.5.)*

Intuitively, `L*` is "what `L` is missing": the layout whose image is the
holes that `L` does not cover. This is the workhorse behind *logical
divide* and *logical product*.

```python
>>> complement(Layout(4, 2))
Layout((2, 1), (1, 8))
```

This is read as: "the layout `4:2` covers `0, 2, 4, 6`; the missing offset
that fits *between* its strides is `1` (so the complement starts with
`2:1`); after `4:2`'s coverage, the complement continues with stride 8
(so any further extension is at offset 8). The trailing `1:8` mode marks
where extension would resume."

PyCuTe's free `complement(L)` returns the **minimal** complement (just
enough to plug the gaps in the strides). To extend the complement up to a
given size, pass `extend=` — a shape — to the public API:

```python
>>> complement(Layout(4, 2), extend=20)
Layout((2, 3), (1, 8))                   # |L*| extended to fit a 20-element layout
```

This is how `logical_divide` builds an extended complement large enough to compose with
tensors and generate complete tile- and grid-modes.

```python
>>> complement(Layout((2, 2), (1, 6)))
Layout((3, 1), (2, 12))
```

The codomain of `(2, 2):(1, 6)` is the integer line, so the complement is
also a single integer-codomain layout that fills the holes.

The three conditions above are the **weak** post-condition, always guaranteed.
When `L`'s sorted strides additionally form a *divisible* chain — each running
extent `d_{k-1} * s_{k-1}` divides the next stride `d_k` — `(L, L*)` also
*tiles* the codomain (a bijection onto a contiguous range), the **strong**
post-condition. On a non-divisible chain the result still satisfies the weak
post-condition but under-fills (does not tile); this is exactly the regime
[`left_inverse`](#left-inverse) rejects instead of accepting a weaker result.

(See [`test_complement.py`](../test/test_complement.py) for the
strong post-condition checker, which also verifies that `(L, L*)` admits
a generalized inverse.)

### Complement on coordinate strides

`complement` works equally on coordinate-strided layouts. Sorting respects
the colexicographical ordering of the strides:

```python
>>> complement(Layout(3, E(0)))
Layout((1,), (3@0,))                     # 3:E(0) already fills mode 0; only the extension marker remains
>>> complement(Layout((2, 5, 3), (4 * E(1), 5 * E(0), 16 * E(1))))
Layout(((5, 1), (4, 2, 1)), ((1@0, 25@0), (1@1, 8@1, 48@1)))
```

(See [`test_complement.py::TestComplement::test_complement_coord`](../test/test_complement.py).)

## Logical divide

> `logical_divide(A, B)` splits `A` into:
>   * mode-0 — the elements pointed to by `B` (the *tile*),
>   * mode-1 — the layout of those tiles (the *grid*).
>
> Formally:
> `A ⊘ B = A ∘ (B, B*)` *(Whitepaper, §3.5.2)*.

```python
>>> logical_divide(Layout(24), Layout(4, 2))
Layout((4, (2, 3)), (2, (1, 8)))
```

For the result `R = (R[0], R[1])`:

* `R[0]` (the tile) has shape compatible with `B`.
* `R(i, 0) == A(B(i))` for every `i ∈ Z(B)` — composition with `B`.
* Every offset of `A` appears somewhere in the image of `R`.

```python
>>> A = Layout((6, 6), (1, 12))
>>> B = Layout((6, 3), (3, 1))
>>> R = logical_divide(A, B)
>>> R
Layout((((2, 3), 3), 2), (((3, 12), 1), 36))
>>> rank(R)
2
>>> compatible(B, R[0])
True
```

(See
[`test_logical_divide.py`](../test/test_logical_divide.py) for many
examples.)

### Dividing a single mode

A `None` tiler leaf is a no-op, so a tiler names the modes it divides and says
nothing about the others. Subscripting `logical_divide` builds that tiler for
you: `logical_divide[i, j, ...](A, B)` divides mode `(i, j, ...)` of `A` by `B`
and leaves every other mode of `A` unchanged.

```python
>>> A = Layout((3, 28))
>>> logical_divide[1](A, Layout(4, 1))          # divide mode 1 only
Layout((3, (4, 7)), (1, (3, 12)))
>>> logical_divide(A, (None, Layout(4, 1)))     # the tiler it builds
Layout((3, (4, 7)), (1, (3, 12)))
```

Every algebra operation whose right-hand side applies by-mode takes the same
subscript, each rebuilding `A` with only the named mode changed:
`coalesce[1](A)`, `composition[0](A, B)`, `logical_product[0](A, B)`, and
`zipped_divide[0](A, B)`. This is the
[mode-indexed operator](./01_htuple.md#mode-indexed-operators) convention that
`shape[0]`, `size[0, 1]`, and `get[0]` also follow.

### `zipped_divide`, `tiled_divide`, `flat_divide`

`logical_divide` preserves *semantics* of each mode (M-mode in, M-mode out)
but the resulting modes are
`((TileM, RestM), (TileN, RestN), ...)`. Often you want them rearranged.

PyCuTe implements `zipped_divide` (zips tiles together and rest together):

```python
>>> A = Layout((9, 32), ...)
>>> tiler = (Layout(3, 3), Layout((2, 4), (1, 8)))
>>> ld = logical_divide(A, tiler)        # ((3,3), (8,4))
>>> zd = zipped_divide(A, tiler)         # ((3,8), (3,4))
>>> zd[0]                                # tile modes zipped: (3, 8)
```

`zipped_divide(A, B) = logical_divide(A, tiler_to_layout(B))`. PyCuTe does
not currently implement `tiled_divide` or `flat_divide` as separate
functions, but they are simple rearrangements of `zipped_divide` if you
need them.

### Associativity

A useful identity: zipped_divide and composition commute through
by-mode composition. The unit test
[`test_logical_divide.py::TestLogicalDivide::test_zipped_divide_associativity`](../test/test_logical_divide.py)
checks the SM70 8x8x4 MMA case:

```python
data_layout = Layout((32, 64))
tiler_mn = (8, 8)
clayout_tv = Layout(((2, 2, 2), (2, 2, 2)), ((1, 16, 4), (8, 2, 32)))
c0 = composition(zipped_divide(data_layout, tiler_mn), (clayout_tv, None))
c1 = zipped_divide(data_layout, composition(tiler_mn, clayout_tv))
assert c0 == c1
```

## Logical product

> `logical_product(A, B)` reproduces `A` as a tile inside the layout of
> tiles `B`:
>
> `A ⊗ B = (A, A* ∘ B)` *(Whitepaper, §3.5.1)*.

The result is rank-2; mode-0 is `A`, mode-1 is `B` "with each element
replaced by a unique copy of `A`".

```python
>>> logical_product(Layout((2, 2), (4, 1)), Layout(6, 1))
Layout(((2, 2), (2, 3)), ((4, 1), (2, 8)))
>>> logical_product(Layout(3, 1), Layout(4, 1))
Layout((3, 4), (1, 3))
```

Post-conditions checked by
[`test_logical_product.py`](../test/test_logical_product.py):

* `rank(R) == 2`
* `R[0] == A`
* `compatible(B, R[1])`

### `blocked_product` and `raked_product`

`logical_product` always produces a rank-2 result. Often the natural
output for tile-of-tiles is *rank-sensitive*: you want the column mode of
the tile to combine with the column mode of the tile-of-tiles, and likewise
for rows. `blocked_product` and `raked_product` are the *rank-sensitive*
products that handle this.

```python
>>> tile = Layout((2, 5), (5, 1))           # 2x5 row-major tile
>>> grid = Layout((3, 4), (1, 3))           # 3x4 col-major grid of tiles
>>> blocked_product(tile, grid)
... # 2x5 row-major tiles laid out 3x4 column-major
>>> raked_product(tile, grid)
... # 2x5 row-major tile interleaved through the 3x4 grid (cyclic distribution)
```

The difference: `blocked_product` puts the tile mode *first*, then the grid
mode (so each tile is contiguous in memory); `raked_product` swaps that
order (so the tile elements are interleaved across grid positions).

(See *(Whitepaper, §3.5.1, Related Products)* and the C++
[02_layout_algebra.md § Blocked and Raked Products](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md)
for diagrams. PyCuTe's checked post-conditions and a worked
`blocked_product` vs. `raked_product` comparison live in
[`test_blocked_raked.py`](../test/test_blocked_raked.py).)

## Right inverse

> A *right inverse* of `L : Z_|L| → D` is an injective layout
> `L^‡ : D_{L^‡} → Z_|L|` such that
> `L^‡(L(L^‡(k))) == L^‡(k)` for every `k`.
>
> *(Whitepaper, §3.4.1.)*

In the common case where the codomain `D` is the integers, this reduces to
`L(L^‡(k)) == k` — the canonical right-inverse condition. PyCuTe always
returns the **largest right-inverse** by convention.

```python
>>> right_inverse(Layout((4, 8), (1, 4)))
Layout(32, 1)                            # the codomain is contiguous

>>> right_inverse(Layout((4, 8), (8, 1)))
Layout((8, 4), (4, 1))                   # row-major → its transpose

>>> right_inverse(Layout((4, 8), (1, 5)))
Layout(4, 1)                             # only the first 4 offsets are dense
```

For a `bijective` layout, the right inverse and left inverse coincide, and
it is also the full inverse `L^{-1}`.

(See [`test_inverse_right.py`](../test/test_inverse_right.py).)

### Application: vectorization

A common application of `right_inverse` is determining the largest
contiguous sub-layout shared between two layouts. This is the
*AutoVectorization* opportunity discussed in *(Whitepaper, §3.4.2)*. The
key observation:

```python
# Find the size K of the maximum common sublayout of A and B:
common = composition(B, right_inverse(A))    # = (I_K, X)
K = size(common[0])                          # size of the identity portion
```

(See
[`test_inverse_left.py::TestLeftInverse::test_left_inverse_app`](../test/test_inverse_left.py)
for a related application.)

## Left inverse

> A *left inverse* of `L` is a layout `L^†` such that
> `L(L^†(L(k))) == L(k)` for every `k ∈ Z_|L|`.
>
> *(Whitepaper, §3.4.3.)*

When `L` is injective, this reduces to `L^†(L(k)) == k`.

```python
>>> left_inverse(Layout((4, 8), (1, 4)))
Layout(32, 1)
>>> left_inverse(Layout((4, 8), (1, 5)))
Layout((5, 8), (1, 4))                   # larger than the right inverse
```

Note the difference from `right_inverse`: when the layout's image is
*non-contiguous*, the left inverse extends the codomain to recover the
original coordinate, whereas the right inverse truncates to the
contiguous portion.

**Pre-condition.** `left_inverse` only handles a layout that is left-invertible
*as a layout*: ordering the nonzero modes by stride as `d_0 < d_1 < ...` (with
sizes `s_k`), each stride must divide the next (`d_{k-1} | d_k`) and clear the
previous mode (`d_k >= d_{k-1} * s_{k-1}`). This *ordered-chain* condition is
sufficient but not necessary for injectivity, so `left_inverse` raises
`ValueError` for

* a **non-injective** `L` — overlapping strides, e.g. `Layout((63, 2), (1, 1))`; and
* an injective `L` whose strides **cannot be chained** — coprime strides, e.g.
  `Layout((2, 2), (2, 3))` (which *does* have a valid layout left inverse, but is
  rejected as a deliberate simplification).

The non-overlap half (`d_k >= d_{k-1} * s_{k-1}`) is the same requirement
[`complement`](#complement) enforces; the two diverge only on divisibility —
`complement` tolerates a non-divisible chain (returning the weaker ordered,
disjoint result), whereas `left_inverse` must reject it to guarantee a true
inverse.

(See [`test_inverse_left.py`](../test/test_inverse_left.py).)

### Application: TMEM admissibility

A useful application: given a data layout `A` and an instruction layout
`T`, determine whether every offset of `T` lies in the image of `A`, and
if so, where:

```python
# Suppose A is a TMEM data layout and T is an instruction's offset map.
# A^† ∘ T tells us where the instruction touches A, in coordinates.
locations = composition(left_inverse(A), T)
# Check admissibility:
for i in range(size(T)):
    assert A(left_inverse(A)(T(i))) == T(i)
```

(See *(Whitepaper, §3.4.4)*.)

## Nullspace

> `nullspace(L)` is the layout of all coordinates `c ∈ Z(L)` for which
> `L(c) == 0`.

```python
>>> nullspace(Layout((4, 5), (E(0), E(1))))
Layout(1, 0)                              # injective: trivial null space
>>> nullspace(Layout((4, 5), (0, E(1))))
Layout(4, 1)                              # mode 0 has stride 0 ⇒ contributes nothing
>>> nullspace(Layout((2, 4, 6), (1, 2, 0)))
Layout(6, 8)
```

`nullspace` is implemented in
[`Layout._nullspace`](../pycute/layout.py): it collects all stride-0
modes of the (flattened) layout and returns a new layout whose shape is
those mode sizes and whose stride is the corresponding prefix products of
the layout's shape.

Post-condition (see
[`test_nullspace.py`](../test/test_nullspace.py)):

```python
for i in range(size(nullspace(L))):
    assert L(nullspace(L)(i)) == 0
```

`nullspace` is useful for predication and for detecting "broadcast" modes
where the same data is read for many coordinates.

## Recast

> `recast(L, scale)` rewrites `L` to act on a different element size.
> When `scale = 8`, the layout is "recasting from byte to int8 with 8x
> packing"; when `scale = Fraction(1, 2)`, the layout is "recasting
> from int8 to int4 with 2x unpacking".

`recast` modifies both the shape and the stride to keep the resulting
layout sensible at the new element type. For a leaf shape `s` with stride
`d`:

* `dd, n = stride_value, scale_value` (with `n = 1/scale_unit`)
* If `dd == 0`: shape `s`, stride unchanged.
* If `dd == 1`: shape `ceil(s / n)`, stride 1 (the "shrink" case).
* If `n | dd`: shape `s`, stride `dd / n`.
* If `dd | n`: shape `ceil(s / (n / dd))`, stride `1`.

```python
>>> recast(Layout(24, 1), 8)
Layout(3, 1)
>>> recast(Layout(24, 1), Fraction(1, 2))
Layout(48, 1)
>>> recast(Layout(24, 2), 1)
Layout(24, 2)
>>> recast(Layout(24, 2), 4)
Layout(12, 1)
>>> recast(Layout((4, 4), (4, 1)), 4)
Layout((4, 1), (1, 1))
```

`recast` is the algebraic operation behind `cute::recast<T, U>(tensor)`:
"reinterpret this tensor of `T` as a tensor of `U`, with the layout
rescaled accordingly".

(See [`test_recast.py`](../test/test_recast.py) for an exhaustive
table of integer and `Fraction` scales.)

## `layout_add` and `greatest_common_domain`

PyCuTe also exports two analysis helpers beyond the core Whitepaper algebra:

* **`layout_add(A, B)`** — when `size(A) == size(B)`, return the unique
  (coalesced) layout `R` with `R(i) == A(i) + B(i)` on `[0, size(A))`.
  Requires a common refinement of the coalesced shapes; raises `ValueError`
  otherwise. See [`test_layout_add.py`](../test/test_layout_add.py).

* **`greatest_common_domain(A, B)`** — depends only on `shape(A)` and
  `shape(B)`; returns a rank-1 layout whose shape/stride factorization records
  the greatest aligned common divisor walk. See
  [`test_greatest_common_domain.py`](../test/test_greatest_common_domain.py).

## Generic dispatch via the `algebra` module

[`algebra.py`](../pycute/algebra.py) provides free functions that
dispatch on whether their argument has the corresponding `_method`. This
is what allows you to call `coalesce` on a `Layout`, a `Tensor`, or a
plain integer / tuple (which is auto-promoted with `tiler_to_layout`):

```python
def coalesce(A, profile=1):
    if hasattr(A, '_coalesce'):
        return A._coalesce(profile)
    if A is None:
        return None
    if is_int(A) or is_tuple(A):
        return tiler_to_layout(A)._coalesce(profile)
    raise TypeError(...)
```

The same pattern applies to `coalesce_z`, `composition`, `complement`,
`logical_divide`, `logical_product`, `right_inverse`, `left_inverse`, and
`nullspace`. This is the surface API users should reach for.

## Summary table

| Operation | Returns | Key invariants |
|---|---|---|
| `coalesce(A)` | `Layout` | `depth ≤ 1`, `R(i) == A(i)` for `i ∈ [0, |A|)` |
| `coalesce_z(A)` | `Layout` | `depth ≤ 1`, `R(i) == A(i)` for *all* integers |
| `composition(A, B)` | `Layout` | `B ⪯ R`, `R(i) == A(B(i))` |
| `complement(A)` | `Layout` | weakly congruent codomain, ordered, disjoint from `A` |
| `logical_divide(A, B)` | `Layout` | `R[0]` compatible with `B`; mode-0 is composition with `B` |
| `logical_product(A, B)` | `Layout` | `R[0] == A`; mode-1 compatible with `B` |
| `right_inverse(L)` | `Layout` | `L^‡(L(L^‡(k))) == L^‡(k)` |
| `left_inverse(L)` | `Layout` | `L(L^†(L(k))) == L(k)` |
| `nullspace(L)` | `Layout` | `L(R(i)) == 0` for every `i` |
| `layout_add(A, B)` | `Layout` | `R(i) == A(i) + B(i)` when defined |
| `greatest_common_domain(A, B)` | `Layout` | common shape factorization of `A`, `B` |
| `recast(L, scale)` | `Layout` | layout rescaled to new element size |

## Source and tests

* Source: [`pycute/algebra.py`](../pycute/algebra.py),
  [`pycute/layout.py`](../pycute/layout.py)
* Tests: see the per-operation table at the top of this chapter.

## Copyright

Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
