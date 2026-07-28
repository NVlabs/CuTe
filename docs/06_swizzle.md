# Swizzles and F2 Strides

CuTe's *swizzle* operations transform offsets via XOR rather than ordinary
integer arithmetic. They are the algebraic backbone of bank-conflict-avoiding
shared-memory layouts. PyCuTe expresses swizzles directly as **`F_2`-stride
layouts** — strides drawn from the integer-module `(Z, XOR, ·)` — so the
entire layout algebra (`coalesce`, `composition`, `complement`, inverses)
applies to them with no special cases.

* Source: [`swizzle.py`](../pycute/swizzle.py).
* Tests: [`test_swizzle.py`](../test/test_swizzle.py).

## Why swizzles?

GPU shared memory is divided into banks. If a warp accesses 32 consecutive
elements that all map to the same bank, the bank must serialize them, and
your kernel pays an N× latency penalty. By *permuting* offsets within each
row according to an XOR pattern, we can guarantee that each warp accesses
distinct banks for any natural access pattern.

Mathematically, the permutation is `offset → offset ^ (some_mask)`, where
the mask is itself derived from another portion of the offset. CuTe
expresses this as an integer-module with addition replaced by XOR.

(See *(Whitepaper, §2.3.1, "Field $F_2$")* and the *(Whitepaper, Figure 6,
Binary Swizzle)* example.)

## `F2` strides

The [`F2(value)`](../pycute/swizzle.py) class wraps a Python integer
in a stride scalar where `+` is `^` (XOR) and `*` is regular integer
multiplication.

```python
>>> from pycute import *
>>> F2(0b1010) + F2(0b1100)
F6                              # 0b0110 = 6
>>> F2(0b1010) * 2
F20                             # 0b10100 = 20
>>> 3 * F2(0b1010)
F30
```

Because addition is XOR, every `F2` value is its own additive inverse:
`F2(a) + F2(a) == F2(0)`. This is the property that makes layouts with
`F2` strides represent *invertible* permutations.

(See [`test_swizzle.py::TestF2`](../test/test_swizzle.py).)

`F2` is registered as a `StrideScalar`, so it can appear as a stride leaf
inside any `Layout`:

```python
>>> L = Layout((4, 8), (F2(1), F2(8)))
>>> L(2, 1)
F10                              # F2(1) * 2 + F2(8) * 1 = F2(2 XOR 8) = F2(10)
>>> L(0, 0)
F0
```

When this layout is evaluated, the `inner_product` call inside
`Layout.__call__` calls `F2.__add__` (XOR) instead of `int.__add__`, so
the result is the XOR of the products `c_i * d_i` rather than their sum.

(See [`test_swizzle.py::TestF2Layout`](../test/test_swizzle.py).)

> **Note.** PyCuTe's `F2` stores arbitrary integers; the "field $F_2$"
> aspect is not enforced — `F2(7) + F2(7) == F2(0)` because `7 ^ 7 == 0`
> and that is intentional, but `F2(3) * F2(5)` is *not* defined (only
> integer scaling is). This matches the *(Whitepaper, §2.3.1)* definition.

## The `Swizzle` functor

For inspection, debugging, and direct comparison with C++ `cute::Swizzle<B, M, S>`,
PyCuTe also provides a [`Swizzle(bits, base, shift)`](../pycute/swizzle.py)
*function object* — a plain function on integers that does **not** carry a
shape. It is *not* a `Layout`; it is a permutation on `Z`.

```
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                              ^--^   base (M):  number of LSB to keep constant
                 ^-^       ^-^       bits (B):  number of bits in the YY/ZZ masks
                    ^---------^      shift (S): distance to shift YY into ZZ
```

The functor's action on an offset:

```
offset → offset ^ ((offset & YYY_mask) >> shift)
```

That is: take the `bits` bits in `YYY`, shift them into the `ZZZ`
position, and XOR them in.

```python
>>> s = Swizzle(2, 0, 2)
>>> [s(i) for i in range(16)]
[0, 1, 2, 3, 5, 4, 7, 6, 10, 11, 8, 9, 15, 14, 13, 12]
```

`Swizzle` is its own inverse (XOR is involutive): `sw(sw(i)) == i` for
every integer `i`.

The constructor enforces:

* `bits ≥ 0`
* `base ≥ 0`
* If `shift < 0`, `|shift| ≥ bits` — i.e. negative shifts must be at least
  as wide as the bit field.

(See [`test_swizzle.py::TestSwizzle`](../test/test_swizzle.py) for
the XOR pattern, the involution check, and the constructor's validation.)

## A concrete example

The classic CuTe shared-memory swizzle for `M ⨯ N` row-major fp16 tiles
is an `F_2`-stride layout whose stride leaves XOR together to avoid bank
conflicts on any access pattern compatible with the layout's shape. The
exact constants depend on the tile size and element width; see the C++
documentation for shared-memory atom designs and the *(Whitepaper, Figure
6)* for a complete worked example.

A small illustrative case in PyCuTe:

```python
>>> L = Layout((4, (4, 3)), (F2(1), (F2(5), F2(16))))
>>> [int(L(i, (j, k))) for j in range(4) for k in range(3) for i in range(4)]
... # produces the offsets shown in (Whitepaper, Figure 6)
```

When CuTe needs to know "where does element `(i, (j, k))` live in shared
memory after applying the swizzle?", it evaluates `L(i, (j, k))`, getting
back the XOR-combined offset.

## Linear-algebraic perspective

The *(Whitepaper, §2.4.4 Semi-Linearity)* shows that an `F_2`-stride
layout can be viewed as a matrix-vector product over $\mathbb{F}_2$, where
each stride leaf is one column of a binary matrix. Well-studied
transformations of such matrices include the Bit Permute Complement (BCP)
and Bit Matrix Multiply Complement (BMMC) operations from the
permutation-network literature. In PyCuTe, `F2`-stride layouts participate in
the same layout-algebra API as integer layouts; [`test_swizzle.py`](../test/test_swizzle.py)
covers `F2` arithmetic and rank-2 layout evaluation, while broader algebra
combinations are not exhaustively tested for every `F2` case.

## Source and tests

* Source: [`pycute/swizzle.py`](../pycute/swizzle.py).
* Tests: [`test/test_swizzle.py`](../test/test_swizzle.py)
  covers `F2` arithmetic (`TestF2`), `Swizzle` (`TestSwizzle`), and
  `F2`-stride layouts (`TestF2Layout`). The full algebraic story for
  swizzles is best read in *(Whitepaper, §2.3.1, §2.4.4)*.

## Copyright

Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
