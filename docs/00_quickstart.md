# Getting started with PyCuTe

PyCuTe is a Python library for building, evaluating, composing, and slicing **layouts**
and **tensors** in the style of CuTe. Where C++ CuTe is a header-only template
library tightly coupled to CUDA, PyCuTe is a pure-Python reference
implementation that you can `import` from any Python script. This makes it the
right place to learn the algebra, prototype new transformations, and generate
test vectors for the C++ and DSL implementations.

## What CuTe is, in one paragraph

A CuTe **`Layout`** is a function from coordinates to offsets. It is built from
two hierarchical tuples called the **`Shape`** (which defines the
domain of coordinates) and the **`Stride`** (which defines how coordinates are transformed into
offsets). A small algebra — `coalesce`, `composition`, `complement`,
`right_inverse`, `left_inverse`, `nullspace`, `logical_divide`, 
`logical_product` — lets us manipulate and combine layouts to express tiling, partitioning,
vectorization, and layout analysis with functional algebraic primitives.
A **`Tensor`** is a `Layout` paired with an
**`Accessor`** (e.g. a pointer); evaluating the tensor at a coordinate evaluates
the layout to get an offset and dereferences the accessor at that offset.

That is the whole of CuTe. The rest of this documentation walks through each
piece carefully and shows how the pieces connect.

## Installing and running

PyCuTe is intended to be installed in-place from a source checkout. There are
no required external dependencies. The visualization utilities use
[`svgwrite`](https://pypi.org/project/svgwrite/) and
[`tabulate`](https://pypi.org/project/tabulate/) (and a LaTeX install for
`draw_latex`'s optional PDF output), and the typing tests use
[`sympy`](https://www.sympy.org/), but you can use the layout algebra without
any of those installed. Running the test suite needs
[`pytest`](https://docs.pytest.org); install everything with
`pip install -e ".[test]"`.

From the repository root:

```sh
# Run the test suite (uses pytest) with live logging.
pytest --log-cli-level DEBUG

# Or interactively:
python3
>>> from pycute import *
>>> Layout((3, 4), (4, 1))
Layout((3, 4), (4, 1))
```

The repository layout is:

| Path | Contents |
|---|---|
| [`docs/`](./index.md) | This documentation |
| [`examples/`](../examples/) | Standalone example scripts |
| [`test/`](../test/) | One `test_*.py` per algebraic operation |
| [`pycute/`](../pycute/) | The importable package: `htuple`, `atuple`, `shape`, `stride`, `layout`, `algebra`, `swizzle`, `tensor`, `accessor` |
| [`pycute/util/`](../pycute/util/) | Optional printing/visualization helpers: `print_tensor`, `print_table`, `draw_svg`, `draw_svg_tv`, `draw_latex`, `draw_latex_tv`, and `draw_colors` color functors (`index_grey_8x`, `bank_color_32x`, `thread_color_8x`, …) |

A single `from pycute import *` brings every public name into scope; this is
the convention used throughout this documentation and throughout the
[unit tests](../test/).

## A 60-second tour

Build a layout, evaluate it, and slice it:

```python
>>> from pycute import *
>>> A = Layout((3, (2, 4)), (2, (1, 6)))
>>> print(A)
(3, (2, 4)):(2, (1, 6))

>>> size(A), rank(A), depth(A)
(24, 2, 2)

>>> A(7)             # 1-D coordinate in 24-vector
8
>>> A(1, 2)          # 2-D coordinate in 3x8-matrix
8
>>> A(1, (0, 1))     # natural (h-D) coordinate in 3x(2x4)
8

>>> A[1]             # mode 1 as its own layout
Layout((2, 4), (1, 6))
>>> A[1][0]          # mode 1,0 as its own layout
Layout(2, 1)
```

(See [`test_layout.py`](../test/test_layout.py) for the analogous
expressions inside a unit test.)

Coalesce a layout, viewing it as a function from `int` to `int`:

```python
>>> coalesce(Layout((2, 4, 6), (24, 6, 1)))
Layout((2, 4, 6), (24, 6, 1))
>>> coalesce(Layout((2, (1, 6)), (1, (6, 2))))
Layout(12, 1)
```

(See [`test_coalesce.py`](../test/test_coalesce.py).)

Compose two layouts:

```python
>>> composition(Layout((6, 2), (8, 2)), Layout((4, 3), (3, 1)))
Layout(((2, 2), 3), ((24, 2), 8))
```

(See [`test_composition.py`](../test/test_composition.py).)

Tile a layout with `logical_divide`:

```python
>>> logical_divide(Layout(24, 1), Layout(4, 2))
Layout((4, (2, 3)), (2, (1, 8)))
```

(See [`test_logical_divide.py`](../test/test_logical_divide.py).)

Reproduce a layout with `logical_product`:

```python
>>> logical_product(Layout((2, 2), (4, 1)), Layout(6, 1))
Layout(((2, 2), (2, 3)), ((4, 1), (2, 8)))
```

(See [`test_logical_product.py`](../test/test_logical_product.py).)

Build a `Tensor` and read/write data:

```python
>>> from pycute import *
>>> from pycute.util import print_tensor   # printing helpers live in pycute.util
>>> T = make_tensor(Layout((4, 4), (4, 1)))   # 4x4 row-major
>>> T[1, 2] = 42.0
>>> T[1, 2]
42.0
>>> print_tensor(T)
<...Array...> o (4, 4):(4, 1)
0.0   0.0   0.0   0.0
0.0   0.0   42.0  0.0
0.0   0.0   0.0   0.0
0.0   0.0   0.0   0.0
```

(See [`05_tensor.md`](./05_tensor.md).)

## Where to go next

* Read [HTuples](./01_htuple.md) to understand the universal data structure
  underneath every PyCuTe object.
* Read [Shape and Stride](./02_shape_stride.md) for the algebra of shapes and
  the integer-modules of strides — including PyCuTe's first-class support
  for *coordinate strides* (`ArithTuple` and its basis elements `E(...)`) and
  *F2 strides* (used to describe XOR-based swizzles).
* Read [Layout](./03_layout.md) for the core `Layout` class.
* Read [Layout Algebra](./04_layout_algebra.md) for `coalesce`,
  `composition`, `complement`, `logical_divide`, `logical_product`,
  `right_inverse`, `left_inverse`, `nullspace`, and `recast`.
* When you need a function signature or a worked example, the
  [API reference](./08_api_reference.md) has every function with a link to
  its unit test.

## What PyCuTe is *not*

PyCuTe is a **reference implementation**, not a kernel-launching framework.
It does not generate or invoke CUDA code, does not depend on PyTorch or NVCC,
and does not implement device-side `copy`, `gemm`, MMA atoms, TMA, or
predication. Those concerns belong to
[CUTLASS C++](https://github.com/NVIDIA/cutlass) and the
[CuTe Python DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl.html); PyCuTe is
where the layout algebra lives in pure Python so it can be studied,
extended, and verified.

## Print and visualize

PyCuTe provides several printing utilities — all of them in `pycute.util`,
imported separately from the core algebra:

```python
>>> from pycute.util import print_tensor, print_table
>>> from pycute.util import draw_svg, draw_svg_tv
>>> from pycute.util import draw_latex, draw_latex_tv
>>> from pycute.util import index_grey_8x, thread_color_8x  # default color functors
```

* `print(layout)` — built-in, shows the `(shape):(stride)` notation.
* [`print_tensor(t)`](../pycute/util/print_tensor.py) — render a rank-1 to
  rank-4 layout or tensor as nested ASCII tables.
* [`print_table(t)`](../pycute/util/print_table.py) — render a rank-2
  layout or tensor as a grid table (uses `tabulate` for nicer borders).
* [`draw_svg(layout)`](../pycute/util/draw_svg.py) — render a rank-2
  layout as a colored SVG (offsets shaded, with row/column index labels).
* [`draw_svg_tv(tv_layout, tile_mn)`](../pycute/util/draw_svg.py) — render a
  thread-value layout (the kind that maps `(thread_id, value_id)` to a
  position in a tile) as an SVG with one cell per `(T, V)` pair.
* [`draw_latex(layout)`](../pycute/util/draw_latex.py) — the LaTeX/PDF
  analogue of `draw_svg`, exactly like the `print_latex` function in C++
  CuTe (needs only a LaTeX install for the PDF, no Python packages).
* [`draw_latex_tv(tv_layout, tile_mn)`](../pycute/util/draw_latex.py) —
  the LaTeX/PDF analogue of `draw_svg_tv`.
* [`draw_colors`](../pycute/util/draw_colors.py) — a catalog of `color`
  functors (returning `(r, g, b)` tuples) for the drawers above: the defaults
  `index_grey_8x` / `thread_color_8x`, plus `bank_color_8x`/`16x`/`32x` (bank
  conflicts), `value_color_8x`, `warp_color_8x`, `white`, and `constant(rgb)`.
  Pass any functor of the matching signature to recolor.

See [Visualization](./07_visualization.md) for examples and the relevant
caveats.

## Copyright

Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
