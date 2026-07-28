# PyCuTe Documentation

PyCuTe is a Python implementation of CuTe, the layout-and-tensor abstraction at
the heart of [CUTLASS 3.x](https://github.com/NVIDIA/cutlass) and the
[CuTe DSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl.html).
PyCuTe is intended as the **reference implementation** of the CuTe project: a
small, readable Python library that mirrors the algebra described in the
[CuTe Whitepaper](https://arxiv.org/abs/2603.02298). It is ideal for learning,
experimenting with, and prototyping the layout algebra without the complexity of
CUDA C++ or device codegen.

This documentation is organized in three layers:

1. **A gentle introduction** to CuTe's core ideas, building up from
   hierarchical tuples to layouts to the layout algebra.
2. **A practical reference** for `pycute`'s Python API, with examples drawn
   directly from the unit tests.
3. **Pointers to the Whitepaper** wherever a topic deserves the formal
   treatment that prose cannot give.

The [CuTe Whitepaper](https://arxiv.org/abs/2603.02298) is the primary,
authoritative reference. This documentation defers to it on every formal
definition and post-condition; if the Whitepaper and this documentation
disagree, the Whitepaper wins.

## Table of contents

| File | Topic |
|---|---|
| [`00_quickstart.md`](./00_quickstart.md) | Installation, a 60-second tour, and where to look next |
| [`01_htuple.md`](./01_htuple.md) | `HTuple`s: hierarchical tuples and the toolbox in `htuple.py` |
| [`02_shape_stride.md`](./02_shape_stride.md) | `Shape`, `Stride`, coordinate conversion, and integer-modules (`int`, `ArithTuple`, `F2`) |
| [`03_layout.md`](./03_layout.md) | `Layout`: construction, evaluation, the three coordinate systems, slicing |
| [`04_layout_algebra.md`](./04_layout_algebra.md) | `coalesce`, `composition`, `complement`, `logical_divide`, `logical_product`, `layout_add`, `greatest_common_domain`, inverses, `nullspace`, `recast` |
| [`05_tensor.md`](./05_tensor.md) | `Tensor` and `Accessor`: layouts plus data |
| [`06_swizzle.md`](./06_swizzle.md) | `Swizzle` and `F2`-stride layouts |
| [`07_visualization.md`](./07_visualization.md) | `print_tensor`, `draw_svg`, `draw_svg_tv`, `draw_latex`, `draw_latex_tv`, and `draw_colors` functors |
| [`08_api_reference.md`](./08_api_reference.md) | Index of names in `pycute.__all__`, with examples and test links |

## Quick reference: what is a Layout?

The single sentence of CuTe is:

> A `Layout` is a function from coordinates to offsets, defined by a `Shape`
> and a `Stride` of the same hierarchical profile.

In PyCuTe:

```python
>>> from pycute import *
>>> A = Layout((3, 4), (4, 1))   # 3x4 row-major matrix
>>> A(2, 3)                      # call the layout on a coordinate
11
>>> A(11)                        # 1-D coordinate is also accepted
11
>>> size(A)                      # number of valid coordinates
12
>>> rank(A)                      # number of top-level modes
2
```

Layouts can be combined with the layout algebra:

```python
>>> coalesce(Layout((2, (1, 6)), (1, (6, 2))))
Layout(12, 1)
>>> composition(Layout(12), Layout((4, 3)))
Layout((4, 3), (1, 4))
>>> logical_divide(Layout(24), Layout(4, 2))
Layout((4, (2, 3)), (2, (1, 8)))
```

That short story is the whole of CuTe — every other concept in this
documentation is a refinement, generalization, or application of it.

## How to read this documentation

If you have never seen CuTe before, read the documents in order. The
[Quickstart](./00_quickstart.md), [HTuples](./01_htuple.md), and
[Shapes/Strides](./02_shape_stride.md) chapters establish the language we
need before we can talk about layouts at all.

If you are already familiar with CuTe in C++ or the CuTe DSL, skim the
[Layout](./03_layout.md) and [Layout Algebra](./04_layout_algebra.md) chapters
for the PyCuTe naming conventions and any places where PyCuTe differs from
those implementations (notably, PyCuTe follows the Whitepaper's
treatment of *coordinate strides* and *F2 strides*, which are not first-class
in C++ CuTe).

If you only need a function signature or a worked example, jump to the
[API reference](./08_api_reference.md), which links each exported name to
its unit test where one exists.

## How PyCuTe relates to the C++ and DSL versions

* **The Whitepaper** is the most up-to-date theory and includes
  [coordinate strides](./02_shape_stride.md#coordinate-strides-arithtuple-scaledbasis-and-e)
  and [F2 strides](./06_swizzle.md). PyCuTe implements the Whitepaper's
  hierarchical shapes/strides, layout algebra, coordinate strides, F2 strides,
  and a reference `Tensor` layer; see the [API reference](./08_api_reference.md)
  for the exact exported surface.
* **The [C++ CuTe documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cpp/cute)**
  predates the Whitepaper. Its overall pedagogy is excellent and we draw on it
  here, but a few of its claims (e.g. that strides are integers) have been
  generalized in the Whitepaper. We flag those points where they come up.
* **The [Python CuTe DSL documentation](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl.html)**
  focuses on JIT compilation, kernel authoring, and integration with PyTorch.
  PyCuTe is deliberately a layer below the DSL: it is plain Python, with no
  JIT, no kernel launching, and no GPU dependency.

## Conventions used throughout

* Examples are written for an interpreter prompt, prefixed with `>>>`. Any
  block that does not show a prompt is illustrative pseudo-code, not a
  literal transcript.
* Whitepaper references look like *(Whitepaper, §3.2)* and refer to sections of
  the [CuTe Whitepaper on arXiv](https://arxiv.org/abs/2603.02298).
* Test references use full pytest node IDs where possible, e.g.
  [`test_layout.py::TestLayout`](../test/test_layout.py).
* Source references look like [`layout.py`](../pycute/layout.py) and link
  to the corresponding module in [`pycute/`](../pycute/).
* In CuTe, the word *mode* means "one entry of a hierarchical tuple, possibly
  itself a tuple". This is the same word used in the C++ docs and the
  Whitepaper.

## Copyright

Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
