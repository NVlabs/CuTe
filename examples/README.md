# PyCuTe examples

Small examples that exercise PyCuTe end-to-end. Run commands below from the
repository root so that the `examples` package and local `pycute` sources are on
the import path. A virtual environment is recommended; see
[Installation](../README.md#installation).

## `readme_figures.py`

Regenerates the SVG figures embedded in the [top-level README](../README.md):
row-major, column-major, and blocked layouts; an SM80 `16×8` thread-value
layout; and two shared-memory layouts with `F2` (XOR) swizzled strides.

Install the visualization dependencies, then run:

```sh
pip install -e ".[viz]"
python3 -m examples.readme_figures
```

The script writes SVGs to [`docs/images/`](../docs/images/) regardless of the
current working directory. See
[Visualization](../docs/07_visualization.md) for the drawing APIs.

## `einsum.py`

Implements a minimal binary `einsum` by folding a tensor contraction into the
reference `batch_gemm` primitive. Given
`einsum("a_modes,b_modes->c_modes", A, B, C)`, labels are classified as row
(`M`), column (`N`), reduction (`K`), or batch (`L`) modes, then viewed as the
canonical `A:(M,K,L)`, `B:(N,K,L)`, and `C:(M,N,L)` layouts.

`C` is caller-provided and the result accumulates into it. Only explicit
subscripts (`->`) are supported; labels may appear at most once per operand,
there is no broadcasting, and each label must occur in at least two operands.

```python
from pycute import Layout, make_tensor
from examples.einsum import einsum

A = make_tensor(Layout((3, 5)))  # m,k
B = make_tensor(Layout((4, 5)))  # n,k
C = make_tensor(Layout((3, 4)))  # m,n; initially zero
einsum("mk,nk->mn", A, B, C)
```

The companion `einsum_test.py` uses a dependency-free brute-force oracle:

```sh
pytest examples/einsum_test.py
```

## `algorithms/copy.ipynb`

Walkthrough of the COPY algorithm (Whitepaper §2.6.1): applications that are
all `copy(src, dst)` with different layouts (memcpy, gather/scatter, broadcast,
transpose, …), then the layout analysis `pycute.alg.copy` uses to reshape that
loop — common domain, nullspace, and alignment — against the element-at-a-time
`pycute.alg.ref.copy`.

```sh
pip install -e ".[viz]"   # optional inline SVG layout figures
jupyter notebook examples/algorithms/copy.ipynb
```

Unlike the scripts above, the notebook does not need to run from the repository
root: when `pycute` is not importable it walks up from the kernel's working
directory to the checkout and puts that on `sys.path`, so a bare
`jupyter notebook` works without installing anything.

Unit coverage for both loops lives in `test/test_alg_copy.py`.

## Adding an example

Keep each example self-contained, import from `pycute` (and `pycute.util` for
visualization), and add a short section here describing how to run it and what
it produces.
