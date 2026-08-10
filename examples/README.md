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

## `einfold.py`

Rearranges modes by naming them: `einfold("in_modes->out_modes", value)` is a
generalized transpose that permutes, groups, ungroups, repeats and drops the
modes of a `Layout`, a `Tensor` or an `HTuple`. Each side of the expression
denotes a profile whose leaves are one-character names — as in `einsum`'s
subscripts, separators between modes are implied — so a group descends into a
nested source mode on the left and builds a new one on the right. Top-level
modes the input leaves unnamed are appended unchanged, so an expression only
names the modes it rearranges.

`Layout`s produce `Layout`s and `Tensor`s produce `Tensor`s over the source's
accessor, so no data is copied. Anything else is matched against its
[`profile`](../docs/01_htuple.md#profiles-congruence-and-weak-congruence) and
rebuilt as tuples, which asks nothing of its leaves — so a Shape or a Stride
folds exactly as the Layout it was read from.

```python
from pycute import Layout, make_tensor, shape
from examples.einfold import einfold

A = make_tensor(Layout(((12, 4), 42, 5, 7)))
shape(einfold("ijkm -> imkj", A))        # ((12, 4), 7, 5, 42)  -- permute
shape(einfold("(ab)cde -> c(ade)b", A))  # (42, (12, 5, 7), 4)  -- split and regroup
shape(einfold("ij -> (ij)", A))          # (((12, 4), 42), 5, 7)  -- 5, 7 pass through
shape(einfold("ijk -> ik", A))           # ((12, 4), 5, 7)  -- drop mode j

einfold("ijkm -> imkj", shape(A))        # folds the Shape on its own
```

The companion `einfold_test.py` is one series of such examples, each checked as
a Layout, as a Tensor and as the source's Shape and Stride:

```sh
pytest examples/einfold_test.py
```

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
