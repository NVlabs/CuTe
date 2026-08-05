# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pycute.alg.copy and pycute.alg.ref.copy"""

import inspect
import json
import unittest
from pathlib import Path

from pycute import *
from pycute.alg import copy as copy_opt
from pycute.alg.ref import copy as copy_ref

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "algorithms" / "copy.ipynb"


def _fill_iota(t: Tensor) -> None:
  for i in range(size(t)):
    t[i] = float(i)


def _flat_values(t: Tensor) -> list:
  return [t[i] for i in range(size(t))]


class _LoggingPtr(MutableAccessor):
  """An accessor that records the absolute offset of every store."""

  def __init__(self, inner, log=None, base=0):
    self.inner = inner
    self.log = [] if log is None else log
    self.base = base

  def __getitem__(self, idx):
    return self.inner[idx]

  def __setitem__(self, idx, value):
    self.log.append(self.base + int(idx))
    self.inner[idx] = value

  def __add__(self, idx):
    return _LoggingPtr(self.inner + idx, self.log, self.base + int(idx))


def _vector_slices(src_layout, dst_layout) -> tuple[Layout, Layout]:
  """The (src, dst) layouts of the innermost run `copy` hands to `_memcpy`."""
  src, dst = make_tensor(src_layout), make_tensor(dst_layout)
  common = greatest_common_domain(src, dst)
  src_c, dst_c = logical_divide(src, common), logical_divide(dst, common)

  null_dst = nullspace(dst_c.layout[0])
  src_n = logical_divide[0](src_c, null_dst)[(0, None), None]
  dst_n = logical_divide[0](dst_c, null_dst)[(0, None), None]

  # Align both to destination-memory order, then take the common contiguous run
  inv_dst = right_inverse(dst_n.layout[0])
  src_v = logical_divide[0](src_n, inv_dst)
  dst_v = logical_divide[0](dst_n, inv_dst)

  run = coalesce(src_v.layout[0][0])[0]
  vec = size(run) if stride(run) == 1 else 1

  src_v = logical_divide(src_v, vec)
  dst_v = logical_divide(dst_v, vec)
  return src_v[:, 0].layout, dst_v[:, 0].layout


def _writes(copy_fn, src_layout, dst_layout) -> tuple[list, list]:
  """Run `copy_fn` and return (destination offsets written, destination values)."""
  src = make_tensor(src_layout)
  _fill_iota(src)
  dst = Tensor(_LoggingPtr(Array(int(coshape(dst_layout)))), dst_layout)
  copy_fn(src, dst)
  return dst.accessor.log, _flat_values(dst)


# The applications of Table 2 (Whitepaper §2.6.1), as walked through in
# examples/algorithms/copy.ipynb.
APPLICATIONS = [
  ("memcpy",       Layout(8, 1),                       Layout(8, 1)),
  ("transpose",    Layout((4, 8), (8, 1)),             Layout((4, 8), (1, 4))),
  ("gather",       Layout((2, 3), (4, 1)),             Layout(6, 1)),
  ("scatter",      Layout(6, 1),                       Layout((2, 3), (4, 1))),
  ("broadcast",    Layout(7, 0),                       Layout(7, 1)),
  ("constant",     Layout(7, 0),                       Layout(7, 0)),
  ("hierarchical", Layout((4, (2, 3)), (1, (4, 12))),  Layout((4, 6), (1, 4))),
  ("subdomain",    Layout((4, 3, 5), (1, 7, 42)),      Layout((6, 10), (1, 9))),
  ("coprime",      Layout((5, 7), (7, 1)),             Layout((7, 5), (5, 1))),
  ("rank1 to 2",   Layout(12, 1),                      Layout((3, 4), (4, 1))),
  ("part. bcast",  Layout((5, 4), (0, 1)),             Layout((5, 4), (4, 1))),
]


class TestCopyRef(unittest.TestCase):
  def test_memcpy(self):
    src = make_tensor(Layout(8, 1))
    dst = make_tensor(Layout(8, 1))
    _fill_iota(src)
    copy_ref(src, dst)
    self.assertEqual(_flat_values(src), _flat_values(dst))

  def test_transpose(self):
    src = make_tensor(Layout((4, 8), (8, 1)))
    dst = make_tensor(Layout((4, 8), (1, 4)))
    _fill_iota(src)
    copy_ref(src, dst)
    for i in range(size(src)):
      self.assertEqual(dst[i], src[i])
    # Physical order differs: row-major vs col-major storage
    self.assertNotEqual(
      [src.accessor[i] for i in range(coshape(src.layout))],
      [dst.accessor[i] for i in range(coshape(dst.layout))],
    )

  def test_gather_scatter(self):
    src = make_tensor(Layout((2, 3), (4, 1)))
    dst = make_tensor(Layout(6, 1))
    _fill_iota(src)
    copy_ref(src, dst)
    self.assertEqual(_flat_values(src), _flat_values(dst))

    src2 = make_tensor(Layout(6, 1))
    dst2 = make_tensor(Layout((2, 3), (4, 1)))
    _fill_iota(src2)
    copy_ref(src2, dst2)
    self.assertEqual(_flat_values(src2), _flat_values(dst2))

  def test_broadcast(self):
    src = make_tensor(Layout(7, 0))
    src[0] = 42.0
    dst = make_tensor(Layout(7, 1))
    copy_ref(src, dst)
    self.assertEqual(_flat_values(dst), [42.0] * 7)

  def test_size_mismatch(self):
    src = make_tensor(Layout(4, 1))
    dst = make_tensor(Layout(8, 1))
    with self.assertRaises(ValueError):
      copy_ref(src, dst)


class TestCopyOpt(unittest.TestCase):
  """The reference implementation is the oracle: whatever the optimized copy
  reshapes, it must land the same elements in the same places."""

  def test_matches_reference_on_every_application(self):
    for label, src_layout, dst_layout in APPLICATIONS:
      with self.subTest(label):
        src = make_tensor(src_layout)
        _fill_iota(src)
        dst_ref = make_tensor(dst_layout)
        dst_opt = make_tensor(dst_layout)
        copy_ref(src, dst_ref)
        copy_opt(src, dst_opt)
        # Equal through the layout, and byte-for-byte in memory: the optimized
        # path must not disturb offsets the layout never reaches.
        self.assertEqual(_flat_values(dst_ref), _flat_values(dst_opt))
        self.assertEqual(
          [dst_ref.accessor[i] for i in range(coshape(dst_layout))],
          [dst_opt.accessor[i] for i in range(coshape(dst_layout))],
        )

  def test_size_mismatch(self):
    src = make_tensor(Layout(4, 1))
    dst = make_tensor(Layout(8, 1))
    with self.assertRaises(ValueError):
      copy_opt(src, dst)

  def test_constant_fill_drops_redundant_writes(self):
    """`dst 7:0` sends every coordinate to one address; a constant source
    writes the same value there 7 times, so 6 of the writes are dropped."""
    layouts = (Layout(7, 0), Layout(7, 0))
    self.assertEqual(nullspace(layouts[1]), Layout(7, 1))   # the whole domain

    ref_offsets, ref_values = _writes(copy_ref, *layouts)
    opt_offsets, opt_values = _writes(copy_opt, *layouts)

    self.assertEqual(ref_offsets, [0] * 7)
    self.assertEqual(opt_offsets, [0])
    self.assertEqual(ref_values, opt_values)

  def test_write_after_write_is_rejected(self):
    """Distinct sources landing on one address is order-dependent, so the
    optimized copy refuses it where the reference silently races."""
    src = make_tensor(Layout(4, 1))
    _fill_iota(src)

    with self.assertRaisesRegex(ValueError, "[Ww]rite-after-write"):
      copy_opt(src, make_tensor(Layout(4, 0)))

    # The reference has no such check: it just leaves the last write standing.
    dst = make_tensor(Layout(4, 0))
    copy_ref(src, dst)
    self.assertEqual(dst[0], 3.0)

  def test_vector_is_contiguous_in_both_or_scalar(self):
    """The innermost run handed to `_memcpy` is `V:1` in *both* tensors, so
    the array copy only has to check alignment and hardware support. When no
    common contiguous run exists, `V` is 1 and the dispatch is scalar."""
    for label, src_layout, dst_layout, expect in [
      ("memcpy",       Layout(8, 1),                      Layout(8, 1),           8),
      ("hierarchical", Layout((4, (2, 3)), (1, (4, 12))),  Layout((4, 6), (1, 4)), 8),
      ("part. bcast",  Layout((5, 4), (0, 1)),             Layout((5, 4), (4, 1)), 4),
      ("subdomain",    Layout((4, 3, 5), (1, 7, 42)),      Layout((6, 10), (1, 9)), 2),
      ("transpose",    Layout((4, 8), (8, 1)),             Layout((4, 8), (1, 4)), 1),
      ("scatter",      Layout(6, 1),                       Layout((2, 3), (4, 1)), 1),
      ("broadcast",    Layout(7, 0),                       Layout(7, 1),           1),
    ]:
      with self.subTest(label):
        src, dst = _vector_slices(src_layout, dst_layout)
        self.assertEqual(size(src), expect)
        self.assertEqual(size(dst), expect)
        if expect > 1:
          # A real vector: the same contiguous run on both sides
          self.assertEqual(coalesce(src), coalesce(dst))
          self.assertEqual(stride(coalesce(src)), 1)

  def test_destination_is_written_in_ascending_runs(self):
    """Dividing by `right_inverse(dst)` walks the domain in destination-memory
    order, so a scattered destination is written as ascending runs."""
    src_layout, dst_layout = Layout(6, 1), Layout((2, 3), (4, 1))

    ref_offsets, _ = _writes(copy_ref, src_layout, dst_layout)
    opt_offsets, _ = _writes(copy_opt, src_layout, dst_layout)

    self.assertEqual(ref_offsets, [0, 4, 1, 5, 2, 6])   # hops between the rows
    self.assertEqual(opt_offsets, [0, 1, 2, 4, 5, 6])   # two contiguous runs
    self.assertEqual(sorted(ref_offsets), opt_offsets)  # same addresses either way

    # Offset 3 is outside the layout's image and must stay untouched
    self.assertNotIn(3, opt_offsets)



  def test_incompatible_shapes_fall_back_to_elementwise(self):
    """`(5,7)` and `(7,5)` share no aligned factor, so there is nothing to
    tile and the optimized copy degrades to the reference's element-at-a-time."""
    src_layout, dst_layout = Layout((5, 7), (7, 1)), Layout((7, 5), (5, 1))
    self.assertEqual(greatest_common_domain(src_layout, dst_layout), Layout((1,), (0,)))

    ref_offsets, ref_values = _writes(copy_ref, src_layout, dst_layout)
    opt_offsets, opt_values = _writes(copy_opt, src_layout, dst_layout)

    self.assertEqual(opt_offsets, ref_offsets)          # no reordering available
    self.assertEqual(len(opt_offsets), 35)              # and no writes saved
    self.assertEqual(ref_values, opt_values)


class TestCopyNotebook(unittest.TestCase):
  """`examples/algorithms/copy.ipynb` prints `copy`'s source with
  `inspect.getsource`, so its *stored* output is a verbatim second copy of this
  module that goes stale the moment `copy` is edited."""

  def test_stored_source_output_is_current(self):
    if not NOTEBOOK.exists():
      self.skipTest(f"{NOTEBOOK} not present")
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    printers = [
      c for c in cells
      if c["cell_type"] == "code" and "inspect.getsource(copy)" in "".join(c["source"])
    ]
    self.assertEqual(len(printers), 1, "expected exactly one cell printing copy's source")

    stored = "".join(
      "".join(o.get("text", []))
      for o in printers[0]["outputs"] if o.get("output_type") == "stream"
    )
    self.assertEqual(stored.strip(), inspect.getsource(copy_opt).strip(),
                     "re-run the notebook cell that prints `copy`'s source")


if __name__ == "__main__":
  unittest.main()
