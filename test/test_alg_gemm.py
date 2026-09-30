# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for examples/algorithms/gemm.ipynb, the walkthrough of pycute.alg.ref.gemm"""

import inspect
import json
import unittest
from pathlib import Path

from pycute.alg.ref import gemm
from examples.einsum import _classify
from examples.im2col import OutOfBoundsAccessor

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "algorithms" / "gemm.ipynb"


class TestGemmNotebook(unittest.TestCase):
  """`examples/algorithms/gemm.ipynb` prints three sources with
  `inspect.getsource`, so each *stored* output is a verbatim second copy of that
  source that goes stale the moment it is edited."""

  def test_stored_source_outputs_are_current(self):
    if not NOTEBOOK.exists():
      self.skipTest(f"{NOTEBOOK} not present")
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]

    for name, obj, is_whole_output in [
        ("gemm",                            gemm,                            True),
        ("_classify",                       _classify,                       True),
        ("OutOfBoundsAccessor.__getitem__", OutOfBoundsAccessor.__getitem__, False),
    ]:
      call = f"inspect.getsource({name})"
      with self.subTest(call):
        printers = [c for c in cells
                    if c["cell_type"] == "code" and call in "".join(c["source"])]
        self.assertEqual(len(printers), 1, f"expected exactly one cell calling {call}")

        stored = "".join(
          "".join(o.get("text", []))
          for o in printers[0]["outputs"] if o.get("output_type") == "stream"
        )
        message = f"re-run the notebook cell that calls {call}"
        if is_whole_output:
          self.assertEqual(stored.strip(), inspect.getsource(obj).strip(), message)
        else:
          self.assertIn(inspect.getsource(obj), stored, message)


if __name__ == "__main__":
  unittest.main()
