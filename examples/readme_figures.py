# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regenerate the SVG figures embedded in the top-level README.

Run from the repository root (needs the ``viz`` extra for ``svgwrite``):

    python -m examples.readme_figures

Every figure is a plain PyCuTe ``Layout`` drawn with ``pycute.util`` -- the
same one-liners a reader can paste into an interpreter. Output is written to
``docs/images/``.
"""

import os

from pycute import *
from pycute.util import draw_svg, draw_svg_tv, bank_color_8x

# docs/images/, resolved relative to this file so the script works from anywhere.
IMAGES_DIR = os.path.normpath(
  os.path.join(os.path.dirname(__file__), os.pardir, "docs", "images"))


def main():
  os.makedirs(IMAGES_DIR, exist_ok=True)
  path = lambda name: os.path.join(IMAGES_DIR, name)

  # Same 8x8 shapes with different strides.
  draw_svg(Layout((8, 8), (8, 1)), path("layout_row_major.svg"), bank_color_8x)
  draw_svg(Layout((8, 8), (1, 8)), path("layout_col_major.svg"), bank_color_8x)
  draw_svg(Layout(((4,2), (2,4)), ((1,32), (4,8))), path("layout_blocked.svg"), bank_color_8x)

  # Thread-value layout of an SM80 16x8 MMA C-tile: (tid, vid) -> offset in a
  # (16, 8) tile. draw_svg_tv folds the linear offset into the tile and labels
  # each cell with the (thread, value) that owns it.
  SM80 = Layout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
  draw_svg_tv(SM80, tile_mn=(16, 8), filename=path("tv_sm80_16x8.svg"))

  # Shared-memory bank coloring (bank = offset % 8).
  # The F2(9)/F2(1) strides XOR the row/col offset so that
  # every row/col spans all 8 banks.
  draw_svg(Layout((8, 8), (F2(1), F2(9))), path("smem_swizzled_19.svg"), color=bank_color_8x)
  draw_svg(Layout((8, 8), (F2(9), F2(1))), path("smem_swizzled_91.svg"), color=bank_color_8x)


if __name__ == "__main__":
  main()
