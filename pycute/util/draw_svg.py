# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for SVG generation of CuTe Layouts
"""

from typing import Union

from pycute import *

from .draw_colors import thread_color_8x, white


def _draw_index_labels(dwg, M, N, cell_size, margin):
  # Throughout, `text-anchor=middle` + `dy=0.35em` centers a single line of
  # text on its point in a renderer-portable way: it relies only on baseline
  # positioning (universally supported), not on `dominant-baseline`, which
  # several SVG renderers ignore on <text> elements.
  # Row indices, down the left margin
  for m in range(M):
    dwg.add(dwg.text(
      str(m), insert=(margin//2, margin + m*cell_size + cell_size//2),
      dy=["0.35em"], text_anchor="middle", font_size="8px", font_family="monospace"))
  # Column indices, across the top margin
  for n in range(N):
    dwg.add(dwg.text(
      str(n), insert=(margin + n*cell_size + cell_size//2, margin//2),
      dy=["0.35em"], text_anchor="middle", font_size="8px", font_family="monospace"))


def draw_svg(tensor : Union[Tensor, LayoutBase], filename="layout.svg", color=white):
  """
  Save a rank-2 `Tensor` or `Layout` as an SVG table with row/column labels.

  A `Layout` labels each cell with its offset, a `Tensor` with the element
  stored there. A rank-1 input is drawn as a single row. `color` is a functor
  `color(idx) -> (r, g, b)` keyed on the cell's offset; `draw_colors` holds a
  catalog of them.

  Requires the optional `svgwrite` package (`pip install pycute[viz]`).

  Pre-conditions:
    rank(tensor) is 1 or 2; otherwise a ValueError is raised
  """
  try:
    import svgwrite
  except ImportError as e:
    raise ImportError(
      "draw_svg requires the optional 'svgwrite' package. "
      "Install it with `pip install svgwrite` or `pip install pycute[viz]`."
    ) from e

  if isinstance(tensor, LayoutBase):
    tensor = Tensor(ImplicitAccessor(0), tensor)
  if rank(tensor) == 1:
    tensor = Tensor(tensor.accessor, make_layout([Layout(1,0), tensor.layout]))
  if rank(tensor) != 2:
    raise ValueError(f"Expected a rank-2 Layout, got {tensor.layout}")

  # Cell size in pixels, with a one-cell margin for the index labels
  cell_size = 20
  margin = cell_size

  # Grid size
  M, N = size[0](tensor), size[1](tensor)

  # Create SVG canvas
  dwg = svgwrite.Drawing(filename, size=(margin + N*cell_size, margin + M*cell_size))

  # Draw grid cells
  for i in range(M):
    for j in range(N):
      # Label by the value, color by the offset
      value  = tensor[i, j]
      offset = tensor.layout(i, j)
      x = margin + j * cell_size
      y = margin + i * cell_size

      # Draw rectangle
      dwg.add(dwg.rect(
        insert=(x, y), size=(cell_size, cell_size),
        fill=svgwrite.rgb(*color(offset), mode='RGB'), stroke='black'))

      # Add label text
      dwg.add(dwg.text(
        str(value), insert=(x + cell_size//2, y + cell_size//2),
        dy=["0.35em"], text_anchor="middle", font_size="10px", font_family="Arial"))

  _draw_index_labels(dwg, M, N, cell_size, margin)

  dwg.save()
  print(f"Saved as {filename}")


def draw_svg_tv(layout : LayoutBase, tile_mn=None, filename="tvlayout.svg", color=thread_color_8x):
  """
  Save a rank-2 thread-value layout as a colored SVG table with `T`/`V` labels.

  The layout maps `(tid, vid)` to a position in the tile. Its codomain may be
  2-D `(m, n)` coordinates, or a linear offset -- in which case a rank-2
  `tile_mn` is required and the offsets are folded into the tile via
  `composition(tile_mn, layout)`. `color` is a functor
  `color(tid, vid) -> (r, g, b)`; where several `(tid, vid)` pairs land on one
  cell, the first drawn wins.

  Requires the optional `svgwrite` package (`pip install pycute[viz]`).

  Pre-conditions:
    rank(layout) == 2 and rank(tile_mn) == 2
    the codomain is 2-D once folded; otherwise a ValueError is raised
  """
  try:
    import svgwrite
  except ImportError as e:
    raise ImportError(
      "draw_svg_tv requires the optional 'svgwrite' package. "
      "Install it with `pip install svgwrite` or `pip install pycute[viz]`."
    ) from e

  if rank(layout) != 2:
    raise ValueError(f"Expected a rank-2 TV Layout")

  tile_mn = coshape(layout) if tile_mn is None else tile_mn
  if rank(tile_mn) != 2:
    raise ValueError(f"Expected a rank-2 MN Tile")

  if congruent(coprofile(layout), 0):
    layout = composition(tile_mn, layout)
  if not congruent(coprofile(layout), (0,0)):
    raise ValueError(f"Expected a 2D codomain (tid,vid) -> (m,n)")

  # Cell size in pixels, with a one-cell margin for the index labels
  cell_size = 20
  margin = cell_size

  # Grid size
  M, N = size[0](tile_mn), size[1](tile_mn)
  filled = [[False for n in range(N)] for m in range(M)]

  # Create SVG canvas
  dwg = svgwrite.Drawing(filename, size=(margin + N*cell_size, margin + M*cell_size))

  # Fill in grid (empty cells are white)
  for i in range(M):
    for j in range(N):
      dwg.add(dwg.rect(
        insert=(margin + j*cell_size, margin + i*cell_size), size=(cell_size, cell_size),
        fill='white', stroke='black'))

  # Draw TV cells
  for tid in range(size[0](layout)):
    for vid in range(size[1](layout)):
      i, j = layout(tid, vid)
      if filled[i][j]:
        continue
      filled[i][j] = True

      x = margin + j * cell_size
      y = margin + i * cell_size

      # Draw rectangle
      dwg.add(dwg.rect(
        insert=(x, y), size=(cell_size, cell_size),
        fill=svgwrite.rgb(*color(tid, vid), mode='RGB'), stroke='black'))

      # Add label text
      dwg.add(dwg.text(
        f"T{tid}", insert=(x + cell_size//2, y + 1*cell_size//4),
        dy=["0.35em"], text_anchor="middle", font_size="8px", font_family="Arial"))
      dwg.add(dwg.text(
        f"V{vid}", insert=(x + cell_size//2, y + 3*cell_size//4),
        dy=["0.35em"], text_anchor="middle", font_size="8px", font_family="Arial"))

  _draw_index_labels(dwg, M, N, cell_size, margin)

  dwg.save()
  print(f"Saved as {filename}")
