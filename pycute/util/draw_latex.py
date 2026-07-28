# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for LaTeX/PDF generation of CuTe Layouts
"""

from pycute import *

from .draw_colors import index_grey_8x, thread_color_8x


_latex_header = (
  "\\documentclass{standalone}\n"
  "\\usepackage{tikz}\n\n"
  "\\begin{document}\n"
  "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},"
  "every node/.style={minimum size=1cm, outer sep=0pt}]\n\n"
)

_latex_footer = (
  "\\end{tikzpicture}\n"
  "\\end{document}\n"
)


def _tikz_rgb(rgb):
  r, g, b = rgb
  return f"{{rgb,255:red,{r};green,{g};blue,{b}}}"


def _grid_and_labels(M, N):
  s = f"\n\\draw[color=black,thick,shift={{(-0.5,-0.5)}}] (0,0) grid ({M},{N});\n\n"
  for m in range(M):
    s += f"\\node at ({m},{-1}) {{\\Large{{\\texttt{{{m}}}}}}};\n"
  for n in range(N):
    s += f"\\node at ({-1},{n}) {{\\Large{{\\texttt{{{n}}}}}}};\n"
  return s


def _write_and_compile(body, filename, compile_pdf):
  import os
  import shutil
  import subprocess

  doc = _latex_header + body + _latex_footer
  with open(filename, "w") as f:
    f.write(doc)
  print(f"Saved as {filename}")

  if not compile_pdf:
    return

  pdflatex = shutil.which("pdflatex")
  if pdflatex is None:
    print(f"  (pdflatex not found on PATH; wrote .tex only. Install a LaTeX "
          f"distribution, or compile manually with `pdflatex {filename}`.)")
    return

  stem = os.path.splitext(filename)[0]
  out_dir = os.path.dirname(os.path.abspath(filename))
  try:
    subprocess.run(
      [pdflatex, "-interaction=nonstopmode", "-halt-on-error",
       "-output-directory", out_dir, filename],
      check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
  except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
    print(f"  (pdflatex failed; wrote .tex only. See {stem}.log for details.)")
    return

  for ext in (".aux", ".log"):
    aux = stem + ext
    if os.path.exists(aux):
      os.remove(aux)
  print(f"Saved as {stem}.pdf")


def draw_latex(layout : LayoutBase, filename="layout.tex", compile_pdf=True, color=index_grey_8x):
  if rank(layout) != 2:
    raise ValueError(f"Expected a rank-2 Layout")

  M, N = size[0](layout), size[1](layout)

  body = f"% Layout: {layout}\n"
  for i in range(M):
    for j in range(N):
      idx = int(layout(i, j))
      body += (f"\\node[fill={_tikz_rgb(color(idx))}] "
               f"at ({i},{j}) {{{idx}}};\n")
  body += _grid_and_labels(M, N)

  _write_and_compile(body, filename, compile_pdf)


def draw_latex_tv(layout : LayoutBase, tile_mn=None, filename="tvlayout.tex", compile_pdf=True, color=thread_color_8x):
  if rank(layout) != 2:
    raise ValueError(f"Expected a rank-2 TV Layout")

  tile_mn = coshape(layout) if tile_mn is None else tile_mn
  if rank(tile_mn) != 2:
    raise ValueError(f"Expected a rank-2 MN Tile")

  if congruent(coprofile(layout), 0):
    layout = composition(tile_mn, layout)
  if not congruent(coprofile(layout), (0,0)):
    raise ValueError(f"Expected a 2D codomain (tid,vid) -> (m,n)")

  M, N = size[0](tile_mn), size[1](tile_mn)
  filled = [[False for n in range(N)] for m in range(M)]

  body = f"% Layout TV: {layout}\n"
  for tid in range(size[0](layout)):
    for vid in range(size[1](layout)):
      i, j = layout(tid, vid)
      if filled[i][j]:
        continue
      filled[i][j] = True
      body += (f"\\node[fill={_tikz_rgb(color(tid, vid))}] "
               f"at ({i},{j}) {{\\shortstack{{T{tid} \\\\ V{vid}}}}};\n")
  body += _grid_and_labels(M, N)

  _write_and_compile(body, filename, compile_pdf)
