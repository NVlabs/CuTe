# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for printing CuTe Layouts as bordered grids
"""

from typing import Union

from pycute import *

def print_table(tensor : Union[Tensor, LayoutBase], print_type=True):
  """
  Render a rank-2 layout or tensor as a single bordered grid, one cell per
  coordinate.

  Like `print_tensor`, a `Layout` is rendered through an `ImplicitAccessor` and
  a `Tensor` shows its elements, with `print_type` prepending the
  `Shape:Stride` header. A non-rank-2 input falls back to a plain `print`.

  Requires the optional `tabulate` package (`pip install pycute[viz]`).
  """
  if (print_type):
    print(tensor)

  if isinstance(tensor, LayoutBase):
    return print_table(Tensor(ImplicitAccessor(0), tensor), print_type=False)

  try:
    from tabulate import tabulate
  except ImportError as e:
    raise ImportError(
      "print_table requires the optional 'tabulate' package. "
      "Install it with `pip install tabulate` or `pip install pycute[viz]`."
    ) from e

  if rank(tensor) == 1:
    data = [[tensor[m] for m in range(size[0](tensor))]]
    print(tabulate(data, tablefmt="grid"))
  elif rank(tensor) == 2:
    data = [[tensor[m,n] for n in range(size[1](tensor))] for m in range(size[0](tensor))]
    print(tabulate(data, tablefmt="grid"))
  else:
    print(tensor)