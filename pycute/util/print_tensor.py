# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for printing CuTe Layouts and Tensors as ASCII tables
"""

from typing import Union

from pycute import *


def print_tensor(tensor : Union[Tensor, LayoutBase], print_type=True):
  """
  Print a layout or tensor of rank 1 through 4 as nested ASCII tables.

  A `Layout` is rendered through an `ImplicitAccessor`, so each cell shows its
  offset; a `Tensor` shows the element stored there. Ranks 3 and 4 are printed
  as a series of rank-2 slices. `print_type` prepends the `Shape:Stride` header.

  Pre-conditions:
    rank(tensor) <= 4; otherwise a ValueError is raised
  """
  if (print_type):
    print(tensor)

  if isinstance(tensor, LayoutBase):
    return print_tensor(Tensor(ImplicitAccessor(0), tensor), print_type=False)

  match rank(tensor):
    case 1:
      for i in range(size(tensor)):
        print(f"{tensor[i]:<5}", end=" ")
      print()
    case 2:
      for m in range(size[0](tensor)):
        for n in range(size[1](tensor)):
          print(f"{tensor[m,n]:<5}", end=" ")
        print()
    case 3:
      for k in range(size[2](tensor)):
        if k > 0: print("-" * 2 * size[1](tensor) + f"--  k = {k:<2} --" + "-" * 3 * size[1](tensor))
        print_tensor(tensor[None, None, k], print_type=False)
    case 4:
      for p in range(size[3](tensor)):
        if p > 0: print("=" * 2 * size[1](tensor) + f"==  p = {p:<2} ==" + "=" * 3 * size[1](tensor))
        print_tensor(tensor[None, None, None, p], print_type=False)
    case _:
      raise ValueError(f"Expected rank <= 4 and found {tensor}")
