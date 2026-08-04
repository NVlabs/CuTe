# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
COPY — element-wise transfer between tensors of equal size (Whitepaper §2.6.1).

A rank-1 algorithm: iterate the flat index space and remap through each
tensor's layout. Different `(src, dst)` layout pairs realize memcpy, gather,
scatter, broadcast, transpose, and related transfers.
"""

from __future__ import annotations

from pycute import Tensor, size


def copy(src: Tensor, dst: Tensor) -> None:
  """
  Reference COPY: `dst[i] = src[i]` for every flat index `i`.
  """
  if size(src) != size(dst):
    raise ValueError(f"copy: size mismatch {size(src)} != {size(dst)}")

  for i in range(size(dst)):
    dst[i] = src[i]
