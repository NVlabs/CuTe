# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Algorithms expressed over PyCuTe Tensors (Whitepaper §2.6).

These are host-side, element-by-element implementations meant for learning
and prototyping — not device kernels.

* `pycute.alg.copy` — optimized COPY (may fall back to the reference)
* `pycute.alg.ref.copy` — reference / naive COPY
"""

from .copy import copy

__all__ = ["copy"]
