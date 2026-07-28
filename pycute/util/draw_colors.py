# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Coloring functors shared by the CuTe Layout visualizers (draw_svg / draw_latex).

A `color` functor maps a cell key to an ``(r, g, b)`` tuple with each component
in ``[0, 255]``. Two key shapes are used by the drawers:

* offset functors take a single integer offset: ``color(idx) -> (r, g, b)``
* thread-value functors take a ``(tid, vid)`` pair: ``color(tid, vid) -> (r, g, b)``

This module provides a small catalog of common colorings:

* offset:        ``index_grey_8x`` (default), ``bank_color_8x/16x/32x``
* thread-value:  ``thread_color_8x`` (default), ``value_color_8x``, ``warp_color_8x``
* constant:      ``white``, and ``constant(rgb)`` to build your own

``index_grey_8x``/``thread_color_8x`` are the defaults used by
``draw_svg``/``draw_latex`` and ``draw_svg_tv``/``draw_latex_tv`` respectively;
pass any functor of the matching signature to override them.
"""

__all__ = [
  "index_grey_8x", "bank_color_8x", "bank_color_16x", "bank_color_32x",  # offset: color(idx)
  "thread_color_8x", "value_color_8x", "warp_color_8x",                  # tv:     color(tid, vid)
  "constant", "white",                                                   # constant / generic
]


# 8 RGB-255 greyscale colors, indexed mod 8
_greyscale_colors = [(255, 255, 255), (230, 230, 230), (205, 205, 205), (180, 180, 180),
                     (155, 155, 155), (130, 130, 130), (105, 105, 105), ( 80,  80,  80)]

# 8 RGB-255 light colors, indexed mod 8 (matches cute::print_latex)
_tv_colors = [(175, 175, 255), (175, 255, 175), (255, 255, 175), (255, 175, 175),
              (210, 210, 255), (210, 255, 210), (255, 255, 210), (255, 210, 210)]

# 32 RGB-255 light colors evenly spaced around the hue wheel (a "spectrum").
# Used for shared-memory bank colorings; the 16- and 8-entry variants are
# evenly-spaced subsamples, so they stay distinct and visually consistent.
_spectrum_32 = [
  (255, 128, 128), (255, 151, 128), (255, 175, 128), (255, 199, 128),
  (255, 223, 128), (255, 247, 128), (239, 255, 128), (215, 255, 128),
  (191, 255, 128), (167, 255, 128), (143, 255, 128), (128, 255, 135),
  (128, 255, 159), (128, 255, 183), (128, 255, 207), (128, 255, 231),
  (128, 255, 255), (128, 231, 255), (128, 207, 255), (128, 183, 255),
  (128, 159, 255), (128, 135, 255), (143, 128, 255), (167, 128, 255),
  (191, 128, 255), (215, 128, 255), (239, 128, 255), (255, 128, 247),
  (255, 128, 223), (255, 128, 199), (255, 128, 175), (255, 128, 151),
]
_spectrum_16 = _spectrum_32[::2]   # 16 evenly-spaced colors
_spectrum_8  = _spectrum_32[::4]   #  8 evenly-spaced colors


# ---- Offset colorings: color(idx) -> (r, g, b) ----

def index_grey_8x(idx):
  """Greyscale shade by ``idx % 8`` -> ``(r, g, b)``; default offset coloring."""
  return _greyscale_colors[idx % len(_greyscale_colors)]


def bank_color_8x(idx):
  """Color by ``idx % 8`` -> ``(r, g, b)`` from the light spectrum.

  Like ``bank_color_32x`` but cycling every 8 -- handy for 8-bank groupings
  or when fewer, more distinct colors read better.
  """
  return _spectrum_8[idx % len(_spectrum_8)]


def bank_color_16x(idx):
  """Color by ``idx % 16`` -> ``(r, g, b)`` from the light spectrum.

  Like ``bank_color_32x`` but cycling every 16.
  """
  return _spectrum_16[idx % len(_spectrum_16)]


def bank_color_32x(idx):
  """Color by shared-memory bank ``idx % 32`` -> ``(r, g, b)``.

  Spreads the 32 banks around a light spectrum so equal-bank cells share a
  color -- handy for spotting shared-memory bank conflicts.
  """
  return _spectrum_32[idx % len(_spectrum_32)]


# ---- Thread-value colorings: color(tid, vid) -> (r, g, b) ----

def thread_color_8x(tid, vid):
  """Color by ``tid % 8`` -> ``(r, g, b)`` (``vid`` ignored); default TV coloring."""
  return _tv_colors[tid % len(_tv_colors)]


def value_color_8x(tid, vid):
  """Color by value index ``vid % 8`` -> ``(r, g, b)`` (``tid`` ignored)."""
  return _tv_colors[vid % len(_tv_colors)]


def warp_color_8x(tid, vid):
  """Color by warp ``(tid // 32) % 8`` -> ``(r, g, b)`` (32 threads per warp)."""
  return _tv_colors[(tid // 32) % len(_tv_colors)]


# ---- Constant / generic ----

def constant(rgb):
  """Return a coloring functor that ignores its key and always yields ``rgb``."""
  return lambda *args: rgb


def white(*args):
  """Constant white ``(255, 255, 255)``; valid for either functor signature."""
  return (255, 255, 255)