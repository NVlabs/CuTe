# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Methods for layout swizzling
"""

from .stride import *
from .layout import *


class F2(StrideScalar):
  """
  An algebraic type that can be used as a stride scalar for swizzling.

  This type replaces integer addition with binary xor.
  """
  def __init__(self, value):
      self.val_ = value

  def __int__(self):
      return self.val_

  # ------------------------------------------------------------------
  # Equality / ordering
  # ------------------------------------------------------------------

  def __eq__(self, other):
    if isinstance(other, F2):
      return self.val_ == other.val_
    if is_int(other):
      return self.val_ == other
    return NotImplemented

  def __ne__(self, other):
    eq = self.__eq__(other)
    return eq if eq is NotImplemented else not eq

  # ------------------------------------------------------------------
  # Algebra
  # ------------------------------------------------------------------  

  def __add__(self, other):
    if isinstance(other, F2):
      return F2(self.val_ ^ other.val_)
    if is_int(other) and other == 0:
      return self                                      # additive identity
    raise TypeError(f"F2 + non-F2: {self} + {other!r}")
  def __radd__(self, other):
    return self.__add__(other)

  def __mul__(self, other):
    if not (is_int(other) or isinstance(other, F2)):
      raise TypeError(f"F2 * non-int/F2: {self} * {other!r}")
    r = 0
    c = self.val_
    v = other.val_ if isinstance(other, F2) else other
    while v != 0:
      if v & 1: r ^= c
      v >>= 1
      c <<= 1
    return F2(r)
  def __rmul__(self, other):
    return self.__mul__(other)

  # ------------------------------------------------------------------
  # Pretty printing
  # ------------------------------------------------------------------

  def __str__(self):
    return f"F{self.val_}"
  def __repr__(self):
    return f"F{self.val_}"
  def __format__(self, format_spec):
    return format(str(self), format_spec)


def shiftr(a: int, s: int) -> int:
  return a >> s if s >= 0 else a << -s


def shiftl(a: int, s: int) -> int:
  return a << s if s >= 0 else a >> -s


class Swizzle:
  """
  A generic Swizzle functor that can be applied to indices or addresses
  0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                                ^--^  Base is the number of least-sig bits to keep constant
                   ^-^       ^-^      Bits is the number of bits in the mask
                      ^---------^     Shift is the distance to shift the YYY mask
                                        (pos shifts YYY to the right, neg shifts YYY to the left)
  
  e.g. Given
  0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx
  the result is
  0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ xor YY
  """
  def __init__(self, bits, base, shift):
    if bits < 0: raise ValueError(f"Swizzle({bits}, {base}, {shift})")
    if base < 0: raise ValueError(f"Swizzle({bits}, {base}, {shift})")
    if shift < 0 and abs(shift) < bits: raise ValueError(f"Swizzle({bits}, {base}, {shift})")
    self.bits = bits
    self.base = base
    self.shift = shift
    bit_msk = (1 << bits) - 1
    self.yyy_msk = bit_msk << (base + max(0,shift))
    self.zzz_msk = bit_msk << (base - min(0,shift))

  # operator ()    (transform integer)
  def __call__(self, offset):
    return offset ^ shiftr(offset & self.yyy_msk, self.shift)

  # print and str
  def __str__(self):
    return f"SW_{self.bits}_{self.base}_{self.shift}"

  # error msgs and representation
  def __repr__(self):
    return f"Swizzle({self.bits},{self.base},{self.shift})"