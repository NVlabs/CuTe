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
      # Idempotent on F2, so wrapping a value that may already be an F2 is safe.
      self.val_ = value.val_ if isinstance(value, F2) else value

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

  # Ordering is by the underlying value, so `F2`s order by leading bit first and
  # by their lower-order terms only to break ties -- i.e. by the bit-field a
  # stride writes into. This is the `F2` reading of "stride magnitude", and it is
  # what the layout algebra's stride sorts want.
  def __lt__(self, other):
    return self.val_ <  int(other)
  def __le__(self, other):
    return self.val_ <= int(other)
  def __gt__(self, other):
    return self.val_ >  int(other)
  def __ge__(self, other):
    return self.val_ >= int(other)

  # ------------------------------------------------------------------
  # Algebra
  # ------------------------------------------------------------------

  # An operand of a type `F2` cannot combine with yields `NotImplemented`, so
  # Python falls back on that operand's reflected operator -- which is how an
  # `ArithTuple` carrying `F2` leaves gets to handle `F2 * ArithTuple` -- and
  # raises only if neither side knows what to do. A *nonzero int* is different: it
  # is an operand `F2` understands and must refuse, since mixing `Z`'s addition
  # into XOR is the hazard this guard exists for.

  def __add__(self, other):
    if isinstance(other, F2):
      return F2(self.val_ ^ other.val_)
    if is_int(other):
      if other == 0:
        return self                                    # additive identity
      raise TypeError(f"F2 Incompatibility: {self} + {other!r}")
    return NotImplemented
  def __radd__(self, other):
    return self.__add__(other)

  # XOR is its own inverse, so subtraction is addition and negation is identity.
  __sub__  = __add__
  __rsub__ = __radd__

  def __neg__(self):
    return self

  def __mul__(self, other):
    if not (is_int(other) or isinstance(other, F2)):
      return NotImplemented
    r = 0
    c = self.val_
    v = other.val_ if isinstance(other, F2) else other
    # An operand acts through its bits, and a negative integer has no finite bit
    # expansion -- shifting it right never reaches 0. Reject it rather than spin.
    if v < 0:
      raise ValueError(f"F2 * negative: {self} * {other!r}")
    while v != 0:
      if v & 1: r ^= c
      v >>= 1
      c <<= 1
    return F2(r)
  def __rmul__(self, other):
    return self.__mul__(other)

  def __divmod__(self, other):
    """
    Euclidean division in `F2`: the unique `(q, r)` satisfying

      `self == q * other + r`   with   `deg(r) < deg(other)`,

    where an `F2` value's bits are the coefficients of a polynomial over the
    two-element field and `deg(F2(v)) == v.bit_length() - 1`.

    `F2`'s `*` is a carry-less product, so this is polynomial long division
    rather than integer division: `divmod(F2(0b1011), 0b11) == (F2(0b110),
    F2(0b1))`, whereas `divmod(11, 3) == (3, 2)` in `Z`.

    Both results are `F2`. Keeping the quotient in `F2` is what lets `idx2crd`
    chain `divmod` across the modes of a shape and `crd2idx` recompose the
    result, with every step staying carry-less.

    Since `deg(r) < deg(other)` implies `r < other`, the remainder is always an
    in-bounds coordinate of an extent-`other` mode. When `other` is a power of
    two -- the only case in which the modes of an `F2` layout occupy disjoint
    bit-fields -- the split is exactly the pair of bit-fields, and the
    quotient's value is the ordinary shift:

      `divmod(F2(a), 2**k) == (F2(a >> k), F2(a & (2**k - 1)))`.

    Examples:
      divmod(F2(0b10110), 4)  == (F2(0b101), F2(0b10))
      divmod(F2(0b1011), 3)   == (F2(0b110), F2(0b1))
      divmod(F2(42), 1)       == (F2(42), F2(0))
    """
    if not (is_int(other) or isinstance(other, F2)):
      raise TypeError(f"F2 divmod non-int/F2: {self} divmod {other!r}")
    d = int(other)
    if d == 0:
      raise ZeroDivisionError(f"F2 divmod zero: {self} divmod {other!r}")
    q, r = 0, self.val_
    deg_d = d.bit_length() - 1
    while r.bit_length() > deg_d:            # deg(r) >= deg(d): cancel r's leading term
      shift = r.bit_length() - 1 - deg_d
      q ^= 1 << shift
      r ^= d << shift
    return F2(q), F2(r)

  # ------------------------------------------------------------------
  # CuTe hooks
  # ------------------------------------------------------------------

  def _is_static(self):
    """True iff the underlying value is statically known."""
    return is_static(self.val_)

  def _unit(self):
    """
    `F2`'s multiplicative unit, which `unit()` uses to lift an integer into this
    algebra: `n * F2(1)` is `F2(n)`, whereas `n * 1` would stay in `Z` and later
    scale by ordinary rather than carry-less multiplication.
    """
    return F2(1)

  def _idx2crd(self, shape):
    """
    Decompose this value into the natural coordinates of `shape`, which is what
    lets a value drawn from an `F2` layout's codomain be fed back in as a
    coordinate.

    The colexicographical `divmod` chain peels the extents off carry-lessly,
    while `crd2idx` recomposes with prefix products taken in `Z`. The two are
    mutually inverse only where those prefix products agree, i.e. where every
    extent multiplies the running product without carrying -- as power-of-two
    extents always do. A shape that carries is rejected rather than decomposed
    into a coordinate that will not recompose.

    Examples:
      idx2crd(F2(0b10110), (4, 8))    == (F2(0b10), F2(0b101))
      idx2crd(F2(0b10110), (3, 3))    == (F2(0b1), F2(0b1101))   # only 3 is a divisor
      idx2crd(F2(0b10110), (3, 3, 3)) -> ValueError    # F2(3) * F2(3) == F2(5) != 9
    """
    if not is_tuple(shape):              # Identity, nothing to decompose
      return self
    def divmod_seq(idx):
      pps = 1                            # Running colex prefix product, in Z
      for s in flatten(shape)[:-1]:      # The last extent is never a divisor
        if F2(pps) * F2(s) != pps * s:
          raise ValueError(f"_idx2crd({self}, {shape}): extent {s} carries into the prefix "
                           f"product {pps}, so the F2 and Z decompositions of {shape} differ")
        pps *= s
        idx,rem = divmod(idx,s)          # (idx // s, idx % s)
        yield rem
      yield idx                          # Avoid mod on last shape
    return unflatten(divmod_seq(self), shape)

  def _coshape_bound(self, contributions):
    """
    Bound the image of a layout with an `F2` codomain, given every mode's
    maximal contribution `(s-1) * d` (`self` is one of them).

    XOR cancels bits instead of accumulating them, so the image is bounded
    neither by the sum of the modes' maxima nor even by the largest of them:
    `Layout(4, F2(3))` reaches 6 while its maximal contribution is 5. What
    *is* bounded is the bit-span. A carry-less product's leading bit is the
    sum of its operands' leading bits, so no coordinate of a mode can reach
    past that mode's own contribution, and the codomain -- a space of
    bit-vectors, not an interval -- is bounded by the smallest power of two
    above them all.
    """
    return 1 << max(int(c).bit_length() for c in leaves(contributions))

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
