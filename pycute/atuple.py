# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Arithmetic Tuples and related utilities.

An `ArithTuple` is an element of `Z^S`: a hierarchical tuple of stride scalars
under elementwise addition and scalar multiplication, with implicit
zero-extension along trailing positions. It is the single carrier for both
*coordinate strides* (one-term sums like `E(0)`) and *coordinate sums*
(multi-term sums like `3*E(0) + 5*E(1)`).

A leaf is ordinarily an integer, hence `Z^S`, but may be any `StrideScalar`, and
every operation defers to the leaf's own algebra. An `F2` leaf therefore adds by
XOR while its siblings keep integer addition, which is what lets one coordinate
axis carry a swizzled offset while another stays an ordinary index.

An instance carries one field, `self.data`, holding the children verbatim as
given. The same algebraic element therefore admits several representations --
`ArithTuple(1, 0)`, `ArithTuple((1,))` and `E(0)` all denote `1*e_0` while
holding different `data` -- and equality, which extends trailing positions by
zero, identifies them.

A scalar and an `ArithTuple` always differ in depth, and the single-scalar
passthrough in `__new__` means `ArithTuple(1)` is the depth-0 `int 1`; use
`ArithTuple((1,))`, `ArithTuple(1, 0)` or `E(0)` for the depth-1 element.

Pretty printing is hybrid: a single nonzero leaf renders in the basis form
`value@p_n@...@p_0`, everything else as a Python tuple.
"""

from itertools import zip_longest

from .typedefs import is_int, is_static, is_stride_scalar, Integer, StrideScalar
from .htuple import is_tuple, get, lift
from .shape import idx2crd


def _colex_lt(A, B):
  """
  Strict colex order on `StrideScalar | ArithTuple`.

  Walks the dense view from the highest position downward, deferring to the
  leaf type's own ordering; raises on rank mismatch (a nonzero leaf compared
  with an ArithTuple).
  """
  if not isinstance(A, ArithTuple) and not isinstance(B, ArithTuple):
    return A < B
  if not isinstance(A, ArithTuple) and A != 0:
    raise ValueError(f"colex_lt: rank-incompatible {A!r} < {B!r}")
  if not isinstance(B, ArithTuple) and B != 0:
    raise ValueError(f"colex_lt: rank-incompatible {A!r} < {B!r}")
  A_data = A.data if isinstance(A, ArithTuple) else ()
  B_data = B.data if isinstance(B, ArithTuple) else ()
  for i in reversed(range(max(len(A_data), len(B_data)))):
    a = A_data[i] if i < len(A_data) else 0
    b = B_data[i] if i < len(B_data) else 0
    if _colex_lt(a, b): return True
    if _colex_lt(b, a): return False
  return False


def _atuple_eq(A, B):
  """
  Equality under implicit zero-extension.

  Each operand is a scalar leaf or an `ArithTuple`. Trailing positions extend
  by zero, so the unique additive identity `int 0` is equal to an all-zero
  `ArithTuple` of any rank, and two `ArithTuple`s are equal whenever every
  (explicit or implicit) child agrees.
  """
  def view(x):    # -> child sequence, or None if rank-incompatible
    if isinstance(x, ArithTuple): return x.data
    return () if x == 0 else None
  if not isinstance(A, ArithTuple) and not isinstance(B, ArithTuple):
    return A == B
  a, b = view(A), view(B)
  if a is None or b is None: return False
  return all(_atuple_eq(x, y) for x, y in zip_longest(a, b, fillvalue=0))

# =====================================================================
# ArithTuple
# =====================================================================

class ArithTuple(StrideScalar):
  """
  An element of the hierarchical module `Z^S`: a hierarchical tuple of stride
  scalars under elementwise addition and scalar multiplication, with implicit
  zero-extension along trailing positions.

  Closed under `+`, `-` and scalar `*`, elementwise and to any depth:

      ArithTuple(A,B,ArithTuple(C,D)) + ArithTuple(W,X,ArithTuple(Y,Z))
        := ArithTuple(A+W,B+X,ArithTuple(C+Y,D+Z))
      X * ArithTuple(A,B,ArithTuple(C,D))
        := ArithTuple(X*A,X*B,ArithTuple(X*C,X*D))

  Addition forms an abelian *group*: `int 0` is the unique identity and every
  element has a negation, so `-` is elementwise too. Unlike `+` it does not
  commute, so `0 - x` negates `x`. Adding or subtracting a nonzero scalar is an
  incompatibility error.

  Unhashable, because equivalent representations hold different `data` and so a
  structural hash would violate `a == b => hash(a) == hash(b)`.

  Examples:
    ArithTuple(1, 2, 3) + (7, 8, 9)               == (8, 10, 12)
    ArithTuple(1, 2, 3) * 4                       == (4, 8, 12)
    0 - ArithTuple(1, 2)                          == (-1, -2)
    E(0) == ArithTuple(1, 0) == ArithTuple((1,))
    ArithTuple(0, 0) == 0 == ArithTuple((0,))
    ArithTuple((5,)) != 5                                     # different depths
    ArithTuple(1, 2) + 1                          -> TypeError
  """
  __slots__ = ("data",)

  # ------------------------------------------------------------------
  # Construction
  # ------------------------------------------------------------------

  def __new__(cls, *args):
    """
    Public, fully-validating constructor: accepts stride scalars, ArithTuples,
    tuples/lists, or varargs.

    Nested raw tuples and lists are recursively lifted to `ArithTuple`
    children; a single scalar or single ArithTuple passes through unchanged.
    """
    if len(args) == 1:
      arg = args[0]
      if is_stride_scalar(arg):        # includes ArithTuple, int, Integer, F2
        return arg
      if not is_tuple(arg):
        raise TypeError(f"ArithTuple({arg!r})")
      seq = arg
    else:
      seq = args                       # varargs: ArithTuple(a, b, c)
    # Convert nested tuples / lists to ArithTuple children; _set stores
    # the lifted sequence verbatim.
    data = []
    for x in seq:
      if is_stride_scalar(x):          # Any stride scalar may be a leaf, not just int
        data.append(x)
      elif is_tuple(x):
        data.append(cls(x))
      else:
        raise TypeError(f"ArithTuple: bad leaf {x!r}")
    return cls._set(data)

  @classmethod
  def _set(cls, data):
    """Store a sequence of already-lifted children verbatim as `self.data`."""
    obj = object.__new__(cls)
    obj.data = tuple(data)
    return obj

  # ------------------------------------------------------------------
  # Algebra
  # ------------------------------------------------------------------

  def __add__(self, other):
    if not (is_stride_scalar(other) or is_tuple(other)):
      return NotImplemented
    other = ArithTuple(other)          # lift / passthrough
    if not isinstance(other, ArithTuple):
      if other == 0:
        return self                    # additive identity, of any leaf algebra
      raise TypeError(f"ArithTuple Incompatibility: {self} + {other}")
    return ArithTuple._set([a + b for a, b in zip_longest(self.data, other.data, fillvalue=0)])

  def __radd__(self, other):
    return self.__add__(other)         # commutative

  def __sub__(self, other):
    if not (is_stride_scalar(other) or is_tuple(other)):
      return NotImplemented
    other = ArithTuple(other)          # lift / passthrough
    if not isinstance(other, ArithTuple):
      if other == 0:
        return self
      raise TypeError(f"ArithTuple Incompatibility: {self} - {other}")
    return ArithTuple._set([a - b for a, b in zip_longest(self.data, other.data, fillvalue=0)])

  def __neg__(self):
    return ArithTuple._set([-c for c in self.data])

  def __rsub__(self, other):
    return (-self) + other

  def __mul__(self, other):
    if isinstance(other, ArithTuple) or not is_stride_scalar(other):
      return NotImplemented            # Scalars only, but of any leaf algebra
    return ArithTuple._set([c * other for c in self.data])

  def __rmul__(self, other):
    return self.__mul__(other)

  def __matmul__(self, other):
    """
    `x @ i` wraps `x` at outer index `i`: `i` leading zeros, then `x`.

    Examples:
      E(0) @ 1 == E(1, 0)
    """
    if not is_int(other):
      return NotImplemented
    if other < 0:
      raise ValueError(f"{self} @ {other}: negative index")
    return ArithTuple._set((0,) * other + (self,))

  # ------------------------------------------------------------------
  # Tuple interface
  # ------------------------------------------------------------------

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    return self.data[i]

  def __iter__(self):
    return iter(self.data)

  # ------------------------------------------------------------------
  # Equality / ordering
  # ------------------------------------------------------------------

  def __eq__(self, other):
    if isinstance(other, ArithTuple) or is_int(other):
      return _atuple_eq(self, other)
    if is_tuple(other):
      return _atuple_eq(self, ArithTuple(other))
    return NotImplemented

  def __ne__(self, other):
    eq = self.__eq__(other)
    return eq if eq is NotImplemented else not eq

  def __lt__(self, other):
    return _colex_lt(self, ArithTuple(other))

  def __gt__(self, other):
    return _colex_lt(ArithTuple(other), self)

  def __le__(self, other):
    other_l = ArithTuple(other)
    return _colex_lt(self, other_l) or self == other_l

  def __ge__(self, other):
    other_l = ArithTuple(other)
    return _colex_lt(other_l, self) or self == other_l

  # ------------------------------------------------------------------
  # CuTe hooks
  # ------------------------------------------------------------------

  def _idx2crd(self, shape):
    """
    Pad `self.data` with zeros up to `len(shape)` and dispatch back to the
    regular `idx2crd`. Returns a plain Python tuple.
    """
    if not is_tuple(shape):
      raise ValueError(f"_idx2crd({self}, {shape}): rank mismatch")
    if len(shape) < len(self.data):
      raise ValueError(f"_idx2crd({self}, {shape}): rank exceeds shape")
    return idx2crd(self.data + (0,) * (len(shape) - len(self.data)), shape)

  def _is_static(self):
    """True iff every coefficient is static"""
    return all(is_static(c) for c in self.data)

  # ------------------------------------------------------------------
  # Pretty printing
  # ------------------------------------------------------------------

  def __str__(self):
    rep = basis_repr(self)
    if len(rep) == 1:
      value, mode = rep[0]
      return "@".join(str(t) for t in (value,) + mode[::-1])
    return "(" + ", ".join(str(c) for c in self.data) + ")"

  def __repr__(self):
    return str(self)


# =====================================================================
# Factory functions
# =====================================================================

def ScaledBasis(value, mode=()):
  """
  A scaled basis vector at path `mode`, with `value` kept verbatim at the leaf.

  Returns the canonical scalar / `ArithTuple` representation, so an empty `mode`
  collapses to `value` itself and `ScaledBasis(F2(1), (0,))` is an `F2`-valued
  axis.

      ScaledBasis(A,[])    := A
      ScaledBasis(A,[0])   := (A,0,0,...)
      ScaledBasis(A,[1])   := (0,A,0,...)
      ScaledBasis(A,[0,0]) := ((A,0,0,...),0,0,...)
      ScaledBasis(A,[0,1]) := ((0,A,0,...),0,0,...)
      ScaledBasis(A,[1,0]) := (0,(A,0,0,...),0,...)
      ScaledBasis(A,[1,1]) := (0,(0,A,0,...),0,...)

  Examples:
    ScaledBasis(42, [])      == 42
    ScaledBasis(42, [0])     == 42 * E(0)
    ScaledBasis(42, [1, 0])  == 42 * E(1, 0)
  """
  return lift(value, mode=mode, make=ArithTuple._set)


def E(*mode):
  """
  Unit basis element: `E(*mode) == ScaledBasis(1, mode)`.

  The usual way to write a coordinate stride.

      E()    := 1
      E(0)   := (1,0,0,...)
      E(1)   := (0,1,0,...)
      E(0,0) := ((1,0,0,...),0,0,...)
      E(0,1) := ((0,1,0,...),0,0,...)
      E(1,0) := (0,(1,0,0,...),0,...)
      E(1,1) := (0,(0,1,0,...),0,...)

  Examples:
    E()                  == 1
    E(0)                 == ArithTuple(1, 0)
    E(1)                 == ArithTuple(0, 1)
    E(0, 1)              == ArithTuple(ArithTuple(0, 1), 0)
    E(1, 0)              == ArithTuple(0, ArithTuple(1, 0))
    Layout((4, 5), (E(0), E(1)))(2, 3) == ArithTuple(2, 3)
  """
  return ScaledBasis(1, mode)


class V:
  """
  Basis-scalar shortcut: `V(value) @ i` is sugar for `ScaledBasis(value, (i,))`.

      V(1)     := 1
      V(1)@0   := (1,0,0,...)
      V(1)@1   := (0,1,0,...)
      V(1)@0@0 := ((1,0,0,...),0,0,...)
      V(1)@1@0 := ((0,1,0,...),0,0,...)
      V(1)@0@1 := (0,(1,0,0,...),0,...)
      V(1)@1@1 := (0,(0,1,0,...),0,...)

  Examples:
    V(1) @ 0    == E(0)
    V(42) @ 1   == 42 * E(1)
    V(F2(1)) @ 0 == ScaledBasis(F2(1), (0,))
  """
  __slots__ = ("value",)

  def __init__(self, value):
    self.value = value

  def __matmul__(self, i):
    if not is_int(i):
      raise TypeError(f"V({self.value}) @ {i!r}")
    return ScaledBasis(self.value, (i,))


# =====================================================================
# Basis-element accessors.
#
# All three accessors are thin inspectors of `basis_repr`.
# =====================================================================

def basis_repr(x):
  """
  Algebraic decomposition of `x` into scaled basis vectors.

  Each entry is one nonzero leaf of `x` with its path, and its value is that
  leaf verbatim, of whatever algebra it belongs to. When every leaf of `x` is
  zero the decomposition collapses to the single rank-zero term `[(0, ())]`,
  matching the decomposition of `int 0`.

  Post-conditions:
    x == sum(v * E(*s) for v, s in basis_repr(x))
    len(basis_repr(x)) >= 1

  Examples:
    basis_repr(5 * E(1, 2))          == [(5, (1, 2))]
    basis_repr(3 * E(0) + 5 * E(1))  == [(3, (0,)), (5, (1,))]
    basis_repr(0)                    == [(0, ())]
    basis_repr(ArithTuple(0, 0))     == [(0, ())]
  """
  def walk(y, prefix):
    if isinstance(y, ArithTuple):
      for i, c in enumerate(y.data):
        yield from walk(c, prefix + (i,))
    elif is_stride_scalar(y) and y != 0:
      yield (y, prefix)
  result = list(walk(x, ()))
  return result if result else [(0, ())]


def is_basis(x):
  """
  True iff `x` is a single scaled basis vector `v * E(*s)`.

  Every Python `int` qualifies, being the rank-zero basis element.

  Examples:
    is_basis(5 * E(1, 2))          == True
    is_basis(0)                    == True
    is_basis(3 * E(0) + 5 * E(1))  == False
  """
  return len(basis_repr(x)) == 1


# =====================================================================
# Convenience functions
# =====================================================================

def make_basis_like(profile, mode=()):
  """
  Build a `profile`-shaped tuple of unit basis elements, one per leaf position.

  Post-conditions:
    congruent(result, profile)
    get[path](result) == E(*path)   for every leaf path of profile

  Examples:
    make_basis_like((10, 20))         == (E(0), E(1))
    make_basis_like((10, (20, 30)))   == (E(0), (E(1, 0), E(1, 1)))
    make_basis_like(10)               == E()
    congruent(make_basis_like((10, (20, 30))), (10, (20, 30))) == True
  """
  if is_tuple(profile):
    return tuple(make_basis_like(s, mode + (i,))
                 for i, s in enumerate(profile))
  return E(*mode)


def proj(x, basis):
  """
  Extract from `x` the part at the position implied by `basis`.

  Pre-conditions:
    is_basis(basis); a multi-term sum raises a TypeError

  Examples:
    proj((7, 9), E(0))                == 7
    proj((7, 9), 2 * E(1))            == 9
    proj((7, 9), 42)                  == (7, 9)          # rank-zero path
    proj((7, 9), E(0) + E(1))         -> TypeError
  """
  rep = basis_repr(basis)
  if len(rep) != 1:
    raise TypeError(f"proj: {basis!r} is not a basis element")
  return get(x, mode=rep[0][1])


def unit(basis):
  """
  The multiplicative unit of `basis`'s algebra, at `basis`'s basis path.

  Drops a stride scalar's magnitude while keeping the algebra and the axis it
  lives on, so that `unit(d) * n` rebuilds a stride of magnitude `n` in the same
  place -- which is how `recast` produces a stride of the same *type* as its
  input. A scalar type supplies `_unit` when its algebra's identity is not
  `int 1`; `F2` does, since `int 1` would scale by ordinary rather than
  carry-less multiplication.

  Pre-conditions:
    is_basis(basis); a multi-term sum raises a TypeError

  Examples:
    unit(5)                        == 1                          # Z
    unit(2 * E(1))                 == E(1)                       # Z^S: axis kept
    unit(F2(9))                    == F2(1)                      # F2, via _unit
    unit(ScaledBasis(F2(9), (0,))) == ScaledBasis(F2(1), (0,))    # an F2 axis
    unit(E(0) + E(1))              -> TypeError
  """
  if hasattr(basis, '_unit'):
    return basis._unit()
  rep = basis_repr(basis)
  if len(rep) != 1:
    raise TypeError(f"unit: {basis!r} is not a basis element")
  value, mode = rep[0]
  return ScaledBasis(value._unit() if hasattr(value, '_unit') else 1, mode)


def as_tuple(atuple):
  """
  Materialize an `ArithTuple`, or a tuple/list of them, as a plain nested tuple.

  Examples:
    as_tuple(ArithTuple(1, 2, (3, 4)))  == (1, 2, (3, 4))
    as_tuple(42)                        == 42
  """
  if isinstance(atuple, ArithTuple) or is_tuple(atuple):
    return tuple(as_tuple(v) for v in atuple)
  return atuple
