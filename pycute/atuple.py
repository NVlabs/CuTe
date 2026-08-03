# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Arithmetic Tuples and related utilities.

An :class:`ArithTuple` is an element of ``Z^S``: a hierarchical tuple 
of ints under elementwise addition and scalar multiplication, with 
implicit zero-extension along trailing positions.

Representation
==============
An :class:`ArithTuple` instance carries one field:

    self.data : tuple[int | ArithTuple]

Children are either a Python ``int`` or an :class:`ArithTuple`.
``data`` is stored verbatim from the constructor arguments, so the same 
algebraic element admits multiple equivalent representations — e.g. 
``ArithTuple(1, 0)``, ``ArithTuple((1,))``, and ``E(0)`` all denote 
``1*e_0 + 0*e_1 + ...`` and have ``data`` ``(1, 0)``, ``(1,)``, and 
``(1,)`` respectively.

``int`` and :class:`ArithTuple` always have different depth: ``int 1``
is the depth-0 scalar, while ``ArithTuple((1,))`` is the depth-1 element
with one explicit leaf. The single-int passthrough in :meth:`__new__`
means ``ArithTuple(1)`` returns the depth-0 ``int 1``; use
``ArithTuple((1,))``, ``ArithTuple(1, 0)``, or ``E(0)`` to construct a
depth-1 element.

Equality
========
:func:`__eq__` walks both operands with trailing positions extended by 
implicit zeros. Consequences:

    ArithTuple(1, 0) == ArithTuple((1,)) == E(0)
    int 0 == ArithTuple((0,)) == ArithTuple(0, 0) # zero equality
    int 5 != ArithTuple((5,))                     # different depths

Construction
============
:class:`ArithTuple(*args)` is the public, fully-validating constructor.
Accepts ints, ArithTuples, tuples / lists, or varargs. Recursively lifts 
nested tuples to :class:`ArithTuple` children. Single-int and single-
:class:`ArithTuple` args pass through unchanged.

Pretty printing is hybrid: a single nonzero leaf renders as
``value@p_n@...@p_0`` basis form, everything else as a Python tuple.
"""

from itertools import zip_longest

from .typedefs import is_int, is_static, Integer, StrideScalar
from .htuple import is_tuple, get
from .shape import idx2crd


def _colex_lt(A, B):
  """Strict colex order on ``int | ArithTuple``. Walks the dense view
  from the highest position downward; raises on rank mismatch (a
  nonzero int compared with an ArithTuple)."""
  if is_int(A) and is_int(B):
    return A < B
  if is_int(A) and A != 0:
    raise ValueError(f"colex_lt: rank-incompatible {A!r} < {B!r}")
  if is_int(B) and B != 0:
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
  """Equality under implicit zero-extension.

  Each operand is an :class:`int` or an :class:`ArithTuple`. Trailing
  positions extend by zero, so the unique additive identity ``int 0``
  is equal to an all-zero :class:`ArithTuple` of any rank, and two
  :class:`ArithTuple`s are equal whenever every (explicit or
  implicit) child agrees."""
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
  """An element of the hierarchical integer-module ``Z^S``: a 
  hierarchical tuple of ints under elementwise addition and 
  scalar multiplication, with implicit zero-extension along 
  trailing positions.

  Closed under elementwise addition and scalar multiplication::

      ArithTuple(A,B,ArithTuple(C,D)) + ArithTuple(W,X,ArithTuple(Y,Z))
        == ArithTuple(A+W,B+X,ArithTuple(C+Y,D+Z))
      X * ArithTuple(A,B,ArithTuple(C,D))
        == ArithTuple(X*A,X*B,ArithTuple(X*C,X*D))

  Adding a scalar to an ArithTuple is an incompatibility error.
  """
  __slots__ = ("data",)

  # ------------------------------------------------------------------
  # Construction
  # ------------------------------------------------------------------

  def __new__(cls, *args):
    """Public, fully-validating constructor. Recursively lifts nested
    raw tuples / lists to :class:`ArithTuple` children. Single-int and 
    single-ArithTuple args pass through unchanged."""
    if len(args) == 1:
      arg = args[0]
      if isinstance(arg, ArithTuple) or is_int(arg):
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
      if isinstance(x, ArithTuple) or is_int(x):
        data.append(x)
      elif is_tuple(x):
        data.append(cls(x))
      else:
        raise TypeError(f"ArithTuple: bad leaf {x!r}")
    return cls._set(data)

  @classmethod
  def _set(cls, data):
    """Store a sequence of already-lifted children verbatim as ``self.data``."""
    obj = object.__new__(cls)
    obj.data = tuple(data)
    return obj

  # ------------------------------------------------------------------
  # Algebra
  # ------------------------------------------------------------------

  def __add__(self, other):
    other = ArithTuple(other)          # lift / passthrough
    if is_int(other):
      if other == 0:
        return self
      raise TypeError(f"ArithTuple Incompatibility: {self} + {other}")
    return ArithTuple._set([a + b for a, b in zip_longest(self.data, other.data, fillvalue=0)])

  def __radd__(self, other):
    return self.__add__(other)         # commutative

  def __mul__(self, other):
    if not is_int(other):
      raise TypeError(f"{self} * {other!r}")
    return ArithTuple._set([c * other for c in self.data])

  def __rmul__(self, other):
    return self.__mul__(other)

  def __matmul__(self, other):
    """``x @ i`` wraps ``x`` at outer index ``i`` 
    (i leading zeros, then ``x`` at position ``i``)."""
    if not is_int(other):
      raise TypeError(f"{self} @ {other!r}")
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
    """Pad ``self.data`` with zeros up to ``len(shape)`` and dispatch
    back to the regular :func:`idx2crd`. Returns a plain Python tuple."""
    if not is_tuple(shape):
      raise ValueError(f"_idx2crd({self}, {shape}): rank mismatch")
    if len(shape) < len(self.data):
      raise ValueError(f"_idx2crd({self}, {shape}): rank exceeds shape")
    padded = self.data + (0,) * (len(shape) - len(self.data))
    return idx2crd(padded, shape)

  def _weakly_congruent(self, profile):
    """``self`` is weakly congruent to ``profile`` iff every explicit
    child is admissible at the corresponding position. Trailing
    positions (where ``self`` has implicit 0) are always admissible."""
    from .shape import weakly_congruent
    if not is_tuple(profile):
      return False                     # rank-1+ vs rank-0
    if len(self.data) > len(profile):
      return False                     # rank exceeds profile
    return all(weakly_congruent(c, p) for c, p in zip(self.data, profile))

  def _congruent(self, profile):
    """Trailing zeros extend implicitly, so the same admissibility
    check serves both ``congruent`` and ``weakly_congruent``."""
    return self._weakly_congruent(profile)

  def _is_static(self):
    """True iff every coefficient is static"""
    return all(is_static(c) for c in self.data)

  # ------------------------------------------------------------------
  # Pretty printing
  # ------------------------------------------------------------------

  def __str__(self):
    rep = basis_repr(self)
    if len(rep) == 1:
      value, seq = rep[0]
      return "@".join(str(t) for t in (value,) + seq[::-1])
    return "(" + ", ".join(str(c) for c in self.data) + ")"

  def __repr__(self):
    return str(self)


# =====================================================================
# Factory functions
# =====================================================================

def ScaledBasis(value, seq=()):
  """A scaled basis vector at path ``seq``. Returns the canonical
  ``int`` / :class:`ArithTuple` representation::

      ScaledBasis(A,[])    := A
      ScaledBasis(A,[0])   := (A,0,0,...)
      ScaledBasis(A,[1])   := (0,A,0,...)
      ScaledBasis(A,[0,0]) := ((A,0,0,...),0,0,...)
      ScaledBasis(A,[0,1]) := ((0,A,0,...),0,0,...)
      ScaledBasis(A,[1,0]) := (0,(A,0,0,...),0,...)
      ScaledBasis(A,[1,1]) := (0,(0,A,0,...),0,...)
  """
  result = value
  for i in reversed(seq):
    result = ArithTuple._set((0,) * i + (result,))
  return result


def E(*seq):
  """Unit basis element. ``E(*seq) == ScaledBasis(1, seq)``::

      E()    := 1
      E(0)   := (1,0,0,...)
      E(1)   := (0,1,0,...)
      E(0,0) := ((1,0,0,...),0,0,...)
      E(0,1) := ((0,1,0,...),0,0,...)
      E(1,0) := (0,(1,0,0,...),0,...)
      E(1,1) := (0,(0,1,0,...),0,...)
  """
  return ScaledBasis(1, seq)


class V:
  """Basis-scalar shortcut. ``V(value) @ i`` is sugar for
  ``ScaledBasis(value, (i,))``::

      V(1)     := 1
      V(1)@0   := (1,0,0,...)
      V(1)@1   := (0,1,0,...)
      V(1)@0@0 := ((1,0,0,...),0,0,...)
      V(1)@1@0 := ((0,1,0,...),0,0,...)
      V(1)@0@1 := (0,(1,0,0,...),0,...)
      V(1)@1@1 := (0,(0,1,0,...),0,...)
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
# All three accessors are thin inspectors of :func:`basis_repr`.
# =====================================================================

def basis_repr(x):
  """Algebraic decomposition of ``x`` into scaled basis vectors.

  Returns a non-empty list of ``(v, seq)`` pairs ``result`` such that::

      x == sum(v * E(*s) for v, s in result)

  Each entry corresponds to one nonzero leaf of ``x`` with its path.
  When every leaf of ``x`` is zero (an ``int 0`` or an
  :class:`ArithTuple` whose every leaf is zero) the decomposition
  collapses to the single rank-zero term ``[(0, ())]``."""
  def walk(y, prefix):
    if isinstance(y, ArithTuple):
      for i, c in enumerate(y.data):
        yield from walk(c, prefix + (i,))
    elif is_int(y) and y != 0:
      yield (y, prefix)
  result = list(walk(x, ()))
  return result if result else [(0, ())]


def is_basis(x):
  """True iff ``x`` is a single scaled basis vector ``v * E(*s)``."""
  return len(basis_repr(x)) == 1


# =====================================================================
# Convenience functions
# =====================================================================

def make_basis_like(profile, seq=()):
  """Build a ``profile``-shaped tuple of unit basis elements, with paths
  matching each leaf position of ``profile``."""
  if is_tuple(profile):
    return tuple(make_basis_like(s, seq + (i,))
                 for i, s in enumerate(profile))
  return E(*seq)


def proj(x, profile):
  """Extract from ``x`` the part at the position implied by ``profile``.

  ``profile`` must be a single scaled basis vector ``v * E(*s)`` — i.e.,
  a stride leaf produced by ``leaves(layout.stride)``: a Python ``int``
  (path ``()``, returns ``x`` unchanged), or an :class:`ArithTuple`
  with exactly one nonzero leaf (uses that leaf's path)."""
  rep = basis_repr(profile)
  if len(rep) != 1:
    raise TypeError(f"proj: {profile!r} is not a basis element")
  return get(x, rep[0][1])


def unit(profile):
  """The multiplicative unit of ``profile``'s algebra, at ``profile``'s basis path.

  Drops a stride scalar's magnitude while keeping the algebra and the axis it
  lives on, so that ``unit(d) * n`` rebuilds a stride of magnitude ``n`` in the
  same place -- which is how ``recast`` produces a stride of the same *type* as
  its input::

      unit(5)        == 1                        # Z
      unit(2 * E(1)) == E(1)                     # Z^S: axis kept
      unit(F2(9))    == F2(1)                    # F2, via ``_unit``

  A scalar type supplies ``_unit`` when its algebra's identity is not ``int 1``;
  ``F2`` does, since ``int 1`` would scale by ordinary multiplication rather than
  carry-lessly. Otherwise ``profile`` must be a single scaled basis vector.
  """
  if hasattr(profile, '_unit'):
    return profile._unit()
  rep = basis_repr(profile)
  if len(rep) != 1:
    raise TypeError(f"unit: {profile!r} is not a basis element")
  return E(*rep[0][1])


def as_tuple(obj):
  """Materialize an :class:`ArithTuple` (or a Python tuple/list of them)
  as a plain nested Python tuple."""
  if isinstance(obj, ArithTuple) or is_tuple(obj):
    return tuple(as_tuple(v) for v in obj)
  return obj
