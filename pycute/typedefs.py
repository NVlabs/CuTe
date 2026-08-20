# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Type definitions for PyCuTe: the scalar ABCs, their predicates, and the
`HTuple` type-alias vocabulary.

The single home of PyCuTe's type language (Whitepaper, §2). Three
non-overlapping predicates discriminate a leaf, each answering one question:

  `is_tuple(x)`          (in `htuple.py`) the structural boundary between an
                         HTuple node and an HTuple leaf. Purely syntactic, so
                         `int`, `ArithTuple`, `F2` and `Layout` are all leaves.
  `is_int(x)`            "acts as an ordinary integer scalar": shapes,
                         coordinates, sizes, divisors.
  `is_stride_scalar(x)`  "may sit at a Layout's stride leaf", an algebraic
                         contract of `+` and `*` by an integer.

The latter two are decided by ABC registration rather than by duck-typing, so
that membership is a deliberate semantic claim: `F2` defines `__add__` and
`__mul__` yet is not an `Integer`, because its `+` is XOR.

The `Tiler` alias, whose leaf is a `Layout`, lives in `layout.py` beside that
class; every other alias is here.
"""

import builtins as _builtins

from abc import ABC, abstractmethod
from typing import Any, Union, TypeAlias


class Integer(ABC):
  """
  Abstract base class for *integer-shaped* scalar types in PyCuTe.

  Membership is by **registration**, not duck-typing, so that it states a
  semantic property rather than detecting it syntactically -- a type
  advertising `__add__` and `__mul__` need not add like an integer.

    -- `int`, and any class inheriting from it, is recognized automatically.
    -- `bool` and `float` are excluded, even though `bool` subclasses `int`.
    -- Every other type must be registered, by `register_integer_type` or
       `Integer.register`. `numpy.integer` and `sympy.Expr` are registered on
       import, when importable.

  Examples:
    isinstance(7, Integer)      == True
    isinstance(True, Integer)   == False
    isinstance(1.0, Integer)    == False
    isinstance(F2(1), Integer)  == False    # its + is XOR
  """
  @classmethod
  def __subclasshook__(cls, c):
    if c in (bool, float):
      return False
    if issubclass(c, int):
      return True
    # Defer to ABC registration. Returning `NotImplemented` (rather
    # than `False`) lets `isinstance(x, Integer)` fall through to
    # the standard registry check for types that aren't `int`
    # subclasses but have been explicitly registered.
    return NotImplemented


def register_integer_type(*types: type) -> None:
  """
  Declare one or more types as integer-shaped for PyCuTe purposes.

  After registration `is_int` is true of their instances, and PyCuTe's shape,
  coordinate and divisor code treats them as ordinary integers. Idempotent:
  registering an already-registered type is a no-op.

    import pycute
    pycute.register_integer_type(mylib.MyIntegerType)
  """
  for t in types:
    Integer.register(t)


def is_int(x) -> bool:
  """True iff `x` is an instance of an `Integer`-registered type."""
  return isinstance(x, Integer)


def is_static(x) -> bool:
  """
  True iff `x` is an `Integer` whose value is known at "compile time".

  Grounded in the standard `int()` protocol rather than in any
  library-specific attribute: `x` is static iff it is an integer equal to its
  own `int()` coercion. A `sympy.Symbol`, or any expression carrying a free
  symbol, is therefore dynamic, because its `int()` raises.

  Examples:
    is_static(7)        == True
    is_static(F2(3))    == True
    is_static(1.0)      == False
  """
  if hasattr(x, '_is_static'):
    return x._is_static()
  if not is_int(x):
    return False
  try:
    return bool(x == int(x))
  except Exception:
    return False


def divmod(a, b):
  """
  Quotient and remainder `(a // b, a % b)`, with an overridable fast path.

  Dispatches through the built-in `divmod` protocol first, so a type may supply
  a fused `__divmod__` / `__rdivmod__`, and otherwise falls back to `//` and `%`.

  Examples:
    divmod(7, 3)             == (2, 1)
    divmod(F2(0b1011), 0b11) == (F2(0b110), F2(0b1))   # F2 supplies its own
  """
  try:
    return _builtins.divmod(a, b)
  except TypeError:
    return a // b, a % b


# ---------------------------------------------------------------------------
# Best-effort auto-registration of common third-party integer-like types.
#
# When `numpy` / `sympy` are importable we register their integer-shaped
# base classes so that PyCuTe "just works" for the common case (e.g. sympy
# symbolic shapes / strides, numpy integer dtypes) without requiring any
# user setup.
#
# If a package isn't installed, the try-import fails fast and the
# corresponding registration is silently skipped. For other custom
# symbolic / numeric integer types, use `register_integer_type`
# explicitly.
# ---------------------------------------------------------------------------

try:
  import numpy as _np
  Integer.register(_np.integer)
except ImportError:
  pass

try:
  import sympy as _sym
  # `sympy.Expr` is the base of every arithmetic sympy expression --
  # Symbol, Integer, Rational, Mul, Add, Pow, etc. Registering it covers
  # the "any sympy expression flowing through PyCuTe shape / stride
  # arithmetic" case that the previous duck-typing supported.
  Integer.register(_sym.Expr)
except ImportError:
  pass

# ---------------------------------------------------------------------------
# Stride scalar (Whitepaper, §2.3.1 Integer-Semimodules).
#
# The second scalar ABC: any leaf that can sit at a Layout's stride position.
# Its contract is algebraic -- `+` / `*` by an integer (an
# integer-semimodule) -- so `int` and every `Integer` qualify (they are
# registered below), while `ArithTuple` / `F2` subclass it.
# ---------------------------------------------------------------------------

class StrideScalar(ABC):
  """
  Abstract base class for the leaf types a `Layout`'s stride may hold.

  The contract is algebraic -- `__add__`, `__radd__`, `__mul__`, `__rmul__`,
  making an integer-semimodule (Whitepaper, §2.3.1) -- so `int` and every
  `Integer` qualify and are registered below, while `ArithTuple` and `F2`
  subclass it.

  Examples:
    is_stride_scalar(7)      == True
    is_stride_scalar(E(0))   == True
    is_stride_scalar(F2(1))  == True
    is_stride_scalar(1.0)    == False
  """
  @abstractmethod
  def __add__(self, other):  pass

  @abstractmethod
  def __radd__(self, other): pass

  @abstractmethod
  def __mul__(self, other):  pass

  @abstractmethod
  def __rmul__(self, other): pass

# Register built-in stride scalar types
StrideScalar.register(int)
StrideScalar.register(Integer)


def is_stride_scalar(value) -> bool:
  """True iff `value` is an instance of a `StrideScalar` type."""
  return isinstance(value, StrideScalar)


# ---------------------------------------------------------------------------
# HTuple type vocabulary (Whitepaper, §2.1 Tuples/HTuples, §2.2 Shape, §2.3 Stride).
#
# Documentation-grade aliases spelled in the 3.10-compatible form (`TypeAlias`
# + string forward-refs, not the PEP 695 `type` statement which is 3.12+).
# They are *hints*, not runtime checks: the carriers are plain
# `int`/`tuple`/`list`, and the structural contracts (congruence,
# positivity, ...) are enforced at runtime by `congruent` /
# `weakly_congruent` / `compatible` and the `is_*` predicates.
#
# Because both scalar leaf types (`Integer`, `StrideScalar`) are
# defined above in this module, every alias here is a *direct* reference -- no
# forward-ref to a higher layer, and no `TYPE_CHECKING` import. The one alias
# whose leaf is a `Layout` -- `Tiler` -- lives in `layout.py` beside that
# class.
# ---------------------------------------------------------------------------

#: An HTuple with unconstrained leaves; used for generic, profile-shaped args.
HTuple: TypeAlias = Union[Any, tuple["HTuple", ...], list["HTuple"]]

#: A congruence *profile*: an HTuple whose leaf values are irrelevant -- only its
#: tuple/leaf tree structure matters (see `congruent`).
Profile: TypeAlias = HTuple

#: An HTuple(Integer): the shared carrier of shapes and coordinates.
IntTuple: TypeAlias = Union[Integer, tuple["IntTuple", ...], list["IntTuple"]]

#: A shape: an HTuple of (positive) integers describing per-mode extents (Z+).
Shape: TypeAlias = IntTuple

#: A coordinate into a shape: an HTuple of integers -- natural / integral /
#: flat / admissible -- optionally carrying `None` slice-markers at any level.
Coord: TypeAlias = Union[Integer, None, tuple["Coord", ...], list["Coord"]]

#: An HTuple(StrideScalar): the stride half of a Layout, congruent with its Shape.
Stride: TypeAlias = Union[StrideScalar, tuple["Stride", ...], list["Stride"]]
