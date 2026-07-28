# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Type definitions for PyCuTe: the scalar ABCs, their predicates, and the
``HTuple`` type-alias vocabulary.

This module is the single home of PyCuTe's type language (Whitepaper, §2):
the two scalar ABCs (:class:`Integer`, :class:`StrideScalar`), the ``is_*``
predicates that test them, and the documentation-grade ``HTuple`` aliases
(``HTuple``, ``Profile``, ``IntTuple``, ``Shape``, ``Coord``, ``Stride``).
The one alias whose leaf is a ``Layout`` -- ``Tiler`` -- lives in ``layout.py``
next to that class.

The leaf-vs-tuple discrimination and the "what *kind* of scalar is this"
discrimination are split across a small set of *non-overlapping*
predicates, each answering one narrow question. They form a layered
landscape:

  HTuple structure
  ----------------
  ``is_tuple(x) == isinstance(x, (tuple, list))``      (in ``htuple.py``)
      Purely syntactic: the structural boundary between an HTuple
      internal node and an HTuple leaf. ArithTuple, F2, ints,
      Layouts, etc. are all NOT ``is_tuple`` -- they are HTuple
      leaves.

  Integer scalar
  --------------
  ``is_int(x) == isinstance(x, Integer)``              (defined here)
      Registration-based: ``int`` and its subclasses, plus types
      explicitly registered via :func:`register_integer_type`
      (``numpy.integer`` and ``sympy.Expr`` are auto-registered on
      import when available). Captures *"acts as an ordinary
      integer scalar"* -- shapes, coordinates, sizes, divisors.
      :class:`F2` is NOT ``is_int`` even though it has ``__add__`` /
      ``__mul__``, because its ``+`` is XOR not integer addition;
      membership is a deliberate semantic claim made by registration,
      not duck-typing.

  Stride scalar
  -------------
  ``is_stride_scalar(x) == isinstance(x, StrideScalar)``   (defined here)
      ABC for any leaf that can sit at a Layout's stride leaf. Has
      an algebraic contract: ``__add__`` / ``__radd__`` / ``__mul__``
      / ``__rmul__``. Pre-registered: ``int``, ``Integer`` -- so
      anything ``is_int`` is also ``is_stride_scalar``. Subclassed
      explicitly: :class:`ArithTuple`, :class:`F2`.

These predicates partition the scalar-type space into non-overlapping
responsibilities. In particular, none of them duck-types semantics:
``is_tuple`` is a syntactic Python-type check; ``is_int`` /
``is_stride_scalar`` are ABC-registration-based.
"""

import builtins as _builtins

from abc import ABC, abstractmethod
from typing import Any, Union, TypeAlias


class Integer(ABC):
  """Abstract base class for *integer-shaped* scalar types in PyCuTe.

  Membership is determined by **registration**, not duck-typing. The
  rules are:

    1. ``bool`` and ``float`` are explicitly excluded -- even though
       ``bool`` subclasses ``int``, it is not used as an integer
       value anywhere in PyCuTe shape / coordinate code.
    2. Python ``int`` (and any class inheriting from ``int``) is
       recognized automatically via the :meth:`__subclasshook__`
       shortcut.
    3. Every other integer-shaped type must be registered explicitly,
       either by ABC registration (``Integer.register(MyType)``) or
       via the public helper :func:`register_integer_type`.

  Earlier versions of this ABC tried to *duck-type* membership by
  checking that a type advertised a small set of integer-arithmetic
  methods (``__add__``, ``__mul__``, ``__floordiv__``, ``__mod__``).
  That scheme detects a *semantic* property (acts like an integer) by
  *syntactic* means (has these methods), so it is always one method
  away from a silent miscategorisation. :class:`F2` is the canonical
  hazard: its ``+`` is XOR, not integer addition, and it escapes the
  duck-typing set today only by the accident of not (yet) defining
  ``__floordiv__`` / ``__mod__`` -- the moment such a type grows those
  methods it would be silently captured as an Integer.

  Switching to registration makes Integer membership a deliberate
  semantic claim made by the type's author / user, and eliminates
  the false-positive failure mode entirely.

  See :func:`register_integer_type` and the auto-registration of
  ``numpy.integer`` / ``sympy.Expr`` (when those packages are
  importable) at the bottom of this module.
  """
  @classmethod
  def __subclasshook__(cls, c):
    if c in (bool, float):
      return False
    if issubclass(c, int):
      return True
    # Defer to ABC registration. Returning ``NotImplemented`` (rather
    # than ``False``) lets ``isinstance(x, Integer)`` fall through to
    # the standard registry check for types that aren't ``int``
    # subclasses but have been explicitly registered.
    return NotImplemented


def register_integer_type(*types: type) -> None:
  """Declare one or more types as integer-shaped for PyCuTe purposes.

  After registration, :func:`is_int` returns ``True`` for instances
  of the registered types (and their subclasses), and PyCuTe shape /
  coordinate / divisor code will treat them as ordinary integers.

  Pre-registered automatically on import:

    - ``int`` (and any subclass thereof)
    - ``numpy.integer`` if ``numpy`` is importable
    - ``sympy.Expr``    if ``sympy`` is importable
      (covers ``Symbol``, ``Integer``, ``Mul``, ``Add``, ``Pow``, ...)

  Typical use::

      import pycute
      import mylib
      pycute.register_integer_type(mylib.MyIntegerType)

  Idempotent and cheap: registering an already-registered type is a no-op.
  """
  for t in types:
    Integer.register(t)


def is_int(x) -> bool:
  """True iff ``x`` is an instance of an :class:`Integer`-registered type."""
  return isinstance(x, Integer)


def is_static(x) -> bool:
  """True iff ``x`` is a *concrete*, statically-known value.

  The test is grounded in :func:`is_int` and uses the standard ``int()``
  protocol rather than any library-specific attribute: ``x`` is static iff it
  is an integer that equals its own ``int()`` coercion. Python / ``numpy``
  integers and constant ``sympy`` expressions (e.g. ``sympy.Integer(4)``) are
  static; a ``sympy.Symbol`` or any expression carrying a free symbol
  (``4*N``) is *dynamic* -- its ``int()`` raises.
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
  """Quotient and remainder ``(a // b, a % b)``, with an overridable fast path.

  Dispatches through the built-in :func:`divmod` protocol first -- so a type may
  supply a fused ``__divmod__`` / ``__rdivmod__`` -- and otherwise falls back to
   ``//`` and ``%``.

  Examples:
    divmod(7, 3) == (2, 1)
  """
  try:
    return _builtins.divmod(a, b)
  except TypeError:
    return a // b, a % b


# ---------------------------------------------------------------------------
# Best-effort auto-registration of common third-party integer-like types.
#
# When ``numpy`` / ``sympy`` are importable we register their integer-shaped
# base classes so that PyCuTe "just works" for the common case (e.g. sympy
# symbolic shapes / strides, numpy integer dtypes) without requiring any
# user setup.
#
# If a package isn't installed, the try-import fails fast and the
# corresponding registration is silently skipped. For other custom
# symbolic / numeric integer types, use :func:`register_integer_type`
# explicitly.
# ---------------------------------------------------------------------------

try:
  import numpy as _np
  Integer.register(_np.integer)
except ImportError:
  pass

try:
  import sympy as _sym
  # ``sympy.Expr`` is the base of every arithmetic sympy expression --
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
# Its contract is algebraic -- ``+`` / ``*`` by an integer (an
# integer-semimodule) -- so ``int`` and every :class:`Integer` qualify (they are
# registered below), while :class:`ArithTuple` / :class:`F2` subclass it.
# ---------------------------------------------------------------------------

class StrideScalar(ABC):
  """
  An Abstract Base Class for defining stride scalar types.
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
  """True iff ``value`` is an instance of a :class:`StrideScalar` type."""
  return isinstance(value, StrideScalar)


# ---------------------------------------------------------------------------
# HTuple type vocabulary (Whitepaper, §2.1 Tuples/HTuples, §2.2 Shape, §2.3 Stride).
#
# Documentation-grade aliases spelled in the 3.10-compatible form (``TypeAlias``
# + string forward-refs, not the PEP 695 ``type`` statement which is 3.12+).
# They are *hints*, not runtime checks: the carriers are plain
# ``int``/``tuple``/``list``, and the structural contracts (congruence,
# positivity, ...) are enforced at runtime by ``congruent`` /
# ``weakly_congruent`` / ``compatible`` and the ``is_*`` predicates.
#
# Because both scalar leaf types (:class:`Integer`, :class:`StrideScalar`) are
# defined above in this module, every alias here is a *direct* reference -- no
# forward-ref to a higher layer, and no ``TYPE_CHECKING`` import. The one alias
# whose leaf is a ``Layout`` -- ``Tiler`` -- lives in ``layout.py`` beside that
# class.
# ---------------------------------------------------------------------------

#: An HTuple with unconstrained leaves; used for generic, profile-shaped args.
HTuple: TypeAlias = Union[Any, tuple["HTuple", ...], list["HTuple"]]

#: A congruence *profile*: an HTuple whose leaf values are irrelevant -- only its
#: tuple/leaf tree structure matters (see :func:`congruent`).
Profile: TypeAlias = HTuple

#: An HTuple(Integer): the shared carrier of shapes and coordinates.
IntTuple: TypeAlias = Union[Integer, tuple["IntTuple", ...], list["IntTuple"]]

#: A shape: an HTuple of (positive) integers describing per-mode extents (Z+).
Shape: TypeAlias = IntTuple

#: A coordinate into a shape: an HTuple of integers -- natural / integral /
#: flat / admissible -- optionally carrying ``None`` slice-markers at any level.
Coord: TypeAlias = Union[Integer, None, tuple["Coord", ...], list["Coord"]]

#: An HTuple(StrideScalar): the stride half of a Layout, congruent with its Shape.
Stride: TypeAlias = Union[StrideScalar, tuple["Stride", ...], list["Stride"]]
