# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.typing
"""

import importlib
import logging
import typing

import pytest
import sympy

from pycute import *

logger = logging.getLogger()


class TestTyping:
  def postcondition_typing(self, _cls, _obj, cls, expected: bool):
    logger.info(f" issubclass({_cls}, {cls})")
    logger.info(f" isinstance({_obj}, {cls})")

    assert expected == issubclass(_cls, cls)
    assert expected == isinstance(_obj, cls)

  def test_typing(self):
    self.postcondition_typing(int, 1, Integer, True)
    self.postcondition_typing(float, 1., Integer, False)
    self.postcondition_typing(str, 'hi', Integer, False)
    self.postcondition_typing(bool, False, Integer, False)
    x = sympy.symbols('x')
    self.postcondition_typing(type(x), x, Integer, True)
    # Arithmetic on a sympy symbol returns a sympy.Mul, also covered
    # by the sympy.Expr auto-registration.
    self.postcondition_typing(type(4 * x), 4 * x, Integer, True)

  def test_f2_is_not_integer(self):
    """F2 has integer-shaped *syntax* (``+``, ``*``) but non-integer
    *semantics* (``+`` is XOR). Membership in :class:`Integer` is
    decided by registration, not duck-typing; F2 simply isn't
    registered, so the classification is unambiguous."""
    from pycute.swizzle import F2
    assert not issubclass(F2, Integer)
    assert not isinstance(F2(5), Integer)

  def test_register_integer_type_round_trip(self):
    """A user-defined type that doesn't subclass ``int`` and isn't
    registered must NOT be classified as Integer until
    :func:`register_integer_type` says so."""
    from pycute.typedefs import register_integer_type
    class MyOpaqueInt:
      def __init__(self, v): self.v = v
    assert not isinstance(MyOpaqueInt(3), Integer)
    register_integer_type(MyOpaqueInt)
    assert isinstance(MyOpaqueInt(3), Integer)

  def test_layout_with_sympy_2(self):
    M, N, DM, DN = sympy.symbols("M,N,DM,DN")
    TILE_M, TILE_N = 128, 512
    layout_a = Layout((M, N), (DM, DN))
    layout_t = zipped_divide(layout_a, (TILE_M, TILE_N))
    logger.info(f"{layout_t}")
    assert layout_t.shape[0] == (TILE_M, TILE_N)
    assert layout_t.shape[1] == ((TILE_M + M - 1) // TILE_M, (TILE_N + N - 1) // TILE_N)

    assert layout_t.stride[0] == (DM, DN)
    assert layout_t.stride[1] == (TILE_M*DM, TILE_N*DN)


class TestTypeAliases:
  """Defend the documentation-grade type vocabulary (Whitepaper §2.1-2.4):
  ``HTuple``, ``Profile``, ``IntTuple``, ``Shape``, ``Coord``, ``Stride``,
  ``Tiler``.

  These aliases are *hints*, not runtime-enforced, so the tests assert
  (a) they are exported, (b) their leaf structure matches the Whitepaper, and
  (c) they resolve without the ``NameError`` that an unimported-name draft
  raises -- the concrete regression these guard against.
  """

  ALIASES = ("HTuple", "Profile", "IntTuple", "Shape", "Coord", "Stride", "Tiler")

  def test_aliases_exported(self):
    import pycute
    for name in self.ALIASES:
      assert hasattr(pycute, name), f"{name} missing from pycute namespace"
      assert name in pycute.__all__, f"{name} missing from pycute.__all__"

  def test_typing_helpers_not_exported(self):
    """The typing machinery used to spell the aliases must not leak into the
    public API surface."""
    import pycute
    for name in ("Any", "Union", "TypeAlias", "TYPE_CHECKING"):
      assert name not in pycute.__all__, f"{name} leaked into pycute.__all__"

  def test_alias_module_locations(self):
    """The scalar ABCs and the leaf-scalar HTuple aliases are centralized in
    ``typedefs.py``; only ``Tiler`` (leaf = ``Layout``) lives in ``layout.py``.

    ``typedefs.py`` imports nothing from pycute, so membership in its namespace
    is proof of definition there (not an ``import *`` leak). ``__module__`` on
    the ABCs pins the module that defines them."""
    typedefs = importlib.import_module("pycute.typedefs")
    layout = importlib.import_module("pycute.layout")
    central = {"HTuple", "Profile", "IntTuple", "Shape", "Coord", "Stride",
               "Integer", "StrideScalar"}
    assert central <= set(vars(typedefs))
    assert "Tiler" in vars(layout)
    assert "Tiler" not in vars(typedefs)          # Tiler stays with Layout
    assert Integer.__module__ == "pycute.typedefs"
    assert StrideScalar.__module__ == "pycute.typedefs"

  def test_typedefs_is_the_only_alias_module(self):
    """``typedefs.py`` is the sole home of the type aliases: there is no second
    ``int_typing`` module for them to drift apart in."""
    with pytest.raises(ModuleNotFoundError):
      importlib.import_module("pycute.int_typing")

  def test_alias_leaf_types(self):
    """Each concrete HTuple specialization carries the Whitepaper leaf type."""
    assert Integer in typing.get_args(IntTuple)
    assert Integer in typing.get_args(Shape)
    assert Integer in typing.get_args(Coord)
    assert StrideScalar in typing.get_args(Stride)

  def test_coord_admits_none(self):
    """A ``Coord`` may carry ``None`` slice-markers (per the slicing path in
    ``Tensor.__getitem__`` / ``Layout._offset_and_slice``)."""
    assert type(None) in typing.get_args(Coord)
    assert type(None) not in typing.get_args(Shape)   # a Shape may not

  def test_tiler_includes_layout(self):
    """A ``Tiler`` leaf is an ``Integer`` or a ``Layout`` (the latter a
    forward-ref, since ``Tiler`` is defined alongside ``Layout``)."""
    args = typing.get_args(Tiler)
    assert Integer in args
    assert any(getattr(a, "__forward_arg__", None) == "Layout" for a in args)

  @staticmethod
  def _unwrap(op):
    """A ``@ModeOpDecorator`` name (``get``, ``shape``, ``coshape``, ...) is a
    ``ModeOp`` instance, not a function; its annotations live on ``.func``."""
    return getattr(op, "func", op)

  def test_hints_resolve(self):
    """Every annotated public entry point resolves -- the defense against the
    original ``NameError: name 'Any'/'Tuple' is not defined`` regression.

    With the vocabulary centralized in ``typedefs.py`` (the lowest layer), every
    module resolves standalone -- no ``TYPE_CHECKING`` forward-ref namespaces
    needed anymore."""
    modules = {
      "pycute.typedefs": ["is_int", "is_static", "is_stride_scalar",
                               "register_integer_type"],
      "pycute.htuple":  ["congruent", "weakly_congruent", "wrap",
                              "flatten", "product", "slice_", "get"],
      "pycute.stride":  ["stride", "inner_product", "prefix_product",
                              "coshape", "coprofile"],
      "pycute.shape":   ["shape", "size", "rank", "depth", "compatible",
                              "common_refinement", "crd2idx", "idx2crd",
                              "coordinates"],
      "pycute.layout":  ["make_layout", "make_ordered_layout",
                              "tiler_to_layout", "recast"],
      "pycute.algebra": ["composition", "complement", "logical_divide",
                              "logical_product", "layout_add"],
      "pycute.tensor":  ["make_tensor", "identity_tensor", "is_tensor"],
    }
    for modname, fns in modules.items():
      mod = importlib.import_module(modname)
      for fn in fns:
        typing.get_type_hints(self._unwrap(getattr(mod, fn)))   # must not raise

  def test_layout_signatures_use_vocabulary(self):
    """The public ``Layout`` surface is annotated with the vocabulary.
    ``layout.py`` uses ``from __future__ import annotations``, so the raw
    annotations are the alias *names* as strings."""
    from pycute.layout import Layout
    assert Layout.__init__.__annotations__["shape"] == "Shape"
    assert Layout.__init__.__annotations__["stride"] == "Stride"
    assert Layout.__call__.__annotations__["crd"] == "Coord"

  def test_crd2idx_signature_uses_vocabulary(self):
    """``shape.py`` has no ``from __future__ import annotations``, so its raw
    annotations are the alias *objects* themselves."""
    shape_mod = importlib.import_module("pycute.shape")
    ann = shape_mod.crd2idx.__annotations__
    assert ann["crd"] is Coord
    assert ann["shape"] is Shape
    assert ann["return"] is Integer
