# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for examples/einfold.py.

Run from the repository root:
    pytest examples/einfold_test.py
"""

import ctypes
import re
import logging

import pytest

from pycute import *
from examples.einfold import einfold, _parse_modes

logger = logging.getLogger()


class TestEinfold:

  def _check(self, subscripts, source, expected):
    """Fold `source` as a Layout, as a Tensor, and as its Shape and Stride."""
    result = einfold(subscripts, source)
    assert is_layout(result)
    logger.info(f" {subscripts}: {source} -> {result}")
    assert result == expected

    tensor = Tensor(ImplicitAccessor(17), source)
    result = einfold(subscripts, tensor)
    assert is_tensor(result)
    assert result is not tensor
    assert result.accessor is tensor.accessor
    assert result.layout == expected

    assert einfold(subscripts, shape(source))  == shape(expected)
    assert einfold(subscripts, stride(source)) == stride(expected)

  def _check_error(self, subscripts, source, message, error=ValueError):
    """Fold `source` and expect `message`."""
    with pytest.raises(error, match=re.escape(message)):
      einfold(subscripts, source)

  # -- what the subscripts do --------------------------------------------------

  def test_examples(self):
    # Permuting and folding the top-level modes.
    self._check("ijk -> ijk",
                Layout((2, 3, 5), (7, 11, 13)),
                Layout((2, 3, 5), (7, 11, 13)))
    self._check("ijkl -> kilj",
                Layout((2, 3, 5, 7), (1, 2, 6, 30)),
                Layout((5, 2, 7, 3), (6, 1, 30, 2)))
    self._check("abcd -> (ac)(db)",
                Layout((2, 3, 5, 7), (1, 2, 6, 30)),
                Layout(((2, 5), (7, 3)), ((1, 6), (30, 2))))
    self._check("(ab)cd -> d(ac)b",                       # size-1 and zero-stride modes
                Layout(((1, 3), 1, 5), ((0, 7), 0, 11)),
                Layout((5, (1, 1), 3), (11, (0, 0), 7)))
    self._check("(ab)cd -> d(bc)a",                       # noncompact strides
                Layout(((2, 3), 5, 7), ((101, 17), 1009, 43)),
                Layout((7, (3, 5), 2), (43, (17, 1009), 101)))

    # Each pair of parentheses adds one level.
    self._check("ij -> ij",     Layout((2, 3), (1, 2)), Layout((2, 3), (1, 2)))
    self._check("ij -> (ij)",   Layout((2, 3), (1, 2)), Layout(((2, 3),), ((1, 2),)))
    self._check("ij -> ((ij))", Layout((2, 3), (1, 2)), Layout((((2, 3),),), (((1, 2),),)))

    # A group descends into a nested source mode; a name keeps its mode whole.
    self._check("(ab)c -> abc",
                Layout(((2, 3), 5), ((1, 2), 6)),
                Layout((2, 3, 5), (1, 2, 6)))
    self._check("(ab)c -> (ba)c",
                Layout(((2, 3), 5), ((7, 11), 13)),
                Layout(((3, 2), 5), ((11, 7), 13)))
    self._check("tb -> bt",
                Layout(((2, 3), 5), ((7, 11), 13)),
                Layout((5, (2, 3)), (13, (7, 11))))
    self._check("(ab)(cd) -> (acb)d",
                Layout(((2, 3), (5, 7)), ((1, 2), (6, 30))),
                Layout(((2, 5, 3), 7), ((1, 6, 2), 30)))
    self._check("((ab)(cd))e -> e((da)(bc))",
                Layout((((2, 3), (5, 7)), 11), (((1, 2), (6, 30)), 210)),
                Layout((11, ((7, 2), (3, 5))), (210, ((30, 1), (2, 6)))))

    # Whitespace is ignored.
    for subscripts in ["ijk->k(ij)", " i j k -> k ( i j ) "]:
      self._check(subscripts,
                  Layout((2, 3, 5), (1, 2, 6)),
                  Layout((5, (2, 3)), (6, (1, 2))))

    # Repeating an output name duplicates the mode it binds.
    self._check("ij -> iji",   Layout((2, 3), (1, 2)), Layout((2, 3, 2), (1, 2, 1)))
    self._check("ij -> (ii)j", Layout((2, 3), (1, 2)), Layout(((2, 2), 3), ((1, 1), 2)))
    self._check("ij -> ii",    Layout((2, 3), (1, 2)), Layout((2, 2), (1, 1)))
    self._check("tb -> tbt",
                Layout(((2, 3), 5), ((1, 2), 6)),
                Layout(((2, 3), 5, (2, 3)), ((1, 2), 6, (1, 2))))
    self._check("(ab)c -> (aca)b",
                Layout(((2, 3), 5), ((1, 2), 6)),
                Layout(((2, 5, 2), 3), ((1, 6, 1), 2)))

    # Omitting a name drops the mode it binds.
    self._check("ijk -> ik",
                Layout((2, 3, 5), (1, 2, 6)),
                Layout((2, 5), (1, 6)))
    self._check("(ab)c -> ab",
                Layout(((2, 3), 5), ((1, 2), 6)),
                Layout((2, 3), (1, 2)))
    self._check("ijk -> ik",                              # mode 3 is unnamed, so it stays
                Layout((2, 3, 5, 7), (1, 2, 6, 30)),
                Layout((2, 5, 7), (1, 6, 30)))
    self._check("ijk -> (ik)",
                Layout((2, 3, 5, 7), (1, 2, 6, 30)),
                Layout(((2, 5), 7), ((1, 6), 30)))

    # A lone name binds the whole shape, however deep it is.
    self._check("i -> i",   Layout(4, 5),       Layout(4, 5))
    self._check("i -> (i)", Layout(4, 5),       Layout((4,), (5,)))
    self._check("i -> i",   Layout((4,), (5,)), Layout((4,), (5,)))
    self._check("(i) -> i", Layout((4,), (5,)), Layout(4, 5))
    self._check("i -> i",
                Layout((2, 3, 5), (1, 2, 6)),
                Layout((2, 3, 5), (1, 2, 6)))
    self._check("i -> (i)",
                Layout((2, 3, 5), (1, 2, 6)),
                Layout(((2, 3, 5),), ((1, 2, 6),)))
    self._check("i -> ii",
                Layout((2, 3, 5), (1, 2, 6)),
                Layout(((2, 3, 5), (2, 3, 5)), ((1, 2, 6), (1, 2, 6))))

    # Wrapping and unwrapping are inverses.
    source  = Layout(((2, 3), 5), ((1, 2), 6))
    wrapped = einfold("i -> (i)", source)
    assert wrapped == Layout((((2, 3), 5),), (((1, 2), 6),))
    assert einfold("(i) -> i", wrapped) == source

    # The top-level modes the input leaves unnamed are appended unchanged.
    self._check("(i) -> i",                               # the unnamed suffix re-wraps
                Layout((2, 3, 5), (1, 2, 6)),
                Layout((2, 3, 5), (1, 2, 6)))
    self._check("(i) -> ((i))",
                Layout((2, 3, 5), (1, 2, 6)),
                Layout(((2,), 3, 5), ((1,), 2, 6)))
    self._check("ij -> ji",
                Layout(((12, 4), 42, 5, 7)),
                Layout((42, (12, 4), 5, 7), (48, (1, 12), 2016, 10080)))
    self._check("ij -> (ji)",
                Layout((2, 3, 5, 7), (1, 2, 6, 30)),
                Layout(((3, 2), 5, 7), ((2, 1), 6, 30)))
    self._check("((ab)(cd))e -> e((da)(bc))",
                Layout((((2, 3), (4, 5)), 6, 7)),
                Layout((6, ((5, 2), (3, 4)), 7), (120, ((24, 1), (2, 6)), 720)))

    # A group at the top level is one mode, as it is anywhere else.
    for subscripts in ["ij -> (ij)", "ijk -> (ij)k"]:
      self._check(subscripts,
                  Layout((2, 3, 5, 7), (1, 2, 6, 30)),
                  Layout(((2, 3), 5, 7), ((1, 2), 6, 30)))
    for subscripts in ["(ij) -> ji", "(ij)k -> jik"]:
      self._check(subscripts,
                  Layout(((2, 3), 5, 7), ((1, 2), 6, 30)),
                  Layout((3, 2, 5, 7), (2, 1, 6, 30)))

  # -- what einfold folds ------------------------------------------------------

  def test_htuple_examples(self):
    # An HTuple folds by the same rules as the Layout it could be the shape of.
    assert einfold("ij -> ji",       (2, 3))              == (3, 2)
    assert einfold("ij -> (ij)",     (2, 3))              == ((2, 3),)
    assert einfold("i -> i",         (2, 3, 5))           == (2, 3, 5)
    assert einfold("(ab)c -> (ca)b", ((2, 3), 5))         == ((5, 2), 3)
    assert einfold("ij -> ji",       ((12, 4), 42, 5, 7)) == (42, (12, 4), 5, 7)

    # Only the profile is read, so the leaves can be anything: stride scalars,
    # slice markers, Layouts, names ...
    assert einfold("ij -> ji", (F2(1), F2(2)))         == (F2(2), F2(1))
    assert einfold("ij -> ji", (E(0), E(1)))           == (E(1), E(0))
    assert einfold("ij -> ji", (None, 3))              == (3, None)
    assert einfold("(ab)c -> (cb)a", ((None, 1), 2))   == ((2, 1), None)
    assert einfold("ij -> ji", ("m", "n"))             == ("n", "m")

    # A Tiler's Layouts are its leaves, so each is one whole mode.
    tiler = (Layout((2, 3)), Layout(5))
    assert einfold("ij -> ji", tiler)   == (Layout(5), Layout((2, 3)))
    assert einfold("ij -> (ij)", tiler) == ((Layout((2, 3)), Layout(5)),)

    # A leaf is a rank-1 HTuple, so a lone name binds it whole.
    assert einfold("i -> i",   4) == 4
    assert einfold("i -> (i)", 4) == (4,)
    assert einfold("i -> ii",  4) == (4, 4)

    # So every part of a Layout folds with the Layout.
    A = Layout(((2, 3), 5, 7), ((1, 2), 6, 30))
    B = einfold("(ab)c -> c(ba)", A)
    assert shape(B)  == einfold("(ab)c -> c(ba)", shape(A))
    assert stride(B) == einfold("(ab)c -> c(ba)", stride(A))

    # Each kind of source produces its own kind of result.
    assert type(einfold("ij -> ji", Layout((2, 3)))) is Layout
    assert type(einfold("ij -> ji", (2, 3)))         is tuple

  def test_tensor_views_alias_their_source(self):
    source = make_tensor(Layout((2, 3), (1, 2)), dtype=ctypes.c_int)
    for i in range(2):
      for j in range(3):
        source[i, j] = 10 * i + j

    view = einfold("ij -> ji", source)
    assert is_tensor(view)
    assert view.accessor is source.accessor
    for i in range(2):
      for j in range(3):
        assert view[j, i] == source[i, j]

    view[2, 1] = 123                       # the view indexes the source's storage
    assert source[1, 2] == 123
    source[0, 1] = 456
    assert view[1, 0] == 456

  # -- mode expression parsing -------------------------------------------------

  def test_parse_modes(self):
    cases = [
        ("i", "i"),                                      # a lone name is a leaf
        ("ij", ("i", "j")),
        ("(i)", ("i",)),                                 # a group around a lone
        ("((i))", (("i",),)),                            #   name nests it deeper
        ("(ji)", (("j", "i"),)),                         # every other group is
        ("((ji))", ((("j", "i"),),)),                    #   one mode of its level
        ("(ab)c", (("a", "b"), "c")),
        ("((ab)c)d", ((("a", "b"), "c"), "d")),
        ("(a)b", (("a",), "b")),
        ("  a ( b c ) ", ("a", ("b", "c"))),
        ("  ( i ) ", ("i",)),
        ("i0j1", ("i", "0", "j", "1")),
    ]
    for expression, expected in cases:
      assert _parse_modes(expression) == expected

  # -- what einfold rejects ----------------------------------------------------

  def test_malformed_expression_examples(self):
    cases = [
        ("", "empty mode expression"),
        ("   ", "empty mode expression"),
        ("()", "empty mode group"),
        ("(", "unclosed '('"),
        ("(a(b)", "unclosed '('"),
        (")", "unmatched ')'"),
        ("a)b", "unmatched ')'"),
        ("i,j", "separators between modes are implied"),
        ("i_j", "invalid mode name '_'"),
        ("i+j", "invalid mode name '+'"),
        ("i.j", "invalid mode name '.'"),
        ("[ij]", "invalid mode name '['"),
    ]
    for expression, message in cases:
      with pytest.raises(ValueError, match=re.escape(message)):
        _parse_modes(expression)
      self._check_error(f"{expression} -> ij", Layout((2, 3)), message)
      self._check_error(f"ij -> {expression}", Layout((2, 3)), message)

  def test_error_examples(self):
    # The subscripts hold exactly one arrow.
    for subscripts in ["ij", "ji", "ij>ji", "ij-ji", "ij->ji->k"]:
      self._check_error(subscripts, Layout((2, 3)), "expected exactly one '->'")

    # Input names are distinct, and output names come from the input.
    self._check_error("ii->i", Layout((2, 3)), "input contains repeated names")
    self._check_error("(ij)i->ij", Layout(((2, 3), 5)), "input contains repeated names")
    self._check_error("ij->ik", Layout((2, 3)), "output contains unknown names: ['k']")
    self._check_error("ij->ijk", Layout((2, 3)), "output contains unknown names: ['k']")
    self._check_error("ij->(jk)", Layout((2, 3)), "output contains unknown names: ['k']")

    # The input coarsens the source profile, whatever the source is.
    unmatched = "does not match the source profile"
    self._check_error("ijk -> kji", Layout((2, 3)), unmatched)        # rank 3 vs rank 2
    self._check_error("ijk -> kji", (2, 3), unmatched)
    self._check_error("(ij)k -> ijk", Layout((2, 3)), unmatched)      # group vs scalar mode
    self._check_error("((ij)k)l -> ijkl", Layout((2, 3)), unmatched)
    self._check_error("(ijk)l -> ijkl", Layout(((2, 3), 5)), unmatched)
    self._check_error("(ij)(kl) -> ijkl", Layout(((2, 3), 5, 11)), unmatched)
    self._check_error("(i) -> i", Layout(4, 5), unmatched)            # nothing to unwrap
    for leaf in [42, None, "tensor", F2(1)]:                          # a leaf has one mode
      self._check_error("ij -> ji", leaf, unmatched)

    # A Tiler's Layouts are leaves of it, so no group descends into one.
    self._check_error("(ab)c -> abc", (Layout((2, 3)), Layout(5)), unmatched)
