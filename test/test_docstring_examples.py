# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate every example in every pycute docstring.

The examples are the part of a docstring most likely to rot, since nothing
about editing a function forces its illustrations to keep up. Running them
turns each one into a test, so an example can only be wrong for as long as it
takes CI to notice.

Under an `Examples:` heading, a line is one of:

    f(x) == y             an assertion, which must evaluate true
    f(bad) -> ValueError  an expected failure, which must raise that exception
    name = ...            a statement, setting up the lines after it

A line may wrap onto following lines by indenting them, and the lines of one
docstring share a namespace, so a name bound by an earlier line is available to
a later one. Schematic examples -- those written with free variables to show a
shape rather than a value -- use `:=` and belong in prose, not under an
`Examples:` heading.
"""

import array
import builtins
import ctypes
import importlib
import inspect
import pkgutil
from fractions import Fraction

import pytest

import pycute
from scripts.gen_api_reference import examples

# The names an example may use beyond `pycute` itself: the standard-library
# pieces the documented signatures already mention.
PRELUDE = {"array": array, "ctypes": ctypes, "Fraction": Fraction}


def _documented():
  """Yield `(qualified name, docstring)` for everything pycute documents."""
  modules = [pycute]
  for info in pkgutil.walk_packages(pycute.__path__, f"{pycute.__name__}."):
    try:
      modules.append(importlib.import_module(info.name))
    except ImportError:
      continue                        # an optional visualization dependency

  for module in modules:
    if module.__doc__:
      yield module.__name__, module.__doc__
    for name, obj in vars(module).items():
      if name.startswith("_") or not (inspect.isclass(obj) or callable(obj)):
        continue
      # The `import *` chains put every name in every downstream namespace;
      # attribute each to the module that defines it, so it is visited once.
      if getattr(obj, "__module__", None) != module.__name__:
        continue
      key = f"{module.__name__}.{name}"
      if obj.__doc__:
        yield key, obj.__doc__
      if not inspect.isclass(obj):
        continue
      for attr, member in vars(obj).items():
        func = member.__func__ if isinstance(member, classmethod) else member
        func = func.fget if isinstance(func, property) else func
        if inspect.isfunction(func) and func.__doc__:
          yield f"{key}.{attr}", func.__doc__


DOCSTRINGS = [(name, doc) for name, doc in _documented() if examples(doc)]


def _run(source: str, namespace: dict) -> None:
  """Evaluate one example, raising `AssertionError` if it does not hold."""
  # Trailing commentary is prose, so it is removed before the arrow is looked
  # for -- `f(x) == y  # a -> b` is an assertion, not an expected failure. The
  # wrapped lines then rejoin into one, since an example is one logical line.
  code = " ".join(l.split("#")[0].strip() for l in source.splitlines()).strip()
  expression, arrow, expected = code.partition("->")

  if arrow:
    exception = getattr(builtins, expected.strip(), None)
    if not (isinstance(exception, type) and issubclass(exception, BaseException)):
      raise AssertionError(f"not an exception name: {source}")
    try:
      eval(compile(expression.strip(), "<example>", "eval"), namespace)
    except exception:
      return
    raise AssertionError(f"example did not raise {exception.__name__}: {source}")

  try:
    compiled = compile(code, "<example>", "eval")
  except SyntaxError:
    exec(compile(code, "<example>", "exec"), namespace)     # a setup statement
    return
  if not eval(compiled, namespace):
    raise AssertionError(f"example is not true: {source}")


def test_examples_were_found():
  """Guard against a parser change silently reducing this to a no-op."""
  total = sum(len(examples(doc)) for _, doc in DOCSTRINGS)
  assert total > 200, f"only {total} examples found; parsing likely broke"


@pytest.mark.parametrize("doc", [d for _, d in DOCSTRINGS],
                         ids=[n for n, _ in DOCSTRINGS])
def test_examples(doc):
  """Every example in one docstring, in order, sharing a namespace."""
  namespace = dict(vars(pycute), **PRELUDE)
  for source in examples(doc):
    _run(source, namespace)
