# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ensure the docs keep describing pycute's public API accurately.

The API reference is generated from the docstrings, so running the tests brings
it back in line rather than complaining that it is behind. The tutorial chapters
are written by hand, so their signatures are compared against the implementation
instead.
"""

import ast
import inspect
import re
import warnings
from pathlib import Path

from scripts.gen_api_reference import API_REF, render

ROOT = Path(__file__).resolve().parents[1]
# The API reference is generated, and is checked by regenerating it rather than
# by parsing it back.
HANDWRITTEN = [d for d in sorted((ROOT / "docs").glob("*.md")) if d != API_REF]

BACKTICKED = re.compile(r"`([^`]+)`")
# `Layout._set(...)` and `class Layout(...)` both name a callable to check.
SIGNATURE = re.compile(r"^(?:class\s+)?([A-Za-z_][\w.]*)\((.*)\)$")

P = inspect.Parameter


def test_api_reference_is_up_to_date():
  """Bring the generated reference back in line with the docstrings.

  The reference is a build product, so a stale one is regenerated here rather
  than reported: editing a docstring and running the tests is the whole
  workflow. The rewrite is warned about rather than done silently, since it
  leaves a change in the working tree that still needs committing.

  CI does not rely on this, and checks the committed file with
  `gen_api_reference.py --check` before the tests run.
  """
  text = render()
  if API_REF.read_text(encoding="utf-8") == text:
    return
  API_REF.write_text(text, encoding="utf-8")
  warnings.warn(f"regenerated docs/{API_REF.name} from the docstrings; "
                f"commit it alongside the docstring that changed")


def _doc_namespace() -> dict[str, object]:
  """Every object a heading may name: pycute's exports plus the util helpers."""
  import importlib

  import pycute

  namespace = dict(vars(pycute))
  for module in ("draw_colors", "draw_svg", "draw_latex", "print_table",
                 "print_tensor"):
    mod = importlib.import_module(f"pycute.util.{module}")
    for attr in dir(mod):
      if not attr.startswith("_"):
        namespace.setdefault(attr, getattr(mod, attr))
  return namespace


def _resolve(dotted: str, namespace: dict[str, object]):
  obj = namespace.get(dotted.split(".")[0])
  for attr in dotted.split(".")[1:]:
    obj = getattr(obj, attr, None)
  return obj


def _render_default(value) -> str:
  """Spell a default value the way a heading would write it."""
  if isinstance(value, type) or inspect.isroutine(value):
    return value.__name__
  return repr(value)


def _implemented_params(obj) -> list[tuple]:
  """(name, kind, default) per parameter of `obj`, ignoring annotations."""
  return [
    (p.name, p.kind, None if p.default is P.empty else _render_default(p.default))
    for p in inspect.signature(obj).parameters.values()
  ]


def _documented_params(params: str) -> list[tuple]:
  """(name, kind, default) per parameter of a heading's parameter list."""
  args = ast.parse(f"def _({params}): pass").body[0].args
  required = len(args.args) - len(args.defaults)
  out = [(p.arg, P.POSITIONAL_ONLY, None) for p in args.posonlyargs]
  for i, p in enumerate(args.args):
    default = None if i < required else ast.unparse(args.defaults[i - required])
    out.append((p.arg, P.POSITIONAL_OR_KEYWORD, default))
  if args.vararg:
    out.append((args.vararg.arg, P.VAR_POSITIONAL, None))
  for p, d in zip(args.kwonlyargs, args.kw_defaults):
    out.append((p.arg, P.KEYWORD_ONLY, None if d is None else ast.unparse(d)))
  if args.kwarg:
    out.append((args.kwarg.arg, P.VAR_KEYWORD, None))
  return out


def _agree(documented: list[tuple], implemented: list[tuple]) -> bool:
  if len(documented) != len(implemented):
    return False
  for (doc_name, doc_kind, doc_default), (name, kind, default) in zip(
    documented, implemented
  ):
    if (doc_name, doc_kind) != (name, kind):
      return False
    # A heading may qualify a default with its module (`ctypes.c_double`).
    if doc_default != default and (
      doc_default is None or doc_default.rsplit(".", 1)[-1] != default
    ):
      return False
  return True


def _format(params: list[tuple]) -> str:
  """Render parameters back into Python signature syntax."""
  kinds = [kind for _, kind, _ in params]
  bits: list[str] = []
  for i, (name, kind, default) in enumerate(params):
    if kind is P.KEYWORD_ONLY and not {P.VAR_POSITIONAL, P.KEYWORD_ONLY} & set(kinds[:i]):
      bits.append("*")                # the marker opening the keyword-only group
    prefix = {P.VAR_POSITIONAL: "*", P.VAR_KEYWORD: "**"}.get(kind, "")
    bits.append(f"{prefix}{name}" if default is None else f"{name}={default}")
    if kind is P.POSITIONAL_ONLY and P.POSITIONAL_ONLY not in kinds[i + 1:]:
      bits.append("/")
  return "(" + ", ".join(bits) + ")"


def _documented_signatures(text: str):
  """Yield (line number, name, parameter list) for every signature heading."""
  for lineno, line in enumerate(text.splitlines(), 1):
    if not line.startswith("### "):
      continue
    for span in BACKTICKED.findall(line):
      m = SIGNATURE.match(span.strip())
      if m:
        yield lineno, m.group(1), m.group(2)


def test_documented_signatures_match_implementation():
  """A hand-written heading that spells a signature must spell the real one.

  Parameter names, order, defaults, and -- the easiest thing to get wrong --
  which parameters are keyword-only are all compared, so changing a signature
  without updating the tutorial chapters fails here. A heading that names a
  function without parentheses claims nothing about its parameters and is not
  checked.
  """
  namespace = _doc_namespace()
  checked, wrong = 0, []
  for doc in HANDWRITTEN:
    text = doc.read_text(encoding="utf-8")
    for lineno, name, params in _documented_signatures(text):
      obj = _resolve(name, namespace)
      if obj is None:
        continue                      # not a pycute name; other tests cover this
      try:
        implemented = _implemented_params(obj)
        documented = _documented_params(params)
      except (ValueError, TypeError, SyntaxError):
        continue                      # no introspectable/parseable signature
      checked += 1
      if not _agree(documented, implemented):
        wrong.append(
          f"{doc.name}:{lineno}: documents `{name}{_format(documented)}` "
          f"but the implementation is `{name}{_format(implemented)}`"
        )

  assert not wrong, "Documented signatures are out of date:\n" + "\n".join(wrong)
  # Guard against a heading-format change silently reducing this to a no-op.
  assert checked > 10, f"Only {checked} documented signatures found; parsing likely broke"
