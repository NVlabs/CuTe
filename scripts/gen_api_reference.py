#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate `docs/08_api_reference.md` from the docstrings in `pycute`.

Every word of the reference is derived from the source: `pycute.__all__` fixes
the set of documented names, `inspect` supplies each signature, the docstrings
supply all prose, and the `test/` module docstrings supply the test links. The
only way to change the reference is therefore to change the code, which is what
keeps the two from drifting apart.

    python scripts/gen_api_reference.py            # rewrite the reference
    python scripts/gen_api_reference.py --check    # fail if it is out of date

`parse_docstring` and `examples` are also the parser used by
`test/test_docstring_examples.py` to evaluate every documented example, so the
house docstring style below is enforced rather than merely conventional:

    A one-line summary.

    Prose, only where it says something the conditions and examples cannot.

    Pre-conditions:
      conditions the caller must meet
    Post-conditions:
      properties the result satisfies
    Examples:
      f(x) == y                 # an assertion, evaluated by the test suite
      f(bad) -> ValueError      # an expected failure, also evaluated

Conditions and examples are preferred to prose: they say the same thing more
precisely, and unlike prose they are checked.
"""

from __future__ import annotations

import argparse
import ast
import functools
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pycute                                                    # noqa: E402

API_REF = ROOT / "docs" / "08_api_reference.md"

# Presentation order. Which module a name belongs to is not a choice -- it is
# wherever the name is defined -- but the order the modules are introduced in
# is, so it follows the tutorial chapters rather than the import graph.
MODULE_ORDER = (
  "htuple", "typedefs", "stride", "shape", "atuple", "layout", "algebra",
  "swizzle", "accessor", "tensor",
  "util.print_tensor", "util.print_table", "util.draw_svg", "util.draw_latex",
  "util.draw_colors",
)

# `pycute.util` is imported separately: it is optional (the drawers need
# `svgwrite`, `print_table` needs `tabulate`) so it is deliberately left out of
# the `pycute` namespace, and its exports are named by each module's `__all__`
# or, failing that, by its public callables.
UTIL_PREFIX = "util."

# The section headings the house style recognizes. Spelled out rather than
# matched by shape, so that an ordinary sentence ending in a colon stays prose.
# A heading outside this set renders as prose too, which is visible in the
# output rather than silent.
SECTIONS = ("Args", "Pre-conditions", "Post-conditions", "Notable consequences",
            "Examples")

# Sections whose content is code rather than prose.
CODE_SECTIONS = {"Pre-conditions", "Post-conditions", "Examples"}

_SECTION = re.compile(rf"^\s*({'|'.join(SECTIONS)}):\s*$")
_HEADING_CHARS = re.compile(r"[^\w\s-]")


# ---------------------------------------------------------------------------
# Docstring parsing
# ---------------------------------------------------------------------------

def parse_docstring(doc: str) -> tuple[list[list[str]], list[tuple[str, list[str]]]]:
  """Split a docstring into its leading prose and its named sections.

  Returns `(body, sections)`, where `body` is a list of paragraphs (each a list
  of lines) preceding the first `Name:` heading, and `sections` pairs each
  heading with the dedented lines beneath it.
  """
  lines = inspect.cleandoc(doc or "").splitlines()

  body: list[list[str]] = []
  sections: list[tuple[str, list[str]]] = []
  current: list[str] = []                       # lines of the open paragraph
  section: str | None = None                    # name of the open section

  def flush() -> None:
    while current and not current[-1].strip():
      current.pop()
    if not current:
      return
    if section is None:
      body.append(list(current))
    else:
      sections.append((section, list(current)))
    current.clear()

  for line in lines:
    match = _SECTION.match(line)
    if match:
      flush()
      section = match.group(1)
      continue
    if section is None and not line.strip():
      flush()                                   # a blank line ends a paragraph
      continue
    current.append(line)
  flush()

  return body, [(name, _dedent(text)) for name, text in sections]


def _dedent(lines: list[str]) -> list[str]:
  """Remove the common leading whitespace of the non-blank lines."""
  indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
  cut = min(indents) if indents else 0
  return [l[cut:] if l.strip() else "" for l in lines]


def examples(doc: str) -> list[str]:
  """Extract the individual examples from a docstring's `Examples:` sections.

  An example is one logical line: it starts at the section's own indent level
  and continues through every line indented past it, so an assertion may wrap
  across as many lines as it needs.
  """
  out: list[str] = []
  for name, lines in parse_docstring(doc)[1]:
    if name != "Examples":
      continue
    for line in lines:
      if not line.strip():
        continue
      if line[0].isspace() and out:
        out[-1] += "\n" + line                  # a continuation of the last
      else:
        out.append(line)
  return out


# ---------------------------------------------------------------------------
# API surface discovery
# ---------------------------------------------------------------------------

class Entry:
  """One documented name: how to title it, and the docstring to render."""

  def __init__(self, name, obj, kind, doc, signature=None, methods=()):
    self.name = name
    self.obj = obj
    self.kind = kind                            # class | function | alias
    self.doc = doc
    self.signature = signature
    self.methods = list(methods)

  @property
  def title(self) -> str:
    if self.kind == "alias":
      return self.name
    if self.kind == "class":
      return f"class {self.name}{self.signature or ''}".replace("()", "")
    return f"{self.name}{self.signature or ''}"


P = inspect.Parameter


def _default(value) -> str:
  """Spell a default value the way a signature would be written by hand."""
  if isinstance(value, type) or inspect.isroutine(value):
    return value.__name__
  return repr(value)


def _signature(obj) -> str | None:
  """The call signature of `obj`, with annotations dropped for legibility.

  The annotations are a click away in the source, and spelling them here would
  push the interesting part of a heading -- the parameter names and their
  defaults -- off the end of the line.
  """
  try:
    parameters = list(inspect.signature(obj).parameters.values())
  except (TypeError, ValueError):
    return None

  kinds = [p.kind for p in parameters]
  bits: list[str] = []
  for i, p in enumerate(parameters):
    if p.kind is P.KEYWORD_ONLY and not {P.VAR_POSITIONAL, P.KEYWORD_ONLY} & set(kinds[:i]):
      bits.append("*")                    # the marker opening the keyword-only group
    prefix = {P.VAR_POSITIONAL: "*", P.VAR_KEYWORD: "**"}.get(p.kind, "")
    bits.append(f"{prefix}{p.name}" +
                (f"={_default(p.default)}" if p.default is not P.empty else ""))
    if p.kind is P.POSITIONAL_ONLY and P.POSITIONAL_ONLY not in kinds[i + 1:]:
      bits.append("/")
  return "(" + ", ".join(bits) + ")"


@functools.cache
def _aliases(module_name: str) -> dict[str, tuple[str | None, str]]:
  """A module's type aliases, mapped to their `#:` doc comment and definition.

  A `TypeAlias` can carry neither a docstring nor a useful `__module__`, so it
  is found by source and documented by the run of `#:` comments above it -- the
  one place a docstring will not reach, and the reason this generator reads
  source as well as objects.
  """
  path = ROOT / "pycute" / (module_name.replace(".", "/") + ".py")
  lines = path.read_text(encoding="utf-8").splitlines()

  docs: dict[str, tuple[str | None, str]] = {}
  for node in ast.parse("\n".join(lines)).body:
    if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
      continue
    comment: list[str] = []
    for line in reversed(lines[: node.lineno - 1]):
      if not line.startswith("#:"):
        break
      comment.append(line[2:].strip())
    docs[node.target.id] = (" ".join(reversed(comment)) or None,
                            ast.unparse(node.value))
  return docs


@functools.cache
def _declaration_order(module_name: str) -> dict[str, int]:
  """Source position of each top-level name, so the reference reads in order."""
  path = ROOT / "pycute" / (module_name.replace(".", "/") + ".py")
  tree = ast.parse(path.read_text(encoding="utf-8"))
  order: dict[str, int] = {}
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      order.setdefault(node.name, node.lineno)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
      order.setdefault(node.target.id, node.lineno)
  return order


def _methods(cls) -> list[Entry]:
  """The documented methods of `cls`, in source order.

  A method earns a place in the reference by having a docstring: that keeps the
  dispatch plumbing and the one-line dunders out, without needing a list of
  exceptions here that would drift in its own right.
  """
  out: list[Entry] = []
  for name, obj in vars(cls).items():
    if name in ("__init__", "__new__"):
      continue                                  # the class heading carries these
    if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
      continue                                  # private: described by the class
    func = obj.__func__ if isinstance(obj, classmethod) else obj
    func = func.fget if isinstance(func, property) else func
    if not (inspect.isfunction(func) and func.__doc__):
      continue
    out.append(Entry(f"{cls.__name__}.{name}", func, "function", func.__doc__,
                     _drop_self(_signature(func))))
  out.sort(key=lambda e: e.obj.__code__.co_firstlineno)
  return out


def _drop_self(signature: str | None) -> str | None:
  """Remove the bound `self` / `cls` parameter from a method signature."""
  if signature is None:
    return None
  inner = signature[1:-1]
  head, _, tail = inner.partition(", ")
  if head.split(":")[0].strip() in ("self", "cls"):
    return f"({tail})"
  return signature if head.strip() not in ("self", "cls") else "()"


@functools.cache
def collect() -> dict[str, list[Entry]]:
  """Every documented name, grouped by defining module and in source order."""
  groups: dict[str, list[Entry]] = {name: [] for name in MODULE_ORDER}

  for name in pycute.__all__:
    obj = getattr(pycute, name)
    module = getattr(obj, "__module__", "")
    if isinstance(module, str) and module.startswith("pycute."):
      short = module[len("pycute."):]
    else:
      short = _defining_module(name)             # a type alias: found by source
    if short not in groups:
      raise SystemExit(f"{name}: defined in unlisted module {short!r}")

    # A type alias is checked for first: it is callable, so it would otherwise
    # be mistaken for a function and rendered with `Union`'s own signature.
    if name in _aliases(short):
      doc, definition = _aliases(short)[name]
      groups[short].append(Entry(name, obj, "alias", doc, definition))
    elif inspect.isclass(obj):
      groups[short].append(
        Entry(name, obj, "class", obj.__doc__, _signature(obj), _methods(obj)))
    else:
      groups[short].append(
        Entry(name, obj, "function", obj.__doc__, _signature(obj)))

  for short in [m for m in MODULE_ORDER if m.startswith(UTIL_PREFIX)]:
    module = __import__(f"pycute.{short}", fromlist=["_"])
    for name in _util_exports(module):
      obj = getattr(module, name)
      groups[short].append(
        Entry(name, obj, "class" if inspect.isclass(obj) else "function",
              obj.__doc__, _signature(obj)))

  for short, entries in groups.items():
    order = _declaration_order(short)
    entries.sort(key=lambda e: order.get(e.name, 1 << 30))
  return groups


def _util_exports(module) -> list[str]:
  """A util module's public names: its `__all__`, or its public callables."""
  if hasattr(module, "__all__"):
    return list(module.__all__)
  return [n for n, v in vars(module).items()
          if not n.startswith("_") and callable(v)
          and getattr(v, "__module__", "") == module.__name__]


def _defining_module(name: str) -> str:
  """Find the module whose source assigns `name`, for objects without one."""
  for short in MODULE_ORDER:
    if name in _declaration_order(short):
      return short
  raise SystemExit(f"{name}: no module defines it")


# ---------------------------------------------------------------------------
# Test cross-references
# ---------------------------------------------------------------------------

def test_index() -> tuple[dict[str, list[Path]], list[tuple[Path, str]]]:
  """Map each module to the tests covering it, from the test docstrings.

  A test says what it covers in its own docstring -- `Unit tests for
  pycute.coalesce` -- so every `pycute.X` it names is resolved to the module
  that defines `X`, whether `X` is a module or an exported name. Only the
  docstring's opening paragraph becomes the summary; the rest is written for
  someone reading the test, not the reference.
  """
  name_module = {}
  for short, entries in collect().items():
    for entry in entries:
      name_module.setdefault(entry.name, short)

  covers: dict[str, list[Path]] = {}
  summaries: list[tuple[Path, str]] = []
  for path in sorted((ROOT / "test").glob("test_*.py")):
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    if not doc:
      continue
    opening = inspect.cleandoc(doc).split("\n\n")[0]
    summaries.append((path, " ".join(opening.split()).replace("``", "`")))
    # A reference may name a module (`pycute.util.draw_svg`) or an export
    # (`pycute.coalesce`); the longest dotted prefix that resolves wins.
    for ref in dict.fromkeys(re.findall(r"pycute\.([\w.]*\w)", doc)):
      while ref:
        if (short := ref if ref in MODULE_ORDER else name_module.get(ref)):
          if path not in covers.setdefault(short, []):
            covers[short].append(path)
          break
        ref = ref.rpartition(".")[0]
  return covers, summaries


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _anchor(title: str) -> str:
  """The GitHub heading anchor for a rendered heading."""
  text = _HEADING_CHARS.sub("", title.replace("`", "").strip().lower())
  return re.sub(r"\s+", "-", text)


def _starts_list(lines: list[str]) -> bool:
  """Whether these lines open a `-`/`--` prefixed list."""
  first = next((l for l in lines if l.strip()), "")
  return first.lstrip().startswith("-")


def _bullets(lines: list[str]) -> list[str]:
  """Render `-`/`--` prefixed lines as a markdown list."""
  out: list[str] = []
  for line in lines:
    if not line.strip():
      continue
    if line.lstrip().startswith("-"):
      out.append(re.sub(r"^-+\s*", "* ", line.lstrip()))
    else:
      out.append(f"  {line.strip()}")     # a continuation of the previous item
  return out


def _render_doc(doc: str | None, out: list[str]) -> None:
  """Render a docstring: prose as paragraphs, sections as labelled blocks."""
  if not doc:
    return
  body, sections = parse_docstring(doc)

  for paragraph in body:
    if not all(l.startswith("  ") or not l.strip() for l in paragraph):
      out += ["\n".join(paragraph).replace("::", ":"), ""]
    elif _starts_list(paragraph):
      out += [*_bullets(_dedent(paragraph)), ""]
    else:
      out += ["```", *_dedent(paragraph), "```", ""]   # an indented literal block

  for name, lines in sections:
    if name in CODE_SECTIONS:
      language = "python" if name == "Examples" else ""
      out += [f"*{name}:*", "", f"```{language}", *lines, "```", ""]
    elif _starts_list(lines):
      out += [f"*{name}:*", "", *_bullets(lines), ""]
    else:
      out += [f"*{name}:*", "", *lines, ""]


def render() -> str:
  """The whole reference, as markdown."""
  groups = collect()
  covers, summaries = test_index()

  out = [
    "# PyCuTe API Reference",
    "",
    "Every name in `pycute.__all__`, with its signature, its documentation, and",
    "links to the source and to the tests that exercise it.",
    "",
    "> **Generated file — do not edit.** Every word below comes from a docstring",
    "> in `pycute/`. Edit the docstring, then run",
    "> `python scripts/gen_api_reference.py`.",
    "",
    "The reference is organized by module:",
    "",
  ]
  for short, entries in groups.items():
    names = ", ".join(f"`{e.name}`" for e in entries)
    out.append(f"* [`{short}`](#module-{_anchor(short)}) — {names}")
  out += [
    "",
    "Each entry's *Pre-conditions* and *Post-conditions* are the formal contract",
    "its unit test asserts, and every *Examples* block is evaluated by",
    "`test/test_docstring_examples.py`, so both are true of the code as it is.",
    "",
  ]

  for short, entries in groups.items():
    source = f"pycute/{short.replace('.', '/')}.py"
    out += ["---", "", f"## Module: `{short}`", "",
            f"Source: [`{source}`](../{source})"]
    tests = covers.get(short, [])
    if tests:
      links = ", ".join(f"[`{p.name}`](../test/{p.name})" for p in tests)
      out.append(f"Tests: {links}")
    out.append("")

    module = __import__(f"pycute.{short}", fromlist=["_"])
    _render_doc(module.__doc__, out)

    for entry in entries:
      out += [f"### `{entry.title}`", ""]
      if entry.kind == "alias":
        out += ["```python", f"{entry.name}: TypeAlias = {entry.signature}", "```", ""]
      _render_doc(entry.doc, out)
      for method in entry.methods:
        out += [f"#### `{method.title}`", ""]
        _render_doc(method.doc, out)

  out += ["---", "", "## Tests", "",
          "Each test module states what it covers in its own docstring; this",
          "table is generated from those.", "",
          "| Test | What it checks |", "|---|---|"]
  for path, summary in summaries:
    out.append(f"| [`{path.name}`](../test/{path.name}) | {summary} |")
  out += [
    "",
    "Run them all with:",
    "",
    "```sh",
    "pytest",
    "```",
    "",
    "## Copyright",
    "",
    "Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
    "SPDX-License-Identifier: Apache-2.0",
  ]

  return re.sub(r"\n{3,}", "\n\n", "\n".join(out).rstrip()) + "\n"


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--check", action="store_true",
                      help="exit non-zero if the reference is out of date")
  args = parser.parse_args()

  text = render()
  if not args.check:
    API_REF.write_text(text, encoding="utf-8")
    print(f"Wrote {API_REF.relative_to(ROOT)}")
    return 0

  if API_REF.read_text(encoding="utf-8") == text:
    print(f"OK: {API_REF.relative_to(ROOT)} is up to date")
    return 0
  print(f"{API_REF.relative_to(ROOT)} is out of date; "
        f"re-run scripts/gen_api_reference.py")
  return 1


if __name__ == "__main__":
  sys.exit(main())
