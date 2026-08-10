# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Construct rearranged Layouts, Tensors and HTuples with named modes.

Each side of an "in_modes->out_modes" expression denotes a profile whose leaves
are named modes. The input side names source modes and may descend into nested
modes with parentheses. The output side reorders, regroups, repeats, or drops
those names. Source top-level modes left unnamed are appended unchanged.
"""

from pycute import *


def _parse_modes(expression : str) -> Profile:
  """
  Parse a mode expression into the profile it denotes, with one-character names
  at the leaves.

  Each alphanumeric character names one mode and parentheses group modes into a
  hierarchy. Whitespace is ignored. An expression that names a single mode *is* 
  that mode rather than a tuple holding it, so each pair of
  parentheses around a lone name nests that name one level deeper.

  Examples:
    _parse_modes("i")        == "i"
    _parse_modes("ij")       == ("i", "j")
    _parse_modes("(i)")      == ("i",)
    _parse_modes("((i))")    == (("i",),)
    _parse_modes("(ij)")     == (("i", "j"),)
    _parse_modes("(ij)k")    == (("i", "j"), "k")
    _parse_modes("((ab)c)d") == ((("a", "b"), "c"), "d")
  """
  chars = (ch for ch in expression if not ch.isspace())

  def parse(grouped : bool) -> tuple:
    modes = []
    for ch in chars:
      if ch == '(':
        modes.append(parse(True))
      elif ch == ')':
        if not grouped:
          raise ValueError(f"einfold: unmatched ')' in {expression!r}")
        if not modes:
          raise ValueError(f"einfold: empty mode group in {expression!r}")
        return tuple(modes)
      elif ch == ',':
        raise ValueError(f"einfold: separators between modes are implied, "
                         f"so ',' is not allowed in {expression!r}")
      elif not ch.isalnum():
        raise ValueError(f"einfold: invalid mode name {ch!r} in {expression!r}")
      else:
        modes.append(ch)
    if grouped:
      raise ValueError(f"einfold: unclosed '(' in {expression!r}")
    return tuple(modes)

  modes = parse(False)
  if not modes:
    raise ValueError(f"einfold: empty mode expression {expression!r}")
  return modes[0] if len(list(leaves(modes))) == 1 else modes


def einfold(subscripts: str, value: Layout | Tensor | HTuple) -> Layout | Tensor | HTuple:
  """
  Return `value` with its modes named, reordered and regrouped by `subscripts`
  -- a generalized transpose.

  `subscripts` has the form "in_modes->out_modes", where each side denotes a
  profile whose leaves are named modes (see `_parse_modes`). Output names
  must come from the input: repeating a name duplicates that mode, and omitting a
  name drops it. If the input lists only a prefix of the source's top-level modes,
  the unnamed suffix is appended unchanged.

  Because a side denotes a profile, a lone name is not a top-level mode but the
  whole source -- "i->i" is the identity for any source, "i->(i)" wraps the
  source in one mode, and "(i)->i" unwraps a rank-1 source. Every other group is
  one mode of the level containing it, at the top level as much as within one, so
  "ij->(ij)" folds the two named modes together just as "ijk->(ij)k" does.

  Layouts produce Layouts and Tensors produce Tensors that share the source's
  accessor, so no data is copied. Anything else is matched against its `profile`
  and rebuilt as tuples, which needs nothing of its leaves -- so a Shape, a
  Stride, a Coord with `None` markers or a Tiler of Layouts all fold, and folding
  commutes with reading a Layout's parts:
    einfold(subscripts, shape(A))  == shape(einfold(subscripts, A))
    einfold(subscripts, stride(A)) == stride(einfold(subscripts, A))

  Examples for a Tensor with shape ((12, 4), 42, 5, 7):
    shape(einfold("(ab)cde -> c(ade)b", tensor)) == (42, (12, 5, 7), 4)
    shape(einfold("ijkm -> imkj", tensor))       == ((12, 4), 7, 5, 42)
    shape(einfold("ij -> ji", tensor))           == (42, (12, 4), 5, 7)
    shape(einfold("ijk -> ik", tensor))          == ((12, 4), 5, 7)
    shape(einfold("ij -> (ij)", tensor))         == (((12, 4), 42), 5, 7)
    shape(einfold("i -> (i)", tensor))           == (((12, 4), 42, 5, 7),)

  The same expressions fold that Shape, or any other HTuple, on its own:
    einfold("ij -> ji", ((12, 4), 42, 5, 7))     == (42, (12, 4), 5, 7)
    einfold("ij -> ji", (None, 42))              == (42, None)
  """
  if is_tensor(value):
    return Tensor(value.accessor, einfold(subscripts, value.layout))

  # A Layout rebuilds its mode hierarchy with sublayouts, any other HTuple with
  # tuples. Either way it is the profile that the input is matched against.
  maker = make_layout if is_layout(value) else tuple
  source_profile = profile(value)

  # Parse the expression "in_modes->out_modes" into a mode tree per side.
  parts = subscripts.split("->")
  if len(parts) != 2:
    raise ValueError(f"einfold: expected exactly one '->' in {subscripts!r}")
  input_modes, output_modes = parts
  input_tree  = _parse_modes(input_modes)
  output_tree = _parse_modes(output_modes)

  input_labels = list(leaves(input_tree))
  input_set    = set(input_labels)
  output_set   = set(leaves(output_tree))
  if len(input_set) != len(input_labels):
    raise ValueError(f"einfold: input contains repeated names: {input_modes!r}")
  unknown = sorted(output_set - input_set)
  if unknown:
    raise ValueError(f"einfold: output contains unknown names: {unknown}")

  # An input that lists top-level modes may list only a prefix of them. Name the
  # trailing modes it leaves unnamed -- by position, so that these names cannot
  # collide with parsed ones -- and echo them out. An input that is a lone name
  # binds the whole source, so it leaves nothing over.
  if is_tuple(input_tree) and is_tuple(source_profile):
    passthrough = tuple(range(len(input_tree), len(source_profile)))
    if passthrough:
      input_tree  += passthrough
      output_tree  = wrap(output_tree) + passthrough

  # Each name binds one whole source mode, so the input must coarsen the profile.
  # This also rejects an input that lists more top-level modes than the source.
  if not weakly_congruent(input_tree, source_profile):
    raise ValueError(f"einfold: input {input_modes!r} does not match the "
                     f"source profile {source_profile}")

  # Names bind the source's modes at the leaves of the input,
  # and the output rebuilds the mode hierarchy from them.
  bindings = dict(zip_leaves(input_tree, value))
  return transform_apply_leaf(maker, bindings.get, output_tree)
