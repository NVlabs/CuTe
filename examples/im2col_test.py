# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the `im2col` prototype.

The oracle is a direct N-D convolution: for every output position and every
stencil tap it reads the activation at the coordinate the convolution is defined
to read, treating out-of-range coordinates as zero. Nothing about `im2col`'s
layout algebra is reused, so agreement means the layout addresses exactly what
the definition says.

Run from the repository root:
    pytest examples/im2col_test.py
"""

import ctypes
import inspect
import itertools
import traceback
import unittest

import pycute
from pycute import *
from pycute.alg.ref import gemm
from scripts.gen_api_reference import examples
from test.test_docstring_examples import _run          # the repo's example evaluator

import examples.im2col as im2col_module
from examples.im2col import (OutOfBoundsAccessor, _im2col_layout, im2col,
                             im2col_coord, im2col_padded)


def row_major(shape_) -> Layout:
  """A compact layout over `shape_` with the last mode contiguous."""
  return make_ordered_layout(shape_, tuple(range(len(shape_) - 1, -1, -1)))


def iota(layout, value=lambda i: i) -> Tensor:
  """An int64 Tensor over `layout`, written in logical index order."""
  T = make_tensor(layout, dtype=ctypes.c_int64)
  for i in range(size(layout)):
    T[i] = value(i)
  return T


def zpq_extents(shape_dhw, stencil, lower_dhw, upper_dhw, stride_dhw, stride_trs):
  """
  The output extents, restated independently of `_im2col_layout` so that it has
  an oracle rather than a mirror. CUTLASS writes the same formula as
  `z = 1 + (d + 2*pad - ((t-1)*dilation + 1)) / stride`.
  """
  return tuple((d + lo + hi - ((t - 1) * s + 1)) // u + 1
               for d, t, lo, hi, u, s
               in zip(shape_dhw, stencil, lower_dhw, upper_dhw, stride_dhw, stride_trs))


def derived_zpq(A, stencil, **kw):
  """The `(Z,P,Q)` extents `_im2col_layout` derives, read back off its result."""
  layout, _ = _im2col_layout(A, stencil, **kw)
  return tuple(size(z) for z in layout[0][1])


def conv_reference(act, flt, stencil, *, lower_dhw, upper_dhw, stride_dhw, stride_trs):
  """
  `{(n, *zpq, k): value}` for a direct cross-correlation of `act` by `flt`.

  `act` is `(N, *DHW, C)` and `flt` is `(K, *TRS, C)`. Activation coordinates
  outside `act` contribute nothing, which is zero padding.
  """
  shape_dhw = shape(act)[1:-1]
  N, C, K = size[0](act), shape(act)[-1], size[0](flt)
  zpq = zpq_extents(shape_dhw, stencil, lower_dhw, upper_dhw, stride_dhw, stride_trs)

  out = {}
  for n, k in itertools.product(range(N), range(K)):
    for pos in itertools.product(*(range(z) for z in zpq)):
      total = 0
      for tap in itertools.product(*(range(t) for t in stencil)):
        crd = tuple(u * p + s * t - lo
                    for p, t, u, s, lo in zip(pos, tap, stride_dhw, stride_trs, lower_dhw))
        if all(0 <= c < d for c, d in zip(crd, shape_dhw)):
          for c in range(C):
            total += act[(n, *crd, c)] * flt[(k, *tap, c)]
      out[(n, *pos, k)] = total
  return out


class TestConvShape(unittest.TestCase):
  """The output extents `_im2col_layout` derives, against the formula restated
  in `zpq_extents` -- CUTLASS's
  `z = 1 + (d + 2*pad - ((t-1)*dilation + 1)) / stride`."""

  def test_matches_cutlass_formula(self):
    for d, t, pad, u, s in itertools.product(range(1, 10), range(1, 5), range(0, 3),
                                             range(1, 4), range(1, 4)):
      if (t - 1) * s + 1 > d + 2 * pad:
        continue                                    # stencil does not fit
      with self.subTest(d=d, t=t, pad=pad, u=u, s=s):
        got = derived_zpq(row_major((1, d, 1)), t,
                          lower_dhw=pad, stride_dhw=u, stride_trs=s)
        self.assertEqual(got, (1 + (d + 2 * pad - ((t - 1) * s + 1)) // u,))

  def test_defaults_are_per_mode_not_broadcast(self):
    A = row_major((1, 5, 5, 1))
    self.assertEqual(derived_zpq(A, (3, 3)), (3, 3))
    self.assertEqual(derived_zpq(A, (3, 3), lower_dhw=(1, 1)), (5, 5))
    self.assertEqual(derived_zpq(row_major((1, 5, 6, 7, 1)), (3, 3, 3),
                                 lower_dhw=(1, 1, 1)), (5, 6, 7))
    # Asymmetric padding: one extra row below only
    self.assertEqual(derived_zpq(row_major((1, 5, 1)), (2,),
                                 lower_dhw=1, upper_dhw=0), (5,))
    # An Integer is a rank-1 value, so it does not spread over three modes
    with self.assertRaisesRegex(ValueError, "rank 1"):
      derived_zpq(row_major((1, 5, 6, 7, 1)), (3, 3, 3), lower_dhw=1)

  def test_stencil_that_does_not_fit(self):
    with self.assertRaisesRegex(ValueError, "does not fit"):
      _im2col_layout(row_major((1, 3, 3, 1)), (5, 5))

  def test_spatial_rank_mismatch(self):
    with self.assertRaisesRegex(ValueError, "but expected"):
      _im2col_layout(row_major((1, 5, 5, 1)), (3, 3, 3))

  def test_explicit_shape_zpq_overrides_the_formula(self):
    """`shape_zpq` is how `dgrad` supplies extents the fprop formula does not
    produce; it is still rank-checked."""
    A = row_major((1, 6, 1))
    self.assertEqual(derived_zpq(A, (3,), shape_zpq=(4,)), (4,))
    with self.assertRaisesRegex(ValueError, "shape_zpq has rank"):
      _im2col_layout(A, (3,), shape_zpq=(4, 4))


class TestLayout(unittest.TestCase):
  """The shapes and strides `im2col` produces."""

  def setUp(self):
    self.A = make_tensor(row_major((1, 4, 4, 1)))          # (N,H,W,C)

  def test_shape_and_stride(self):
    B = im2col(self.A, (2, 2))
    self.assertEqual(shape(B), ((1, (3, 3)), ((2, 2), 1)))
    self.assertEqual(stride(B.layout), ((16, (4, 1)), ((4, 1), 1)))

  def test_traversal_stride_scales_the_position_modes_only(self):
    B = im2col(self.A, (2, 2), stride_dhw=(2, 2))
    self.assertEqual(stride(B.layout), ((16, (8, 2)), ((4, 1), 1)))

  def test_dilation_scales_the_tap_modes_only(self):
    B = im2col(self.A, (2, 2), stride_trs=(2, 2))
    self.assertEqual(stride(B.layout), ((16, (4, 1)), ((8, 2), 1)))

  def test_shares_the_activation_storage(self):
    self.assertIs(im2col(self.A, (2, 2)).accessor.base, self.A.accessor.base)

  def test_is_not_injective(self):
    """Every interior element is read once per window containing it, which is
    what materializing `im2col` would pay for in memory."""
    B = im2col(self.A, (2, 2))
    self.assertEqual(size(B), 36)
    self.assertEqual(int(coshape(B.layout)), 16)

  def test_unit_stencil_is_the_activation(self):
    """A 1x1 stencil collapses the tap modes, so `im2col` degenerates to the
    activation itself and a convolution is a plain GEMM."""
    B = im2col(self.A, (1, 1))
    self.assertEqual(shape(B), ((1, (4, 4)), ((1, 1), 1)))
    self.assertEqual(size(B), size(self.A))
    self.assertEqual(int(coshape(B.layout)), int(coshape(self.A.layout)))

  def test_rank_1_and_rank_3_spatial(self):
    for shape_A, stencil, want in [((2, 8, 3), 3, ((2, (6,)), ((3,), 3))),
                                   ((2, 4, 5, 6, 3), (2, 3, 4),
                                    ((2, (3, 3, 3)), ((2, 3, 4), 3)))]:
      with self.subTest(shape_A):
        self.assertEqual(shape(im2col(make_tensor(row_major(shape_A)), stencil)), want)

  def test_too_few_modes(self):
    with self.assertRaisesRegex(ValueError, "at least"):
      im2col(make_tensor(row_major((4, 4))), 2)

  def test_hierarchical_channel_mode_passes_through(self):
    """`N` and `C` are only carried across, so a channel mode already split into
    vectors survives."""
    A = make_tensor(Layout((2, 4, 4, (2, 8)), (256, 64, 16, (8, 1))))
    self.assertEqual(shape(im2col(A, (2, 2))), ((2, (3, 3)), ((2, 2), (2, 8))))

  def test_a_renesting_that_coalesces_gives_the_flat_view(self):
    """`composition` reads each spatial mode through the activation's own layout
    rather than scaling one integer stride, so a mode split into a hierarchy that
    coalesces -- `(2,4):(1,2)` is `8:1` -- composes to exactly the flat view."""
    flat = iota(Layout((1, 4, 8, 1), (32, 8, 1, 1)))
    hier = iota(Layout((1, 4, (2, 4), 1), (32, 8, (1, 2), 1)))
    for stencil in [(2, 2), (3, 3), (2, 4)]:
      with self.subTest(stencil=stencil):
        want, got = im2col(flat, stencil), im2col(hier, stencil)
        self.assertEqual(shape(got), shape(want))
        self.assertEqual(stride(got.layout), stride(want.layout))
        self.assertEqual([got[i] for i in range(size(got))],
                         [want[i] for i in range(size(want))])

  def test_a_blocked_spatial_mode_keeps_its_blocking(self):
    """`(2,4):(1,8)` is two blocks of four with a gap, so it has no flat
    equivalent. The window origins come back blocked to match, and a dilation of
    2 lands on the outer block index, the inner extent being 2."""
    A = iota(Layout((1, (2, 4), 1), (32, (1, 8), 1)))
    B = im2col(A, (2,), stride_trs=(2,))          # dilation 2 divides the blocking
    self.assertEqual(shape(B), ((1, ((2, 3),)), ((2,), 1)))
    self.assertEqual(stride(B.layout), ((32, ((1, 8),)), ((8,), 1)))
    for w, t in itertools.product(range(2), range(3)):
      for tap in range(2):
        self.assertEqual(B[(0, ((w, t),)), ((tap,), 0)], A[0, (w, t + tap), 0])

  def test_both_views_admit_a_blocked_spatial_mode(self):
    """One basis per top-level mode scales through a hierarchy the way an integer
    stride does, so the coordinate view admits whatever the memory view does and
    names the same elements -- at the same size, though basis strides coalesce
    where the activation's cannot, so the grouping of `(Z,P,Q)` can differ."""
    A = iota(Layout((1, (2, 4), 1), (32, (1, 8), 1)))
    for stencil, kw in [((2,), dict(stride_trs=(2,))),
                        ((3,), dict(stride_trs=(2,))),
                        ((4,), dict(stride_dhw=(2,)))]:
      with self.subTest(stencil=stencil, **kw):
        B, C = im2col(A, stencil, **kw), im2col_coord(A, stencil, **kw)
        self.assertEqual((size[0](C), size[1](C)), (size[0](B), size[1](B)))
        self.assertEqual([B[i] for i in range(size(B))],
                         [A[idx2crd(C[i], shape(A))] for i in range(size(C))])

  def test_matches_cutlass_example_59(self):
    """CUTLASS example 59 writes the same 3-D transform by hand, for an NDHWC
    activation with unit traversal stride and dilation and no padding:

        make_shape (make_shape (      N,     Z,   P, Q), make_shape (  C,     T,   R, S)),
        make_stride(make_stride(D*H*W*C, H*W*C, W*C, C), make_stride(  1, H*W*C, W*C, C))

    The same modes and the same strides, with the reduction group in the
    activation's own order -- `((T,R,S),C)` where example 59 writes `(C,(T,R,S))`
    -- so the two differ by a permutation of `K`'s modes and nothing else.
    """
    N, D, H, W, C = 2, 4, 5, 6, 3
    T, R, S = 2, 3, 2
    Z, P, Q = D - T + 1, H - R + 1, W - S + 1

    got = im2col(make_tensor(row_major((N, D, H, W, C))), (T, R, S))
    self.assertEqual(shape(got), ((N, (Z, P, Q)), ((T, R, S), C)))
    self.assertEqual(stride(got.layout),
                     ((D * H * W * C, (H * W * C, W * C, C)),
                      ((H * W * C, W * C, C), 1)))


class TestCoord(unittest.TestCase):
  """The coordinate view, and its agreement with the offset view."""

  def test_reads_the_expected_activation_coordinate(self):
    A = make_tensor(row_major((1, 4, 4, 1)))
    B = im2col_coord(A, (2, 2))
    self.assertEqual(idx2crd(B[(0, (0, 0)), ((0, 0), 0)], shape(A)), (0, 0, 0, 0))
    self.assertEqual(idx2crd(B[(0, (1, 2)), ((1, 1), 0)], shape(A)), (0, 2, 3, 0))

  def test_padding_shows_up_as_a_negative_coordinate(self):
    A = make_tensor(row_major((1, 4, 4, 1)))
    B = im2col_coord(A, (3, 3), lower_dhw=(1, 1))
    self.assertEqual(idx2crd(B[(0, (0, 0)), ((0, 0), 0)], shape(A)), (0, -1, -1, 0))
    self.assertFalse(in_bounds(B[(0, (0, 0)), ((0, 0), 0)], shape(A)))
    self.assertTrue(in_bounds(B[(0, (1, 1)), ((1, 1), 0)], shape(A)))

  def test_origin_lands_in_each_views_codomain(self):
    """The origin is the activation's layout evaluated at the corner coordinate,
    so it comes back as an offset for the memory view and as an `ArithTuple`
    coordinate for the coordinate view."""
    A = make_tensor(Layout((1, 4, 4, 1), (16, 4, 1, 1)))
    off = im2col(A, (3, 3), lower_dhw=(1, 1))
    crd = im2col_coord(A, (3, 3), lower_dhw=(1, 1))
    # (0,-1,-1,0) through row-major NHWC strides is -1*4 + -1*1
    self.assertEqual((off.accessor.address - A.accessor.address)
                     // ctypes.sizeof(ctypes.c_double), -5)
    self.assertEqual(idx2crd(crd[(0, (0, 0)), ((0, 0), 0)], shape(A)), (0, -1, -1, 0))

  def test_agrees_with_the_offset_view_wherever_in_bounds(self):
    """The two views are the same layout over different codomains, so wherever a
    window is inside the activation the offset view must load the element the
    coordinate view names."""
    for shape_A, stencil, kw in [
        ((2, 5, 6, 3), (2, 3), {}),
        ((2, 5, 6, 3), (2, 3), dict(stride_dhw=(2, 2))),
        ((1, 7, 7, 2), (3, 3), dict(stride_trs=(2, 2))),
        ((1, 7, 7, 2), (3, 3), dict(lower_dhw=(1, 1))),
        ((2, 4, 5, 6, 2), (2, 2, 2), dict(stride_dhw=(1, 2, 1), stride_trs=(1, 1, 2))),
    ]:
      with self.subTest(shape_A=shape_A, stencil=stencil, **kw):
        A = iota(row_major(shape_A), lambda i: i + 1)
        off, crd = im2col(A, stencil, **kw), im2col_coord(A, stencil, **kw)
        self.assertEqual(size(off), size(crd))
        for i in range(size(off)):
          coord = idx2crd(crd[i], shape(A))
          if in_bounds(coord, shape(A)):
            self.assertEqual(off[i], A[coord], f"at flat index {i}, coord {coord}")


class TestPadded(unittest.TestCase):
  """The bounds-checking accessor."""

  def test_fills_outside_and_reads_inside(self):
    A = iota(row_major((1, 3, 3, 1)), lambda i: i + 1)
    B = im2col_padded(A, (3, 3), lower_dhw=(1, 1))
    self.assertEqual(B[(0, (1, 1)), ((1, 1), 0)], A[0, 1, 1, 0])   # centre of centre
    self.assertEqual(B[(0, (0, 0)), ((0, 0), 0)], 0)               # padded corner
    self.assertEqual(im2col_padded(A, (3, 3), lower_dhw=(1, 1), fill=-7)
                     [(0, (0, 0)), ((0, 0), 0)], -7)

  def test_slicing_keeps_the_bounds_check(self):
    A = iota(row_major((1, 3, 3, 1)), lambda i: i + 1)
    B = im2col_padded(A, (3, 3), lower_dhw=(1, 1))
    row = B[(0, (0, 0)), None]                                     # the corner window
    # k runs colex over ((R,S),C) so the row index r varies fastest, and the
    # window sits at h,w = r-1, s-1: the r=0 row and the s=0 column are padding.
    self.assertEqual([row[k] for k in range(size(row))],
                     [0, 0,               0,                        # w = -1
                      0, A[0, 0, 0, 0],   A[0, 1, 0, 0],            # w =  0
                      0, A[0, 0, 1, 0],   A[0, 1, 1, 0]])           # w =  1

  def test_hierarchical_channel_mode_is_indexed_as_one_mode(self):
    """`idx2crd` returns a coordinate congruent to the activation, so a
    hierarchical channel mode reaches the tensor as one nested mode rather than
    flattened into its leaves."""
    A = iota(Layout((1, 3, 3, (2, 2)), (36, 12, 4, (2, 1))), lambda i: i + 1)
    B = im2col_padded(A, (3, 3), lower_dhw=(1, 1))
    self.assertEqual(shape(B), ((1, (3, 3)), ((3, 3), (2, 2))))
    self.assertEqual(B[(0, (1, 1)), ((1, 1), (1, 1))], A[0, 1, 1, (1, 1)])
    self.assertEqual(B[(0, (0, 0)), ((0, 0), (0, 0))], 0)     # padded corner

  def test_accessor_directly(self):
    A = make_tensor(row_major((2, 2)))
    A[1, 1] = 5.0
    self.assertEqual(OutOfBoundsAccessor(A)[(1, 1)], 5.0)
    self.assertEqual(OutOfBoundsAccessor(A)[(2, 0)], 0)
    self.assertEqual(OutOfBoundsAccessor(A, fill=-1)[(-1, 0)], -1)


class TestConvolution(unittest.TestCase):
  """`gemm(im2col(act), flt, out)` against a direct convolution."""

  def _check(self, N, C, shape_dhw, K, stencil, **kw):
    num_spatial = len(shape_dhw)
    def per(value):
      return value if isinstance(value, tuple) else (value,) * num_spatial
    stencil_t = per(stencil)
    lower_t   = per(kw.get("lower_dhw", 0))
    upper_t   = lower_t if kw.get("upper_dhw") is None else per(kw["upper_dhw"])
    u_t       = per(kw.get("stride_dhw", 1))
    s_t       = per(kw.get("stride_trs", 1))

    act = iota(row_major((N, *shape_dhw, C)), lambda i: i % 7 - 3)
    flt = iota(row_major((K, *stencil_t, C)), lambda i: i % 5 - 2)
    zpq = zpq_extents(shape_dhw, stencil_t, lower_t, upper_t, u_t, s_t)
    out = make_tensor(row_major((N, *zpq, K)), dtype=ctypes.c_int64)

    # ((N,(Z,P,Q)), ((T,R,S),C)) x (K, ((T,R,S),C)) -> ((N,(Z,P,Q)), K)
    padded = any(lo or hi for lo, hi in zip(lower_t, upper_t))
    A_mk = im2col_padded(act, stencil_t, **kw) if padded else im2col(act, stencil_t, **kw)
    B_nk = make_layout([flt.layout[0],
                        make_layout([make_layout(flt.layout[i] for i in
                                                 range(1, 1 + num_spatial)),
                                     flt.layout[-1]])])
    C_mn = make_layout([make_layout([out.layout[0],
                                     make_layout(out.layout[i] for i in
                                                 range(1, 1 + num_spatial))]),
                        out.layout[-1]])

    gemm(A_mk, Tensor(flt.accessor, B_nk), Tensor(out.accessor, C_mn))

    want = conv_reference(act, flt, stencil_t, lower_dhw=lower_t, upper_dhw=upper_t,
                          stride_dhw=u_t, stride_trs=s_t)
    for crd, value in want.items():
      self.assertEqual(out[crd], value, f"at {crd}")

  def test_2d_single_channel(self):
    self._check(1, 1, (4, 4), 1, (2, 2))

  def test_2d_multi_channel_batched(self):
    self._check(2, 3, (5, 6), 4, (2, 3))

  def test_2d_traversal_stride(self):
    self._check(2, 2, (7, 7), 3, (3, 3), stride_dhw=(2, 2))

  def test_2d_dilation(self):
    self._check(1, 2, (7, 7), 2, (3, 3), stride_trs=(2, 2))

  def test_2d_padding(self):
    self._check(2, 2, (5, 5), 3, (3, 3), lower_dhw=(1, 1))

  def test_2d_asymmetric_padding(self):
    self._check(1, 1, (5, 5), 2, (2, 2), lower_dhw=(1, 1), upper_dhw=(0, 0))

  def test_2d_padding_stride_and_dilation(self):
    self._check(2, 2, (9, 9), 2, (3, 3),
                lower_dhw=(2, 2), stride_dhw=(2, 2), stride_trs=(2, 2))

  def test_1d(self):
    self._check(2, 3, (8,), 4, (3,))

  def test_1d_padded(self):
    self._check(1, 2, (8,), 2, (3,), lower_dhw=1)

  def test_3d(self):
    self._check(2, 2, (4, 5, 6), 3, (2, 3, 2))

  def test_3d_mixed_parameters(self):
    self._check(1, 2, (6, 7, 8), 2, (2, 3, 2),
                lower_dhw=(0, 1, 1), stride_dhw=(1, 2, 1), stride_trs=(1, 1, 2))

  def test_1x1_is_a_plain_gemm(self):
    self._check(2, 4, (5, 5), 3, (1, 1))


class TestValidation(unittest.TestCase):
  """Every parameter is checked by the public view the caller invoked, before any
  layout is built, so all three views reject the same arguments identically."""

  VIEWS = (im2col, im2col_coord, im2col_padded)

  def test_all_views_reject_the_same_arguments(self):
    A = make_tensor(row_major((2, 4, 4, 3)))
    for view in self.VIEWS:
      for label, kw, expected in [
          ("stencil rank",    dict(stencil=(2, 2, 2)),                   "but expected"),
          ("stencil scalar",  dict(stencil=2),                           "but expected"),
          ("lower_dhw rank",  dict(stencil=(2, 2), lower_dhw=(1, 1, 1)), "but expected"),
          ("lower_dhw scalar", dict(stencil=(2, 2), lower_dhw=1),        "but expected"),
          ("stride_trs rank", dict(stencil=(2, 2), stride_trs=(1, 1, 1)), "but expected"),
          ("shape_zpq rank",  dict(stencil=(2, 2), shape_zpq=(1,)),      "but expected"),
          ("stencil too big", dict(stencil=(9, 9)),                      "does not fit"),
      ]:
        with self.subTest(view=view.__name__, case=label):
          with self.assertRaisesRegex(ValueError, expected):
            view(A, **kw)

  def test_an_integer_is_a_rank_1_value_not_a_broadcast(self):
    """Without broadcasting an Integer is admissible exactly where its rank
    fits: one spatial mode, for which it is the rank-1 tuple it denotes."""
    A = make_tensor(row_major((1, 8, 1)))
    self.assertEqual(shape(im2col(A, 3, lower_dhw=1)),
                     shape(im2col(A, (3,), lower_dhw=(1,))))

  def test_one_checker_serves_every_view(self):
    """Normalization lives in `_im2col_layout` alone, so the views cannot drift:
    one bad argument yields one message whichever view was called, and the
    traceback still names the view the caller invoked."""
    A = make_tensor(row_major((2, 4, 4, 3)))
    messages = set()
    for view in self.VIEWS:
      with self.subTest(view=view.__name__):
        try:
          view(A, (2, 2, 2))
        except ValueError as e:
          messages.add(str(e))
          frames = [f.name for f in traceback.extract_tb(e.__traceback__)]
          self.assertIn(view.__name__, frames)
          self.assertIn("_im2col_layout", frames)
        else:
          self.fail("expected a ValueError")
    self.assertEqual(len(messages), 1, f"views disagree: {messages}")


class TestDocstrings(unittest.TestCase):
  """`test/test_docstring_examples.py` walks the `pycute` package only, so borrow
  its evaluator and run this module's examples here."""

  def _documented(self):
    yield "im2col (module)", im2col_module.__doc__
    for name, obj in vars(im2col_module).items():
      if name.startswith("_") or not (inspect.isclass(obj) or callable(obj)):
        continue
      if getattr(obj, "__module__", None) == im2col_module.__name__ and obj.__doc__:
        yield name, obj.__doc__

  def test_examples(self):
    total = 0
    for name, doc in self._documented():
      namespace = dict(vars(pycute), **vars(im2col_module))
      for source in examples(doc or ""):
        with self.subTest(f"{name}: {source}"):
          _run(source, namespace)
        total += 1
    self.assertGreater(total, 20, f"only {total} examples found; parsing likely broke")


class TestDgrad(unittest.TestCase):
  """`lower_trs` with a negative `stride_trs` walks the stencil backwards, which
  is the access pattern `dgrad` needs."""

  def test_reads_the_transposed_stencil(self):
    A = make_tensor(row_major((1, 6, 1)))                # (N,W,C)
    T = 3
    B = im2col_coord(A, T, lower_trs=T - 1, stride_trs=-1, shape_zpq=6, lower_dhw=T - 1)
    # coordinate == z - t, the reversed correlation dgrad performs
    for z, t in itertools.product(range(6), range(T)):
      self.assertEqual(idx2crd(B[(0, (z,)), ((t,), 0)], shape(A))[1], z - t)


if __name__ == "__main__":
  unittest.main()
