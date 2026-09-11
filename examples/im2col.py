# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The `im2col` transformation as a CuTe Layout, for N-D convolution.

A convolution applies a stencil to every position of an activation tensor. The
`im2col` transformation is the change of view that turns that into a GEMM:

    activation  (N, [D,] [H,] [W,] C)
    im2col      ((N,(Z,P,Q)), ((T,R,S),C))    logical (M,K)
    stencil     ( K,          ((T,R,S),C))    logical (N,K)
    output      ((N,(Z,P,Q)),  K         )    logical (M,N)

The transformation itself composes the activation's own spatial sublayout twice:

  * take the activation's spatial modes, `A[1:-1]`;
  * compose each with a position layout `Layout(Z, stride_dhw)` to get the
    output-position modes (Z,P,Q), and with a tap layout `Layout(T, stride_trs)`
    to get the stencil-tap modes (T,R,S);
  * shift the origin to the lower corner, which is where padding lives.

A spatial coordinate comes out as `stride_dhw*z + stride_trs*t + corner` because
both modes walk the same axis. That is also why the result is not injective:
each activation element is read once per window that contains it.

`_im2col_layout` is the primitive: it checks the parameters, derives the output
extents `(Z,P,Q)`, and returns the Layout together with the origin its codomain
must be shifted by. Three views wrap it, differing only in what the layout maps
*into* -- each hands it a different activation layout and pairs the result with
a different accessor:

  * `im2col`        -- offsets into the activation's memory. What GEMM wants,
                       and exact only when no window leaves the activation.
  * `im2col_coord`  -- coordinates of the activation, as `ArithTuple`s. This is
                       the `ArithmeticTupleIterator` the TMA descriptor builds,
                       and what predication needs.
  * `im2col_padded` -- coordinates, read through a bounds-checking accessor that
                       returns `fill` outside the activation. The software
                       equivalent of TMA's out-of-bounds fill, so padded
                       convolutions run through an unmodified GEMM.

Parameter names follow CUTLASS's convolution vocabulary.
Every tuple is in the **activation's** order and every name carries that order
as its suffix: `_dhw` for a parameter indexed by spatial mode, `_trs` for one
indexed by stencil tap. The two run in step, so `stride_dhw[i]` and `stride_trs[i]`
scale the same axis.
"""

from __future__ import annotations

from pycute import *


# ------------------------------------------------------------------- the map

def _im2col_layout(A : Layout,          # (N, [D,] [H,] [W,] C)
                   stencil,             # ([T,] [R,] [S])
                   *,
                   lower_dhw = None,    # padding before each spatial mode; None means 0
                   upper_dhw = None,    # padding after;  None means symmetric
                   stride_dhw = None,   # traversal stride;  None means 1
                   lower_trs = None,    # stencil origin offset;  None means 0
                   stride_trs = None,   # dilation;  None means 1
                   shape_zpq = None):   # output extents, when not the fprop formula
  """
  The `((N,(Z,P,Q)), ((T,R,S),C))` Layout over an activation whose layout is
  `A`, together with the origin its codomain must be shifted by.

  `A[0]` and `A[-1]` carry over verbatim, so a hierarchical image or channel
  mode survives; only the spatial modes are rebuilt, by composing them with
  tilers, and a spatial mode may itself be hierarchical.

  Every parameter carries one entry per spatial mode and
  `None` declares the parameter's default: no padding, unit
  traversal stride and dilation, and `lower_trs` at zero. `upper_dhw = None`
  instead means padding is symmetric, and `shape_zpq = None` derives the output
  extents per spatial mode, with `T` the stencil extent, `p_lo`/`p_hi` the
  padding, `u` the traversal stride and `s` the dilation:

      Z = 1 + (D + p_lo + p_hi - ((T - 1)*s + 1)) // u

  the count of positions at which a dilated stencil of `T` taps fits inside the
  padded activation, taking every `u`-th one.

  Pre-conditions:
    rank(A) >= 3
    Each parameter is `None`, an Integer for a single spatial
    mode, or a tuple of the activation's spatial rank
    Every derived extent is positive; a stencil that does not fit raises
    Each spatial mode of `A` satisfies composition's divisibility conditions
    against `Layout(Z, stride_dhw)` and `Layout(T, stride_trs)`

  Post-conditions:
    result[(n,(z..)), ((t..),c)] == A[n, (stride_dhw*z + stride_trs*t)..,  c]
    shifted by the returned origin
  """
  num_spatial = rank(A) - 2
  if num_spatial < 1:
    raise ValueError(f"im2col: activation {shape(A)} needs at least (N, W, C)")

  stencil = wrap(stencil)
  if len(stencil) != num_spatial:
    raise ValueError(f"im2col: stencil has rank {len(stencil)} but expected {num_spatial}")

  lower_dhw = (0,) * num_spatial if lower_dhw is None else wrap(lower_dhw)
  if len(lower_dhw) != num_spatial:
    raise ValueError(f"im2col: lower_dhw has rank {len(lower_dhw)} but expected {num_spatial}")

  upper_dhw = lower_dhw if upper_dhw is None else wrap(upper_dhw)
  if len(upper_dhw) != num_spatial:
    raise ValueError(f"im2col: upper_dhw has rank {len(upper_dhw)} but expected {num_spatial}")

  stride_dhw = (1,) * num_spatial if stride_dhw is None else wrap(stride_dhw)
  if len(stride_dhw) != num_spatial:
    raise ValueError(f"im2col: stride_dhw has rank {len(stride_dhw)} but expected {num_spatial}")

  lower_trs = (0,) * num_spatial if lower_trs is None else wrap(lower_trs)
  if len(lower_trs) != num_spatial:
    raise ValueError(f"im2col: lower_trs has rank {len(lower_trs)} but expected {num_spatial}")

  stride_trs = (1,) * num_spatial if stride_trs is None else wrap(stride_trs)
  if len(stride_trs) != num_spatial:
    raise ValueError(f"im2col: stride_trs has rank {len(stride_trs)} but expected {num_spatial}")

  A_dhw = A[1:-1]                                   # the spatial modes, as one Layout

  if shape_zpq is None:
    # A spatial mode may be hierarchical, so use `size`.
    shape_zpq = tuple(1 + (size(A_dhw[i]) + lower_dhw[i] + upper_dhw[i]
                           - ((stencil[i] - 1) * stride_trs[i] + 1)) // stride_dhw[i]
                      for i in range(num_spatial))
  else:
    shape_zpq = wrap(shape_zpq)
  if len(shape_zpq) != num_spatial:
    raise ValueError(f"im2col: shape_zpq has rank {len(shape_zpq)} but expected {num_spatial}")
  if any(z <= 0 for z in shape_zpq):
    raise ValueError(f"im2col: stencil {stencil} dilated by {stride_trs} does not fit in "
                     f"{shape(A_dhw)} padded by {lower_dhw}/{upper_dhw}: extent {shape_zpq}")

  # Both halves compose the same spatial sublayout; only the tiler differs, and
  # its stride is the traversal stride for one and the dilation for the other.
  tiler_zpq = tuple(Layout(s, d) for s, d in zip(shape_zpq, stride_dhw))
  tiler_trs = tuple(Layout(s, d) for s, d in zip(stencil,   stride_trs))

  layout = make_layout([
    make_layout([A[ 0], composition(A_dhw, tiler_zpq)]),  # (N,(Z,P,Q))
    make_layout([composition(A_dhw, tiler_trs), A[-1]]),  # ((T,R,S),C)
  ])

  # The first window starts `lower_dhw` before the activation's origin and the
  # first tap a further `lower_trs` along, so the codomain shifts by wherever `A`
  # sends that spatial coordinate.
  origin = A(ArithTuple(0, *lower_trs, 0) - ArithTuple(0, *lower_dhw, 0))
  return layout, origin


# ------------------------------------------------------------------ the views

def im2col(A : Tensor,          # (N, [D,] [H,] [W,] C)
           stencil,             # ([T,] [R,] [S])
           *,
           lower_dhw = None,    # padding before each spatial mode; None means 0
           upper_dhw = None,    # padding after;  None means symmetric
           stride_dhw = None,   # traversal stride;  None means 1
           lower_trs = None,    # stencil origin offset;  None means 0
           stride_trs = None,   # dilation;  None means 1
           shape_zpq = None,    # output extents, when not the fprop formula
           ) -> Tensor:         # ((N,(Z,P,Q)), ((T,R,S),C)) -- logical (M,K)
  """
  The `im2col` view of activation `A`: a rank-2 `(M,K)` tensor, sharing `A`'s
  memory, that a GEMM can contract against a `(K,((T,R,S),C))` stencil tensor.

  `A`'s spatial rank is `rank(A) - 2`, and every parameter below carries that
  many entries: nothing is broadcast, and `None` takes the default at every mode.

  What CUTLASS calls each of them:

  | parameter    | CUTLASS name                |
  |--------------|-----------------------------|
  | `stencil`    | `shape_B[1:-1]`             |
  | `lower_dhw`  | `lower_padding`             |
  | `upper_dhw`  | `upper_padding`             |
  | `stride_dhw` | `traversal_stride`          |
  | `stride_trs` | `dilation`                  |
  | `lower_trs`  | TMA's `lower_srt`, reversed |

  `lower_trs` and a negative `stride_trs` are the freedom `dgrad` needs: the
  spatial coordinate this view reads is

      stride_dhw*z + stride_trs*t + lower_trs - lower_dhw

  so `fprop` leaves `lower_trs` at `0`, while `dgrad` walks the stencil
  backwards with `lower_trs = (T-1)*dilation` and `stride_trs = -dilation`.
  `shape_zpq` overrides the derived output extents for those cases.

  Padding puts the first window before `A`'s origin, where the offsets address
  memory that is not `A`'s -- use `im2col_padded` to read a fill value there, or
  `im2col_coord` to predicate.

  Post-conditions:
    rank(result) == 2  and  shape(result) == ((N,(Z,P,Q)), ((T,R,S),C))
    size[1](result) == C * size(stencil)                    the GEMM's K
    result.accessor.base is A.accessor.base                 no data is copied

  Examples:
    A = make_tensor(Layout((1, 4, 4, 1), (16, 4, 1, 1)))    # (N,H,W,C) row-major
    shape(im2col(A, (2, 2)))                       == ((1, (3, 3)), ((2, 2), 1))
    stride(im2col(A, (2, 2)).layout)               == ((16, (4, 1)), ((4, 1), 1))
    stride(im2col(A, (2, 2), stride_dhw=(2, 2)).layout) == ((16, (8, 2)), ((4, 1), 1))
    stride(im2col(A, (2, 2), stride_trs=(2, 2)).layout) == ((16, (4, 1)), ((8, 2), 1))
    shape(im2col(A, (3, 3), lower_dhw=(1, 1)))          == ((1, (4, 4)), ((3, 3), 1))
    im2col(A, (2, 2), stride_dhw=2)                     -> ValueError

    # A blocked width -- W = 8 in two blocks of four -- does not coalesce, so
    # the window origins come back blocked too. The dilation is what divides it.
    H = make_tensor(Layout((1, (2, 4), 1), (32, (1, 8), 1)))
    shape(im2col(H, (2,), stride_trs=(2,)))             == ((1, ((2, 3),)), ((2,), 1))
    stride(im2col(H, (2,), stride_trs=(2,)).layout)     == ((32, ((1, 8),)), ((8,), 1))
  """
  layout, origin = _im2col_layout(
    A.layout, stencil, lower_dhw=lower_dhw, upper_dhw=upper_dhw,
    stride_dhw=stride_dhw, lower_trs=lower_trs, stride_trs=stride_trs,
    shape_zpq=shape_zpq)
  return Tensor(A.accessor + origin, layout)


def im2col_coord(A : Tensor,          # (N, [D,] [H,] [W,] C)
                 stencil,             # ([T,] [R,] [S])
                 *,
                 lower_dhw = None,    # padding before each spatial mode; None means 0
                 upper_dhw = None,    # padding after;  None means symmetric
                 stride_dhw = None,   # traversal stride;  None means 1
                 lower_trs = None,    # stencil origin offset;  None means 0
                 stride_trs = None,   # dilation;  None means 1
                 shape_zpq = None,    # output extents, when not the fprop formula
                 ) -> Tensor:         # ((N,(Z,P,Q)), ((T,R,S),C)) -- coordinates of `A`
  """
  `im2col`'s layout over `A`'s *coordinates* rather than its memory: reading
  position `(m,k)` returns the `(n, [d,] [h,] [w,] c)` coordinate of `A` that
  `im2col(A, ...)[m,k]` would load.

  This is `im2col`'s construction with `E(i)`, the unit basis of mode `i`, in
  place of `stride(A)`, so a spatial coordinate accumulates from both halves of
  the layout. One basis per top-level mode. It is the tensor a TMA descriptor
  is built from, and it is how padding is detected: a coordinate outside `A`
  is one the activation does not have, which `in_bounds` reports.

  Examples:
    A = make_tensor(Layout((1, 4, 4, 1), (16, 4, 1, 1)))
    B = im2col_coord(A, (2, 2))
    idx2crd(B[(0, (0, 0)), ((0, 0), 0)], shape(A))  == (0, 0, 0, 0)
    idx2crd(B[(0, (1, 2)), ((1, 1), 0)], shape(A))  == (0, 2, 3, 0)
    P = im2col_coord(A, (3, 3), lower_dhw=(1, 1))
    idx2crd(P[(0, (0, 0)), ((0, 0), 0)], shape(A))  == (0, -1, -1, 0)
    in_bounds(P[(0, (0, 0)), ((0, 0), 0)], shape(A))   == False
    in_bounds(P[(0, (1, 1)), ((1, 1), 0)], shape(A))   == True

    # A blocked width both views admit, and where only the memory view can keep
    # the blocking -- same size, grouped differently.
    H = make_tensor(Layout((1, (2, 4), 1), (32, (1, 8), 1)))
    shape(im2col(H, (2,), stride_trs=(2,)))            == ((1, ((2, 3),)), ((2,), 1))
    shape(im2col_coord(H, (2,), stride_trs=(2,)))      == ((1, (6,)), ((2,), 1))
  """
  layout, origin = _im2col_layout(
    Layout(shape(A.layout), tuple(E(i) for i in range(rank(A)))), stencil,
    lower_dhw=lower_dhw, upper_dhw=upper_dhw, stride_dhw=stride_dhw,
    lower_trs=lower_trs, stride_trs=stride_trs, shape_zpq=shape_zpq)
  return Tensor(ImplicitAccessor(0) + origin, layout)


def im2col_padded(A : Tensor,          # (N, [D,] [H,] [W,] C)
                  stencil,             # ([T,] [R,] [S])
                  *,
                  fill = 0,            # value read outside the activation
                  lower_dhw = None,    # padding before each spatial mode; None means 0
                  upper_dhw = None,    # padding after;  None means symmetric
                  stride_dhw = None,   # traversal stride;  None means 1
                  lower_trs = None,    # stencil origin offset;  None means 0
                  stride_trs = None,   # dilation;  None means 1
                  shape_zpq = None,    # output extents, when not the fprop formula
                  ) -> Tensor:         # ((N,(Z,P,Q)), ((T,R,S),C)) -- bounds-checked reads
  """
  `im2col`, read-only, with out-of-bounds positions reading `fill` instead of
  whatever memory the offset would have landed on.

  The layout is `im2col_coord`'s, so every read arrives at the accessor as a
  coordinate of `A`; `OutOfBoundsAccessor` bounds-checks it and then defers to
  `A`. Padding therefore costs a comparison per element and no memory at all.

  Examples:
    A = make_tensor(Layout((1, 3, 3, 1), (9, 3, 1, 1)))
    A[0, 1, 1, 0] = 7.0
    B = im2col_padded(A, (3, 3), lower_dhw=(1, 1))
    B[(0, (1, 1)), ((1, 1), 0)] == 7.0        # centre tap of the centre window
    B[(0, (0, 0)), ((0, 0), 0)] == 0.0        # a padded corner
  """
  A_crd = im2col_coord(A, stencil, lower_dhw=lower_dhw, upper_dhw=upper_dhw,
                       stride_dhw=stride_dhw, lower_trs=lower_trs, stride_trs=stride_trs,
                       shape_zpq=shape_zpq)
  return Tensor(OutOfBoundsAccessor(A, fill) + A_crd.accessor.origin, A_crd.layout)


# ------------------------------------------------------- out-of-bounds reads

class OutOfBoundsAccessor(Accessor):
  """
  Reads a Tensor by coordinate, returning `fill` for coordinates it does not
  have.

  The offsets an `Accessor` normally receives are replaced by `ArithTuple`
  coordinates, which is what a layout with basis strides produces. Offsetting
  accumulates into that coordinate rather than into an address, so slicing an
  `im2col_padded` view keeps working.

  Read-only: a write to a padded position has nowhere to go.

  Examples:
    A = make_tensor(Layout((2, 2)))
    A[1, 1] = 5.0
    OutOfBoundsAccessor(A)[(1, 1)]      == 5.0
    OutOfBoundsAccessor(A)[(2, 0)]      == 0.0
    OutOfBoundsAccessor(A, fill=-1)[(2, 0)] == -1
  """
  def __init__(self, tensor : Tensor, fill = 0, origin = 0):
    self.tensor = tensor
    self.fill = fill
    self.origin = ArithTuple(origin)

  def __add__(self, offset):
    return OutOfBoundsAccessor(self.tensor, self.fill, self.origin + offset)

  def __getitem__(self, coord):
    crd = self.origin + ArithTuple(coord)
    return self.tensor[crd] if in_bounds(crd, shape(self.tensor)) else self.fill

  def __eq__(self, other):
    if not isinstance(other, OutOfBoundsAccessor):
      return NotImplemented
    return (self.tensor == other.tensor and self.fill == other.fill and self.origin == other.origin)

  def __repr__(self):
    return f"OutOfBoundsAccessor({self.tensor}, fill={self.fill}, origin={self.origin})"
