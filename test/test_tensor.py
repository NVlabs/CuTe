# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for pycute.tensor and pycute.accessor

These tests are also worked examples for docs/05_tensor.md.
"""

import array
import ctypes
import logging
import pytest

from pycute import *

logger = logging.getLogger()


class TestTensor:

  def test_make_tensor(self):
    """`make_tensor` allocates an `Array` of size `coshape(layout)`."""
    T = make_tensor(Layout((4, 4), (4, 1)))
    assert is_tensor(T)
    assert T.layout == Layout((4, 4), (4, 1))

    # Also accepts a tiler (int or tuple)
    T2 = make_tensor((3, 5))
    assert T2.layout == Layout((3, 5), (1, 3))

  def test_tensor_evaluation_three_coords(self):
    """A `Tensor` accepts the same coordinate forms as its `Layout`.
    Note: the subscript form `T[i]` is preferred for 1-D ints; the call
    form `T(*args)` requires the rank-correct flat or natural coord."""
    T = make_tensor(Layout((4, 4), (4, 1)), dtype=ctypes.c_int)
    for i in range(size(T)):
      T[i] = i

    # Subscript form: integral / flat / natural coords all map to the same value
    assert T[6] == (T[T.layout(6)] if False else T.accessor[T.layout(6)])
    # Flat coordinate via __getitem__
    assert T[1, 2] == T.accessor[T.layout(1, 2)]

  def test_tensor_setitem_and_getitem(self):
    """Read/write via `__setitem__`/`__getitem__`."""
    T = make_tensor(Layout((4, 4), (4, 1)), dtype=ctypes.c_int)
    T[1, 2] = 42
    T[3, 0] = 99
    assert T[1, 2] == 42
    assert T[3, 0] == 99
    assert T[0, 0] == 0

    # Setting via 1-D index also works
    T[0] = 7   # offset L(0) = 0
    assert T[0, 0] == 7

  def test_tensor_slicing_returns_subtensor(self):
    """Partial coordinates return a sub-tensor sharing the same backing array."""
    # Iterating in 1-D colex order through Layout((4, 4), (4, 1)) writes:
    #   accessor[L(i)] = i for i in 0..15
    # which, viewed as the 4x4 row-major matrix, is the table
    #   row m: [m, 4+m, 8+m, 12+m]   for m = 0..3
    T = make_tensor(Layout((4, 4), (4, 1)), dtype=ctypes.c_int)
    for i in range(size(T)):
      T[i] = i

    # Row 1 of T: T[1, j] for j = 0..3
    row = T[1, None]
    assert is_tensor(row)
    assert row.layout == Layout((4,), (1,))
    assert [row[j] for j in range(4)] == [T[1, j] for j in range(4)]
    assert [row[j] for j in range(4)] == [1, 5, 9, 13]

    # Column 2 of T: T[i, 2] for i = 0..3
    col = T[None, 2]
    assert is_tensor(col)
    assert col.layout == Layout((4,), (4,))
    assert [col[i] for i in range(4)] == [T[i, 2] for i in range(4)]
    assert [col[i] for i in range(4)] == [8, 9, 10, 11]

  def test_tensor_setitem_requires_full_coord(self):
    """Setting a value requires a complete coordinate."""
    T = make_tensor(Layout((4, 4), (4, 1)))
    with pytest.raises(ValueError):
      T[1, None] = 0    # incomplete coordinate

  def test_tensor_slicing_provides_view_semantics(self):
    """Repeated slices of the same coordinate compare equal (same memory and
    same layout) and have a stable printed accessor address."""
    A = make_tensor(Layout((4, 5, 6)), dtype=ctypes.c_double)
    A[1, 2, 3] = 42.0

    v1 = A[:, 2, :]
    v2 = A[:, 2, :]

    # Stable printed identity (the underlying C pointer, not Python's id())
    assert repr(v1.accessor) == repr(v2.accessor)

    # Value equality on the accessors and on the tensors themselves
    assert v1.accessor == v2.accessor
    assert v1 == v2

    # `None`-style slicing is equivalent to `:`-style slicing
    assert A[None, 2, None] == A[:, 2, :]

    # Mutations through one view propagate to the base and to the other view
    v1[1, 3] = 99.0
    assert A[1, 2, 3] == 99.0
    assert v2[1, 3] == 99.0

  def test_tensor_algebra_passes_through_to_layout(self):
    """Algebraic operations on a tensor act on its layout, sharing the accessor."""
    # Use a column-major layout that does coalesce all the way to 16:1
    T = make_tensor(Layout((4, 4), (1, 4)), dtype=ctypes.c_int)
    T[1, 2] = 42       # accessor[ L(1, 2) = 9 ] = 42

    Tc = coalesce(T)
    assert is_tensor(Tc)
    assert Tc.layout == Layout(16, 1)
    assert Tc.accessor is T.accessor
    # The underlying data is unchanged: Tc[9] = accessor[9] = 42
    assert Tc[9] == 42
    assert T[1, 2] == 42

    Td = logical_divide(T, Layout(2, 1))
    assert rank(Td) == 2
    assert Td.accessor is T.accessor


class TestAccessor:

  def test_array_round_trip(self):
    """`Array` provides random-access read/write on a `ctypes` buffer."""
    a = Array(8, dtype=ctypes.c_int)
    for i in range(8):
      a[i] = i * i
    assert [a[i] for i in range(8)] == [0, 1, 4, 9, 16, 25, 36, 49]

  def test_array_offset_writes_through(self):
    """Adding to an `Array` produces a `Ptr` into the same storage."""
    a = Array(8, dtype=ctypes.c_int)
    for i in range(8):
      a[i] = i

    v = a + 3
    assert v[0] == 3
    assert v[2] == 5
    v[1] = 99
    assert a[4] == 99        # write through to the original array

  def test_offset_repr_shows_underlying_address(self):
    """`repr` is stable across re-slicing because it shows the C pointer,
    not the Python `id()` of the wrapper."""
    a = Array(8, dtype=ctypes.c_int)
    assert repr(a + 3) == repr(a + 3)
    assert repr(a + 3) != repr(a + 4)

    expected_addr = ctypes.addressof(a.ptr.contents) + 3 * ctypes.sizeof(ctypes.c_int)
    assert f"{expected_addr:#018x}" in repr(a + 3)

  def test_offset_value_equality(self):
    """Two pointers into the same memory with the same dtype compare equal"""
    a = Array(8, dtype=ctypes.c_int)

    assert a + 3 == a + 3
    assert a + 3 != a + 4

    # An Array and a Ptr at the same address with the same dtype are equal
    assert a == a + 0

    # Different dtypes at the same address are not equal
    b = Array(8, dtype=ctypes.c_double)
    assert a != b

    # Equality with non-accessor objects is well-defined (False, not raise)
    assert a != 0
    assert a + 1 != "anything"

  def test_offsets_canonicalize_to_owning_array(self):
    """Chained offsets flatten: `.base` is always the owning `Array`."""
    a = Array(8, dtype=ctypes.c_int)
    v1 = a + 3
    v2 = v1 + 2
    assert v1.base is a
    assert v2.base is a

    # ...and the absolute pointer is the sum of offsets, not chained casting.
    expected_addr = ctypes.addressof(a.ptr.contents) + 5 * ctypes.sizeof(ctypes.c_int)
    assert ctypes.addressof(v2.ptr.contents) == expected_addr

  def test_pointer_wraps_foreign_storage(self):
    """`Ptr` reads and writes storage owned by someone else."""
    data = array.array('u', 'abcd')
    T = Tensor(Ptr(data.buffer_info()[0], ctypes.c_wchar), Layout(4))
    assert [T[i] for i in range(4)] == list('abcd')

    T[2] = 'Z'
    assert data.tounicode() == 'abZd'   # the write landed in the array.array

  def test_pointer_infers_dtype_and_owner(self):
    """Given the object rather than an address, dtype and owner are inferred."""
    data = array.array('d', [1.0, 2.0, 3.0, 4.0])
    p = Ptr(data)
    assert p.dtype is ctypes.c_double
    assert p.owner is data              # keeps the storage alive
    assert p.base is data

    T = Tensor(p, Layout((2, 2), (2, 1)))
    T[1, 0] = 9.0
    assert data[2] == 9.0

    # A bare address carries no type information, so dtype is required
    with pytest.raises(ValueError):
      Ptr(data.buffer_info()[0])

    # ...and an object with no addressable storage is rejected outright
    with pytest.raises(TypeError):
      Ptr('abcd')

  def test_pointer_offset_and_equality(self):
    """Offsetting a `Ptr` steps by `sizeof(dtype)` and preserves the owner."""
    data = array.array('i', [0, 1, 2, 3, 4, 5])
    p = Ptr(data)

    q = p + 3
    assert q[0] == 3
    assert q.address == p.address + 3 * ctypes.sizeof(ctypes.c_int)
    assert q.owner is data
    assert (q + 1)[0] == 4

    assert p + 3 == q                   # same dtype at the same address
    assert p + 2 != q
    assert p != 0                       # comparison with a non-accessor is False
    assert f"{q.address:#018x}" in repr(q)

  def test_array_is_a_pointer_to_its_own_storage(self):
    """`Array` owns its allocation; offsetting it yields a plain `Ptr` into it."""
    a = Array(8, dtype=ctypes.c_int)
    assert isinstance(a, Ptr)
    assert isinstance(a + 1, Ptr)
    assert a.base is a
    assert (a + 1).base is a

  def test_pointer_over_numpy_array(self):
    """A `Ptr` can view NumPy storage; the layout supplies the strides."""
    np = pytest.importorskip("numpy")
    a = np.arange(12, dtype=np.int32).reshape(3, 4)

    T = Tensor(Ptr(a), Layout((3, 4), (4, 1)))
    assert T.accessor.dtype is ctypes.c_int32
    assert [T[i, j] for i in range(3) for j in range(4)] == list(range(12))

    T[2, 3] = 99
    assert a[2, 3] == 99

    # Every-other-column view: same base address, striding lives in the layout
    V = Tensor(Ptr(a[:, ::2]), Layout((3, 2), (4, 2)))
    assert [V[i, j] for i in range(3) for j in range(2)] == [0, 2, 4, 6, 8, 10]

  def test_implicit_accessor_passes_offset_through(self):
    """`ImplicitAccessor` returns its offset rather than dereferencing."""
    ia = ImplicitAccessor(0)
    assert ia[5] == 5
    assert (ia + 3)[2] == 5

    # With an ArithTuple coordinate base, addition is elementwise
    ia = ImplicitAccessor(ArithTuple(10, 20))
    assert ia[ArithTuple(1, 2)] == (11, 22)


class TestIdentityTensor:

  def test_identity_tensor_returns_coordinate(self):
    """`identity_tensor(shape)[c] == c` for every in-bounds coord `c`."""
    I = identity_tensor((3, 4))
    for i in range(3):
      for j in range(4):
        assert I[i, j] == (i, j)

    # Hierarchical shape
    I = identity_tensor((2, (3, 4)))
    for i in range(2):
      for j in range(3):
        for k in range(4):
          assert I[i, (j, k)] == (i, (j, k))

  def test_identity_tensor_supports_slicing(self):
    """Slicing an identity tensor preserves coordinate semantics on the
    surviving modes (used for predication)."""
    I = identity_tensor((3, 4))
    row = I[1, None]
    for j in range(4):
      assert row[j] == (1, j)
