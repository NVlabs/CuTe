# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Accessors: the memory of a CuTe Tensor

A `Layout` maps a coordinate to an offset; an accessor maps that offset to a
value. Keeping the two apart is what lets one layout address ordinary memory, a
lazily-computed function, or a transformed view without the layout algebra
knowing the difference.

Two ABCs fix the contract -- `Accessor` for read-only access and
`MutableAccessor` for read/write -- and both require `__add__`, so that an
accessor can be *offset* to yield another accessor.
"""

from abc import ABC, abstractmethod


class Accessor(ABC):
  """
  ABC for a read-only random-access handle.
  """
  @abstractmethod
  def __add__(self, idx): pass
  @abstractmethod
  def __getitem__(self, idx): pass

class MutableAccessor(ABC):
  """
  ABC for a read/write random-access handle.
  """
  @abstractmethod
  def __add__(self, idx): pass
  @abstractmethod
  def __getitem__(self, idx): pass
  @abstractmethod
  def __setitem__(self, idx, value): pass


import ctypes


# `memoryview.format` (struct) character to the equivalent ctypes element type.
# 'u' is array('u') (wchar_t); 'w' is Py_UCS4.
_FORMAT_DTYPE = {
  '?': ctypes.c_bool,     'c': ctypes.c_char,
  'b': ctypes.c_byte,     'B': ctypes.c_ubyte,
  'h': ctypes.c_short,    'H': ctypes.c_ushort,
  'i': ctypes.c_int,      'I': ctypes.c_uint,
  'l': ctypes.c_long,     'L': ctypes.c_ulong,
  'q': ctypes.c_longlong, 'Q': ctypes.c_ulonglong,
  'n': ctypes.c_ssize_t,  'N': ctypes.c_size_t,
  'f': ctypes.c_float,    'd': ctypes.c_double,
  'u': ctypes.c_wchar,    'w': ctypes.c_wchar,
  'P': ctypes.c_void_p,
}


def _buffer_address(obj) -> int:
  """
  Address of the first element of `obj`'s storage.

  Understands ctypes objects, `array.array` (`buffer_info`), the NumPy array
  interface (`__array_interface__`), and any writable buffer.
  """
  try:
    return ctypes.addressof(obj)                          # ctypes arrays/scalars
  except TypeError:
    pass
  if hasattr(obj, 'buffer_info'):                         # array.array
    return obj.buffer_info()[0]
  if hasattr(obj, '__array_interface__'):                 # numpy and friends
    return obj.__array_interface__['data'][0]
  try:
    return ctypes.addressof(ctypes.c_char.from_buffer(obj))
  except TypeError as e:
    raise TypeError(f"Ptr: cannot take the address of a {type(obj).__name__}; "
                    f"pass an integer address and a dtype instead") from e


def _buffer_dtype(obj):
  """ctypes element type of `obj`'s storage, or None when it is not known."""
  if isinstance(obj, ctypes.Array):
    return obj._type_
  try:
    return _FORMAT_DTYPE.get(memoryview(obj).format)
  except TypeError:
    return None


class Ptr(MutableAccessor):
  """
  A typed pointer into memory owned elsewhere: the general-purpose accessor.

  `source` is either an integer address -- e.g. `array.buffer_info()[0]` or
  `ndarray.__array_interface__['data'][0]` -- in which case `dtype` is
  required, or an object that can provide one (`array.array`, `numpy.ndarray`,
  a ctypes array, `bytearray`, ...), in which case `dtype` and the storage
  `owner` are inferred from it.

  Element `i` is read and written at `address + i * sizeof(dtype)`. Nothing is
  allocated, copied, or bounds-checked, so a `Ptr` is exactly as valid as the
  address it is handed. Prefer passing the object over a bare address: the
  reference kept in `owner` stops the storage being collected while in use.

  Offsetting yields another `Ptr` anchored to the same owner, so a chain of
  offsets never accumulates wrappers.

  Examples:
    data = array.array('d', [1.0, 2.0, 3.0, 4.0])
    T = Tensor(Ptr(data), Layout((2, 2), (2, 1)))         # dtype inferred
    T[1, 0] = 9.0
    data.tolist() == [1.0, 2.0, 9.0, 4.0]
    Ptr(data.buffer_info()[0], ctypes.c_double)[0] == 1.0 # a raw address
    Ptr(data.buffer_info()[0]) -> ValueError              # ... needs a dtype
  """
  def __init__(self, source, dtype=None, owner=None):
    if isinstance(source, int):
      self.address = source
      self.owner = owner
      if dtype is None:
        raise ValueError(f"Ptr({source:#x}): dtype is required for a raw address")
    else:
      self.address = _buffer_address(source)
      self.owner = source if owner is None else owner
      dtype = _buffer_dtype(source) if dtype is None else dtype
      if dtype is None:
        raise ValueError(f"Ptr: could not infer dtype of {type(source).__name__}; pass dtype explicitly")
    self.dtype = dtype
    self.ptr = ctypes.cast(self.address, ctypes.POINTER(dtype))

  @property
  def base(self):
    """The accessor or object owning the storage; `self` when it is unowned."""
    return self if self.owner is None else self.owner

  def __add__(self, idx):
    return Ptr(self.address + int(idx) * ctypes.sizeof(self.dtype), self.dtype, owner=self.base)

  def __getitem__(self, idx):
    return self.ptr[int(idx)]

  def __setitem__(self, idx, value):
    self.ptr[int(idx)] = self.dtype(value)

  def __eq__(self, other):
    if not isinstance(other, Ptr):
      return NotImplemented
    return self.address == other.address and self.dtype == other.dtype

  def __repr__(self):
    return f"Ptr({self.address:#018x}, {self.dtype.__name__})"


class Array(Ptr):
  """
  A `Ptr` that allocates and owns `size` elements of `dtype`.

  The allocation is kept alive by the `Array` itself, so unlike a bare `Ptr` it
  stays valid for as long as it is reachable.

  Examples:
    a = Array(16, dtype=ctypes.c_int)
    a[5] = 42
    a[5] == 42
  """
  def __init__(self, size, dtype=ctypes.c_double):
    self._raw_storage = (dtype * size)()  # Allocate and keep a reference so it doesn't get GC'd
    super().__init__(ctypes.addressof(self._raw_storage), dtype)

  def __repr__(self):
    return f"Array({self.address:#018x}, {self.dtype.__name__})"


class ImplicitAccessor(Accessor):
  """
  An accessor with no memory behind it: reading offset `i` returns `base + i`.

  Examples:
    ImplicitAccessor(0)[7]          == 7
    ImplicitAccessor(100)[7]        == 107
    (ImplicitAccessor(0) + 100)[7]  == 107
  """
  def __init__(self, base):
    self.base = base

  def __add__(self, offset):
    return ImplicitAccessor(self.base + offset)

  def __getitem__(self, offset):
    return self.base + offset

  def __eq__(self, other):
    if not isinstance(other, ImplicitAccessor):
      return NotImplemented
    return self.base == other.base

  def __repr__(self):
    return f"{{{self.base}}}"


class TransformAccessor(Accessor):
  """
  Wraps an accessor and applies `transform` on every read:
  `TransformAccessor(a, f)[i] == f(a[i])`.

  Offsetting propagates to the inner accessor. Read-only: there is
  no inverse with which to push a write back through `transform`.

  Examples:
    a = Array(4, dtype=ctypes.c_int)
    a[3] = 3
    TransformAccessor(a, lambda x: x * x)[3] == 9
  """
  def __init__(self, accessor, transform):
    self.accessor = accessor
    self.transform = transform

  def __add__(self, offset):
    return TransformAccessor(self.accessor + offset, self.transform)

  def __getitem__(self, offset):
    return self.transform(self.accessor[offset])

  def __eq__(self, other):
    if not isinstance(other, TransformAccessor):
      return NotImplemented
    return self.accessor == other.accessor and self.transform == other.transform

  def __repr__(self):
    return f"TransformAccessor({self.accessor}, {self.transform})"
