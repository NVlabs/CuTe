# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod


class Accessor(ABC):
  @abstractmethod
  def __add__(self, idx):
    pass

  @abstractmethod
  def __getitem__(self, idx):
    pass

class MutableAccessor(ABC):
  @abstractmethod
  def __add__(self, idx):
    pass

  @abstractmethod
  def __getitem__(self, idx):
    pass

  @abstractmethod
  def __setitem__(self, idx, value):
    pass


import ctypes


class ArrayView(MutableAccessor):
  def __init__(self, base, offset=0):
    # Always anchor to the owning Array so a chain of slices never accumulates
    self.base = base.base if isinstance(base, ArrayView) else base
    self.dtype = base.dtype
    byte_offset = offset * ctypes.sizeof(self.dtype)
    self.ptr = ctypes.cast(ctypes.addressof(base.ptr.contents) + byte_offset, ctypes.POINTER(self.dtype))

  def __getitem__(self, idx):
    return self.ptr[int(idx)]

  def __setitem__(self, idx, value):
    self.ptr[int(idx)] = self.dtype(value)

  def __add__(self, idx):
    return ArrayView(self, idx)

  def __eq__(self, other):
    if not isinstance(other, (Array, ArrayView)):
      return NotImplemented
    return (self.dtype is other.dtype and
            ctypes.addressof(self.ptr.contents) == ctypes.addressof(other.ptr.contents) and
            self.base == other.base)

  def __repr__(self):
    return f"ArrayView({ctypes.addressof(self.ptr.contents):#018x}, {self.dtype.__name__})"


class Array(MutableAccessor):
  def __init__(self, size, dtype=ctypes.c_double):
    self.base = self
    self.dtype = dtype
    self._raw_storage = (dtype * size)()  # Allocate and keep a reference so it doesn't get GC'd
    self.ptr = ctypes.cast(self._raw_storage, ctypes.POINTER(self.dtype))

  def __getitem__(self, idx):
    return self.ptr[int(idx)]

  def __setitem__(self, idx, val):
    self.ptr[int(idx)] = self.dtype(val)

  def __add__(self, idx):
    return ArrayView(self, idx)

  def __eq__(self, other):
    if not isinstance(other, (Array, ArrayView)):
      return NotImplemented
    return (self.dtype is other.dtype and
            ctypes.addressof(self.ptr.contents) == ctypes.addressof(other.ptr.contents))

  def __repr__(self):
    return f"Array({ctypes.addressof(self.ptr.contents):#018x}, {self.dtype.__name__})"


class ImplicitAccessor(Accessor):
  def __init__(self, base):
    self.base = base

  def __getitem__(self, offset):
    return self.base + offset

  def __add__(self, offset):
    return ImplicitAccessor(self.base + offset)

  def __eq__(self, other):
    if not isinstance(other, ImplicitAccessor):
      return NotImplemented
    return self.base == other.base

  def __repr__(self):
    return f"{{{self.base}}}"
