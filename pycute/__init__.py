# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .typedefs import *
from .htuple import *
from .atuple import *
from .shape import *
from .layout import *
from .algebra import *
from .swizzle import *

from .accessor import *
from .tensor import *

# Public API: every name re-exported above, minus submodules and the incidental
# stdlib names pulled in transitively by the `import *` chains (`reduce`,
# `zip_longest`, `ABC`, etc.). New PyCuTe definitions are picked up automatically.
import types as _types
_INCIDENTAL = {"annotations", "ABC", "abstractmethod", "Iterable", "Iterator",
               "Union", "Any", "TypeAlias", "reduce", "update_wrapper",
               "zip_longest"}
__all__ = sorted(
    _n for _n, _v in globals().items()
    if not _n.startswith("_")
    and _n not in _INCIDENTAL
    and not isinstance(_v, _types.ModuleType)
)
del _types, _INCIDENTAL
