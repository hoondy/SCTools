"""Public package interface for SCTools."""

import pge as _core

from . import io, pl, pp, tl
from ._version import __version__

_top_level_exports = [
    name
    for name, obj in vars(_core).items()
    if not name.startswith("_") and getattr(obj, "__module__", None) == _core.__name__
]

__all__ = ["__version__", "io", "pl", "pp", "tl", *_top_level_exports]

globals().update({name: getattr(_core, name) for name in _top_level_exports})
