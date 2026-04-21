"""Public package interface for SCTools."""

import pge as _core

from ._version import __version__

__all__ = [
    name
    for name, obj in vars(_core).items()
    if not name.startswith("_") and getattr(obj, "__module__", None) == _core.__name__
]

globals().update({name: getattr(_core, name) for name in __all__})
