"""
Cmlabs - A computational mathematics library for Python
=======================================================

This library provides a collection of functions and classes for
computational mathematics, including interpolation, numerical integration,
differential equations, and optimization.

Subpackages
-----------
::

 cmlabs.interpolate
 cmlabs.differentiate
 cmlabs.integrate
 cmlabs.linalg
"""

import importlib as _importlib


submodules = [
    "interpolate",
    "differentiate",
    "integrate",
    "linalg",
]

__all__ = submodules + [
    "__version__",
]


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f"cmlabs.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'cmlabs' has no attribute '{name}'")
