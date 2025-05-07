"""
=============================================
Differentiation (:mod:`cmlabs.differentiate`)
=============================================

.. currentmodule:: cmlabs.differentiate

The `cmlabs.differentiate` module provides a collection of differentiation methods and
related utilities.

Lagrange differentiation
========================
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   lagrange_derivative

Tests
=====
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   test_lagrange_derivative
   test_lagrange_derivative_from_docs_example
"""

from ._differentiate import *

from .tests.test_differentiate import *
