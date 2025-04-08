"""
=========================================
Interpolation (:mod:`cmlabs.interpolate`)
=========================================

.. currentmodule:: cmlabs.interpolate

The `cmlabs.interpolate` module provides a collection of interpolation methods and
related utilities.

Lagrange interpolation
======================

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   lagrange
   lagrange_remainder


Newton interpolation
====================

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   divided_differences
   newton


Finite differences interpolation
================================

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   forward_differences
   backward_differences
   newtonfd


Tests
=====

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   test_lagrange_degree
   test_lagrange_remainder_1
   test_lagrange_degree_2
   test_lagrange_remainder_2
   test_lagrange_compare_with_newton
   test_lagrange_from_docs_example
   test_lagrange_remainder_from_docs_example
   test_newton_from_docs_example
"""

from ._interpolate import *

from .tests.test_interpolate import *
