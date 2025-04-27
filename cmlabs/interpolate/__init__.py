"""
=========================================
Interpolation (:mod:`cmlabs.interpolate`)
=========================================

.. currentmodule:: cmlabs.interpolate

The `cmlabs.interpolate` module provides a collection of interpolation methods and
related utilities.

Lagrange/Newton interpolation
=============================

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   lagrange
   remainder
   divided_differences
   newton


Finite differences interpolation
================================

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   finite_differences
   forward_differences
   backward_differences
   newtonfd
   newtonbd
   gaussfd
   gaussbd
   stirling
   bessel
   interpolate


Splines
=======
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   CubicSpline


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
   test_interpolate_remainder
   test_cubic_spline
   test_lagrange_from_docs_example
   test_remainder_from_docs_example
   test_newton_from_docs_example
   test_finite_differences_from_docs_example
   test_forward_differences_from_docs_example
   test_backward_differences_from_docs_example
   test_newtonfd_from_docs_example
   test_newtonbd_from_docs_example
   test_gaussfd_from_docs_example
   test_gaussbd_from_docs_example
   test_cubic_spline_from_docs_example

"""

from ._interpolate import *
from ._cubic import *

from .tests.test_interpolate import *
