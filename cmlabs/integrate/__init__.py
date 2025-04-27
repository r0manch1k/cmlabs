"""
=====================================
Integration (:mod:`cmlabs.integrate`)
=====================================

.. currentmodule:: cmlabs.integrate

The `cmlabs.integrate` module provides a collection of integration methods and
related utilities.

Integration
===========
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   rectangle
   midpoint
   trapezoid
   simpsonq
   simpsonc
   weddles
   newton_cotes


Tests
=====
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   test_rectangle_error
   test_midpoint_error
   test_trapezoid_error
   test_simpsonq_error
   test_simpsonc_error
   test_weddles_error
   test_newton_cotes_error
   test_rectangle_from_docs_example
   test_midpoint_from_docs_example
   test_trapezoid_from_docs_example
   test_simpsonq_from_docs_example
   test_simpsonc_from_docs_example
   test_weddles_from_docs_example

"""

from ._integrate import *

from .tests.test_integrate import *
