"""
=====================================
Optimization (:mod:`cmlabs.optimize`)
=====================================

.. currentmodule:: cmlabs.optimize

The `cmlabs.optimize` module provides a collection of optimization algorithms.

Optimization
============
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   find_root_brackets
   bisect
   newton
   secant


Tests
=====
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   test_root
   test_find_root_brackets_from_docs_example
   test_bisect_from_docs_example
   test_newton_from_docs_example
   test_secant_from_docs_example

"""

from ._optimize import *

from .tests.test_optimize import *
