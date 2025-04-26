"""
=====================================
Linear Algebra (:mod:`cmlabs.linalg`)
=====================================

.. currentmodule:: cmlabs.linalg

The `cmlabs.linalg` module provides a collection of functions for
linear algebra algorithms.

Linear Algebra
==============
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   thomas


Tests
=====
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   test_thomas
   test_thomas_from_docs_example

"""

from ._linalg import *

from .tests.test_linalg import *
