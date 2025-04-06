__all__ = ["lagrange"]

import numpy as np


def lagrange(x_arr, y_arr, x):
    r"""Lagrange interpolation polynomial value at `x`.

    .. math::

        \begin{gather}
            P_n(x) = L_n(x) = \sum_{i=0}^{n} l_i(x) f(x_i) \\
            l_i(x) = \prod_{j=0, j \neq i}^{n} \frac{x - x_j}{x_i - x_j}
        \end{gather}

    where :math:`l_i(x)` is the Lagrange basis polynomial for the i-th data point.

    Parameters
    ----------
    x_arr : numpy.ndarray
        The x-coordinates of the data points.
    y_arr : numpy.ndarray
        The y-coordinates of the data points.
    x : numpy.float32
        The x-coordinate at which to evaluate the polynomial.

    Returns
    -------
    lagrange: numpy.float32
        The value of the polynomial at `x`.

    """
    if not isinstance(x_arr, np.ndarray):
        raise TypeError("x must be a numpy.ndarray")

    if not isinstance(y_arr, np.ndarray):
        raise TypeError("y must be a numpy.ndarray or callable")

    if not isinstance(x, np.float32):
        raise TypeError("x0 must be a numpy.float32")

    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")

    n = len(x)
    p = 0.0

    for i in range(n):
        l_i = 1.0
        for j in range(n):
            if i != j:
                l_i *= (x - x_arr[j]) / (x_arr[i] - x[j])
        p += y_arr[i] * l_i

    return p
