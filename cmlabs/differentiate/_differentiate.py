__all__ = ["lagrange_derivative"]

import numpy as np
import itertools


def lagrange_derivative(xvals, yvals, x, k):
    r"""Return the k-th derivative of the Lagrange polynomial.

    .. math::

        \begin{gather}
            L_n^{(k)}(x) = \sum_{i=0}^{n} f(x_i) l_i^{(k)}(x) \\
            l_i^{(k)}(x) = \left(\prod_{j=0, j \neq i}^{n}
            \frac{x - x_j}{x_i - x_j}\right)^{(k)}
        \end{gather}

    Parameters
    ----------
    xvals : array_like, 1-D
        The sorted x-coordinates of the data points.
    yvals : array_like, 1-D
        The y-coordinates of the data points, i.e., f(:math:`x`).
    x : float
        The x-coordinate at which to evaluate the polynomial.
    k : int
        The order of the derivative.

    Returns
    -------
    res: float
        The value of the k-th derivative of the polynomial at :math:`x`.

    See Also
    --------
    lagrange

    Notes
    -----
    This function might be slow but it is useful for testing purposes.
    """

    def lagrange_basis_kth_derivative_numerator(indexes, kth):
        if kth == 0:
            return np.prod([(x - xvals[i]) for i in indexes])
        numerator = 0.0
        for indexes_ in itertools.combinations(indexes, len(indexes) - 1):
            numerator += lagrange_basis_kth_derivative_numerator(indexes_, kth - 1)
        return numerator

    n = len(xvals) - 1
    res = 0.0
    for i in range(n + 1):
        indexes = [j for j in range(n + 1) if j != i]
        li_derivative = lagrange_basis_kth_derivative_numerator(indexes, k)
        denumerator = np.prod([(xvals[i] - xvals[j]) for j in indexes])
        res += (yvals[i] * li_derivative) / denumerator
    return res
