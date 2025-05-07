import numpy as np
import itertools

__all__ = ["lagrange_derivative"]


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

    Complexity analysis.
    The function lagrange_derivative recursively computes the k-th derivative of
    the Lagrange interpolating polynomial. The recursive step involves iterating
    over all subsets of given indices, resulting in a combinatorial growth of the
    number of recursive calls. Therefore, the overall time complexity is approximately
    :math:`O(n^k)`, where :math:`n` is the number of interpolation points and :math:`k`
    is the derivative order. Since the recursion is not based on simple division into
    smaller subproblems, the Master Theorem does not apply. If :math:`k` is fixed and
    small, the algorithm runs in polynomial time with respect to :math:`n`; otherwise,
    for large :math:`k`, the complexity becomes exponential.

    Examples
    --------
    >>> from cmlabs.differentiate import lagrange_derivative
    >>> xvals = np.array([0, 1, 2, 3, 4])
    >>> yvals = np.array([0, 1, 4, 9, 16])
    >>> x = 2.5
    >>> k = 2
    >>> lagrange_derivative(xvals, yvals, x, k)
    6.0
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
