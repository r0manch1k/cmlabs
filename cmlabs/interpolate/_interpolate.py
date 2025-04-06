__all__ = ["lagrange", "lagrange_remainder"]

import numpy as np
import math


def lagrange(xvals, yvals, x):
    r"""Lagrange interpolation polynomial value at `x`.

    .. math::

        \begin{align*}
        P_n(x) &= L_n(x) = \sum_{i=0}^{n} l_i(x) f(x_i) \\
        l_i(x) &= \prod_{j=0, j \neq i}^{n} \frac{x - x_j}{x_i - x_j}
        \end{align*}

    where :math:`l_i(x)` is the Lagrange basis polynomial for the i-th data point.

    Return the value of the Lagrange polynomial that has degree `n = len(xvals)-1`

    Parameters
    ----------
    xvals : array_like, 1-D
        The sorted x-coordinates of the data points.
    yvals : array_like, 1-D
        The y-coordinates of the data points, i.e., f(`x`).
    x : float
        The x-coordinate at which to evaluate the polynomial.

    Returns
    -------
    res: float
        The value of the polynomial at `x`.

    Examples
    --------
    >>> import numpy as np
    >>> from cmlabs.interpolate import lagrange
    >>> xvals = np.array([0, 1, 2])
    >>> yvals = np.array([1, 2, 3])
    >>> x = np.float32(1.5)
    >>> lagrange(xvals, yvals, x)
    2.5

    """
    n = len(xvals) - 1

    if n != len(yvals) - 1:
        raise ValueError("xvals and yvals must have the same length")

    L_n = 0.0

    for i in range(n + 1):
        l_i = 1.0
        for j in range(n + 1):
            if i != j:
                l_i *= (x - xvals[j]) / (xvals[i] - xvals[j])
        L_n += yvals[i] * l_i

    return L_n


def lagrange_remainder(xvals, M, x=None, method="auto"):
    r"""Remainder in Lagrange interpolation formula.

    Common notation for the Lagrange interpolation polynomial is:

    .. math::

        \begin{aligned}
        R_n(x) &= f(x) - L_n(x)
                = \frac{f^{(n+1)}(\xi)}{(n+1)!} \prod_{i=0}^{n} (x - x_i), \quad
                \xi \in (a, b) \\
        \end{aligned}

    where :math:`L_n(x)` is the Lagrange polynomial of degree `n`and
    :math:`R_n(x)` is the remainder term.

    Parameters
    ----------
    xvals : array_like, 1-D
        The x-coordinates of the data points.
    M : float
        The bound for the (n+1)-th derivative of the function.
    x : float, optional
        The x-coordinate at which to evaluate the polynomial.
        Only used if `method` is 'exact'.
    mehtod : {'auto', 'exact', 'bound'}, optional
        Defines the method to compute the remainder.
        The following options are available (default is 'bound'):

          * 'auto' : uses 'exact' if `x` is provided, otherwise uses 'bound'.
          * 'exact' : uses the exact formula for the remainder term.
          * 'bound' : uses the bound formula for the remainder term.

    Returns
    -------
    res: float
        The value of the remainder term.

    Notes
    -----
    The `method` parameter determines how the remainder term is computed.
    If `method` is 'exact', the function will compute the exact value of the
    remainder term using the formula:

    .. math::

        |R_n(x)| \leq \frac{M_{n+1}}{(n+1)!} \prod_{i=0}^{n} |x - x_i|

    where :math:`n` is the length of the `xvals` array, :math:`M_{n+1}` is the bound for
    the (n+1)-th derivative of the function.

    If `method` is 'bound', the function will compute the bound for the remainder term
    using the formula:

    .. math::

        |f(x) - L_n(x)| \leq \frac{M_{n+1}(x_{[n]}-x_{[0]})^{n+1}}{2^{2n+1}(n+1)!}

    where :math:`x_{[i]}` is the i-th element of the sorted `xvals` array

    This formula follows from the properties of Chebyshev polynomials.

    Examples
    --------
    >>> import numpy as np
    >>> from cmlabs.interpolate import lagrange, lagrange_remainder
    >>> # f(x) = sin(x)
    >>> xvals = np.array([0, np.pi/2, np.pi/2])
    >>> yvals = np.array([0, 1/2, 1])
    >>> # M_3 = max |f'''(x)| = 1
    >>> M_3 = 1.0
    >>> x = np.p
    >>> np.sin(x) - lagrange(xvals, yvals, x)
    0.019606781186547573
    >>> # Exact remainder
    >>> lagrange_remainder(xvals, M, x)
    0.020186378047070193
    >>> # Bound remainder
    >>> lagrange_remainder(xvals, M)
    0.020186378047070193

    """
    n = len(xvals) - 1
    xvals = np.sort(xvals)

    if method == "auto":
        if x is None:
            method = "bound"
        else:
            method = "exact"

    if method == "exact":
        res = M / math.factorial(n + 1)
        for i in range(n + 1):
            res *= abs(x - xvals[i])
    elif method == "bound":
        res = (
            M
            * (xvals[-1] - xvals[0]) ** (n + 1)
            / (2 ** (2 * n + 1) * math.factorial(n + 1))
        )

    return res
