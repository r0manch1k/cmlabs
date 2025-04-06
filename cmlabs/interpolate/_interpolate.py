__all__ = ["lagrange", "lagrange_remainder", "divided_differences", "newton"]

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
    >>> xvals = np.array([0, 2, 3, 5])
    >>> yvals = np.array([1, 3, 2, 5])
    >>> x = np.float32(1.5)
    >>> lagrange(xvals, yvals, x)
    3.3375000000000004

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


def divided_differences(xvals, yvals):
    r"""Compute the ordered divided differences list.

    Parameters
    ----------
    xvals : array_like, 1-D
        The x-coordinates of the data points.
    yvals : array_like, 1-D
        The y-coordinates of the data points.

    Returns
    -------
    res: array_like, 1-D
        The divided differences list.

    Notes
    -----
    The divided differences list has the following structure:

    .. math::

        [f(x_0), f(x_0, x_1), f(x_0, x_1, x_2), \ldots, f(x_0, x_1, \ldots, x_n)]

    Algorithm is based on the following formula:

    .. math::

        f(x_0, x_1, \ldots, x_n) = \frac{f(x_1, \ldots, x_n) -
        f(x_0, \ldots, x_{n-1})}{x_n - x_0}

    where :math:`f[x_0, x_1, \ldots, x_n]` is the divided difference of the function
    at the points :math:`x_0, x_1, \ldots, x_n`.

    It starts moving from the right side of the array, so the iterations look like this:

    .. math::

        \begin{aligned}
            \\
            [f(x_0), f(x_1), \ldots, f(x_n)] &\to
            [f(x_0), f(x_0, x_1), \ldots, f(x_{n-1}, x_n)] \\
            [f(x_0), f(x_0, x_1), \ldots, f(x_{n-1}, x_n)] &\to
            [f(x_0), f(x_0, x_1), \ldots, f(x_{n-2}, x_{n-1}, x_n)] \\
            &\vdots \\
            [f(x_0), f(x_0, x_1), \ldots, f(x_1, x_2, \ldots, x_n)] &\to
            [f(x_0), f(x_0, x_1), \ldots, f(x_0, x_1, \ldots, x_n)]
        \end{aligned}

    The divided differences list is used to compute the coefficients of the Newton
    interpolation polynomial.

    """
    n = len(xvals)

    if n != len(yvals):
        raise ValueError("xvals and yvals must have the same length")

    coef = [y for y in yvals]

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (xvals[i] - xvals[i - j])

    return coef


def newton(xvals, x, yvals=None, coef=None):
    r"""Newton interpolation polynomial value at `x`.

    .. math::

        \begin{gather}
        L_n(x) = \sum_{i=0}^{n} f(x_0, x_1, \ldots, x_i) \omega_i(x), \\
        \omega_0(x) = 1, \quad \omega_1(x) = x - x_0, \quad
        \omega_i(x) = \prod_{j=0, j \neq i}^{n} (x - x_j), i \geq 2
        \end{gather}

    where :math:`f(x_0, x_1, \ldots, x_i)` is the divided difference of the function
    at the points :math:`x_0, x_1, \ldots, x_i`.

    Parameters
    ----------
    xvals : array_like, 1-D
        The sorted x-coordinates of the data points.
    x : float
        The x-coordinate at which to evaluate the polynomial.
    yvals : array_like, 1-D, optional
        The y-coordinates of the data points, i.e., f(`x`).
        Only used if `coef` is not provided.
    coef : array_like, 1-D, optional
        The coefficients of the Newton polynomial.
        If not provided, the function will compute the divided differences list
        using the `divided_differences` function.

    Returns
    -------
    res: float
        The value of the polynomial at `x`.

    Notes
    -----
    The `coef` parameter is optional and can be used to provide the coefficients
    of the Newton polynomial. If not provided, the function will compute the
    divided differences list using the `divided_differences` function.

    The divided differences list has the following structure:

    .. math::

        [f(x_0), f(x_0, x_1), f(x_0, x_1, x_2), \ldots, f(x_0, x_1, \ldots, x_n)]

    Use Horner's method to compute the value of the polynomial at `x`.

    .. math::

        \begin{gather}
            P(x) = a_0 + (x - x_0)(a_1 + (x - x_1)(a_2 + \ldots)) \\
            a_0 = f(x_0), \quad a_1 = f(x_0, x_1), \quad
            a_2 = f(x_0, x_1, x_2), \quad \ldots, \quad
            a_n = f(x_0, x_1, \ldots, x_n)
        \end{gather}


    Examples
    --------
    >>> import numpy as np
    >>> from cmlabs.interpolate import newton
    >>> xvals = np.array([0, 2, 3, 5])
    >>> yvals = np.array([1, 3, 2, 5])
    >>> x = np.float32(1.5)
    >>> newton(xvals, x, yvals=yvals)
    3.3375

    """
    n = len(xvals)

    if coef is not None and n != len(coef):
        raise ValueError("coef must have the same length as xvals and yvals")

    if yvals is not None:
        coef = divided_differences(xvals, yvals)
    elif coef is None:
        raise ValueError("Either yvals or coef must be provided")

    res = coef[-1]

    for i in range(n - 2, -1, -1):
        res = res * (x - xvals[i]) + coef[i]

    return res
