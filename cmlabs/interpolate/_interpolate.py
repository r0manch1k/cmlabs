__all__ = [
    "lagrange",
    "lagrange_remainder",
    "divided_differences",
    "newton",
    "forward_differences",
    "backward_differences",
    "newtonfd",
]

import numpy as np
import scipy.special as sps
import math


def lagrange(xvals, yvals, x):
    r"""Lagrange interpolation polynomial value at :math:`x`.

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
        The y-coordinates of the data points, i.e., f(:math:`x`).
    x : float
        The x-coordinate at which to evaluate the polynomial.

    Returns
    -------
    res: float
        The value of the polynomial at :math:`x`.

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

    where :math:`L_n(x)` is the Lagrange polynomial of degree `n`,
    :math:`R_n(x)` is the remainder term and `n = len(xvals)-1`.

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

          * 'auto' : uses 'exact' if :math:`x` is provided, otherwise uses 'bound'.
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
    >>> x = np.pi / 8
    >>> abs(np.sin(x) - lagrange(xvals, yvals, x))
    0.007941567634910107
    >>> # Exact remainder
    >>> lagrange_remainder(xvals, M, x)
    0.010093189023535093
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
    r"""Return the ordered divided differences list.

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

    where :math:`f(x_0, x_1, \ldots, x_n)` is the divided difference of the function
    at the points :math:`x_0, x_1, \ldots, x_n`.

    It starts moving from the right side of the array, so the iterations look like this:

    .. math::

        \begin{aligned}
            \\
            [f(x_0), f(x_1), \ldots, f(x_n)] &\to
            [f(x_0), f(x_0, x_1), \ldots, f(x_{n-1}, x_n)] \\
            [f(x_0), f(x_0, x_1), \ldots, f(x_{n-1}, x_n)] &\to
            [f(x_0), f(x_0, x_1), f(x_0, x_1, x_3), \ldots, f(x_{n-2}, x_{n-1}, x_n)] \\
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
    r"""Newton interpolation polynomial value at :math:`x`.

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
        The y-coordinates of the data points, i.e., f(:math:`x`).
        Only used if `coef` is not provided.
    coef : array_like, 1-D, optional
        The coefficients of the Newton polynomial.
        If not provided, the function will compute the divided differences list
        using the `divided_differences` function.

    Returns
    -------
    res: float
        The value of the polynomial at :math:`x`.

    See Also
    --------
    divided_differences : Compute the divided differences list.

    Notes
    -----
    The `coef` parameter is optional and can be used to provide the coefficients
    of the Newton polynomial. If not provided, the function will compute the
    divided differences list using the `divided_differences` function.

    The divided differences list has the following structure:

    .. math::

        [f(x_0), f(x_0, x_1), f(x_0, x_1, x_2), \ldots, f(x_0, x_1, \ldots, x_n)]

    Use Horner's method to compute the value of the polynomial at :math:`x`.

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


def forward_differences(yvals):
    r"""Return the forward differences table.

    .. math::

        \begin{gather}
            \Delta^k f_i = \Delta^{k-1} f_{i+1} - \Delta^{k-1} f_i, \\
            f_i = f(x_0 + i \cdot h)
        \end{gather}

    Parameters
    ----------
    yvals : array_like, 1-D
        The y-coordinates of the data points.

    Returns
    -------
    res: array_like, 2-D
        The forward differences table.

    Notes
    -----
    The output is the table with following structure:

    .. math::

        \left[\begin{gather}
        [\Delta^0 f_0 \quad \Delta^0 f_1 \quad \ldots \quad \Delta^0 f_n] \\
        [\Delta^1 f_0 \quad \Delta^1 f_1 \quad \ldots \quad \Delta^1 f_{n-1}] \\
        [\Delta^2 f_0 \quad \Delta^2 f_1 \quad \ldots \quad \Delta^2 f_{n-2}] \\
        \vdots \\
        [\Delta^n f_0]
        \end{gather}\right]

    where `n = len(yvals) - 1`.

    Examples
    --------
    >>> import numpy as np
    >>> from cmlabs.interpolate import forward_differences
    >>> yvals = np.array([1, 3, 2, 5])
    >>> forward_differences(yvals)
    >>> [[1, 3, 2, 5], [2, -1, 3], [-3, 4], [7]]

    """
    n = len(yvals) - 1

    table = [np.array(yvals, dtype=float)]

    for k in range(1, n + 1):
        prev = table[-1]
        diff = [prev[j] - prev[j - 1] for j in range(1, len(prev))]
        table.append(np.array(diff, dtype=float))

    return table


def backward_differences(yvals):
    r"""Return the backward differences table.

    .. math::

        \begin{gather}
            \nabla^k f_{i+1} = \nabla^{k-1} f_{i+1} - \nabla^{k-1} f_i, \\
            f_i = f(x_0 + i \cdot h)
        \end{gather}

    Parameters
    ----------
    yvals : array_like, 1-D
        The y-coordinates of the data points.

    Returns
    -------
    res: array_like, 2-D
        The backward differences table.

    Notes
    -----
    The output is the table with following structure:

    .. math::

        \left[\begin{gather}
        [\nabla^0 f_0 \quad \nabla^0 f_1 \quad \ldots \quad \nabla^0 f_n] \\
        [\nabla^1 f_1 \quad \nabla^1 f_1 \quad \ldots \quad \nabla^1 f_n] \\
        [\nabla^2 f_2 \quad \nabla^2 f_1 \quad \ldots \quad \nabla^2 f_n] \\
        \vdots \\
        [\nabla^n f_n]
        \end{gather}\right]

    where `n = len(yvals) - 1`.

    Examples
    --------
    >>> import numpy as np
    >>> from cmlabs.interpolate import backward_differences
    >>> yvals = np.array([1, 3, 2, 5])
    >>> backward_differences(yvals)
    >>> [[1, 3, 2, 5], [2, -1, 3], [-3, 4], [7]]

    """
    n = len(yvals) - 1

    table = [np.array(yvals, dtype=float)]

    for k in range(1, n + 1):
        prev = table[-1]
        diff = [prev[j + 1] - prev[j] for j in range(0, len(prev) - 1)]
        table.append(np.array(diff, dtype=float))

    return table


def newtonfd(xvals, x, yvals=None, coef=None):
    r"""Newton interpolation polynomial value at :math:`x` using forward differences.

    .. math::

        \begin{gather}
            P_n(x) = \sum_{k=0}^{n} \frac{\Delta^k f(x_0)}{k!h^k} \omega_k(x) \\
            P_n(x) = \sum_{k=0}^{n} C^k_t \Delta^k f(x_0)
        \end{gather}

    where :math:`\Delta^k f_0` is the forward difference of the function.

    Parameters
    ----------
    xvals : array_like, 1-D
        The sorted x-coordinates of the data points.
    x : float
        The x-coordinate at which to evaluate the polynomial.
    yvals : array_like, 1-D, optional
        The y-coordinates of the data points, i.e., f(:math:`x`).
        Only used if `coef` is not provided.
    coef : array_like, 2-D, optional
        The forward differences table.
        If not provided, the function will compute the forward differences table
        using the `forward_differences` function.

    Returns
    -------
    res: float
        The value of the polynomial at :math:`x`.

    See Also
    --------
    forward_differences

    Notes
    -----
    The relation between divided difference and forward difference is
    proved by induction.

    .. math::

        \begin{gather}
            f(x_0, x_1, \ldots, x_k) = \frac{\Delta^k f(x_0)}{k! h^k},
            \quad h = x_1 - x_0 = const
        \end{gather}

    So the Newton polynomial can be written as:

    .. math::

        \begin{gather}
            P_n(x) = \sum_{k=0}^{n} \frac{\Delta^k f(x_0)}{k!h^k} \omega_k(x) \\
        \end{gather}

    Using a variable :math:`t = \frac{x - x_0}{h}` we can rewrite the polynomial as:

    .. math::

        \begin{gather}
            P_n(x) = \sum_{k=0}^{n} \frac{t(t-1)\ldots(t-k+1)}{k!} \Delta^k f(x_0) \\
            C^k_n = \frac{n(n-1)\ldots(n-k+1)}{k!}, \quad k \geq 2, \quad C^0_n = 1,
            \quad C^1_n = n \\
            P_n(x) = \sum_{k=0}^{n} C^k_n \Delta^k f(x_0) \\
        \end{gather}

    Recommended to use this method to compute the polynomial value at :math:`x` near the
    first data point :math:`x_0` to avoid numerical errors.

    Examples
    --------
    >>> import numpy as np
    >>> from cmlabs.interpolate import newtonfd
    >>> xvals = np.array([0, 2, 3, 5])
    >>> yvals = np.array([1, 3, 2, 5])
    >>> x = np.float32(1.5)
    >>> newtonfd(xvals, x, yvals=yvals)
    """

    n = len(xvals) - 1
    h = xvals[1] - xvals[0]
    t = (x - xvals[0]) / h

    if yvals is not None:
        coef = forward_differences(yvals)
    elif coef is None:
        raise ValueError("Either yvals or coef must be provided")

    res = coef[0][0]
    for k in range(1, n):
        res += sps.binom(t, k) * coef[k][0]
    return res
