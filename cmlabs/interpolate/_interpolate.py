__all__ = [
    "lagrange",
    "lagrange_remainder",
    "divided_differences",
    "newton",
    "finite_differences",
    "forward_differences",
    "backward_differences",
    "newtonfd",
    "newtonbd",
    "gaussfd",
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
    divided_differences

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


def finite_differences(yvals):
    r"""Return the finite differences table.

    .. math::

        \begin{gather}
            \Delta^k f(x_i) = f(x_{i+1}) - f(x_i), \\
            \nabla^k f(x_i) = f(x_i) - f(x_{i-1})
        \end{gather}

    Parameters
    ----------
    yvals : array_like, 1-D
        The y-coordinates of the data points.

    Returns
    -------
    res: array_like, 2-D
        The finite differences table.

    Notes
    -----
    The output is the table with following structure:

    .. math::

        \left[\begin{gather}
        [\Delta^0 f(x_0) \quad \Delta^0 f(x_1) \quad \ldots \quad \Delta^0 f(x_n)] \\
        [\Delta^1 f(x_0) \quad \Delta^1 f(x_1) \quad \ldots \quad \Delta^1 f(x_{n-1})]\\
        [\Delta^2 f(x_0) \quad \Delta^2 f(x_1) \quad \ldots \quad \Delta^2 f(x_{n-2})]\\
        \vdots \\
        [\Delta^n f(x_0)]
        \end{gather}\right]

    Or it can be treated as...

    .. math::

        \left[\begin{gather}
        [\nabla^0 f(x_0) \quad \nabla^0 f(x_1) \quad \ldots \quad \nabla^0 f(x_n)] \\
        [\nabla^1 f(x_1) \quad \nabla^1 f(x_1) \quad \ldots \quad \nabla^1 f(x_n)]\\
        [\nabla^2 f(x_2) \quad \nabla^2 f(x_1) \quad \ldots \quad \nabla^2 f(x_n)]\\
        \vdots \\
        [\nabla^n f(x_n)]
        \end{gather}\right]

    where `n = len(yvals) - 1`.

    Examples
    --------
    >>> import numpy as np
    >>> from cmlabs.interpolate import finite_differences
    >>> yvals = np.array([1, 3, 2, 5])
    >>> finite_differences(yvals)
    >>> [[1, 3, 2, 5], [2, -1, 3], [-3, 4], [7]]

    """
    n = len(yvals) - 1

    table = [np.array(yvals, dtype=float)]

    for _ in range(1, n + 1):
        prev = table[-1]
        diff = [prev[j] - prev[j - 1] for j in range(1, len(prev))]
        table.append(np.array(diff, dtype=float))

    return table


def forward_differences(yvals):
    r"""Return the forward differences list for :math:`f(x_0)`.

    .. math::

        \begin{gather}
            \Delta^k f(x_i) = \Delta^{k-1} f(x_{i+1}) - \Delta^{k-1} f(x_i)
        \end{gather}

    Parameters
    ----------
    yvals : array_like, 1-D
        The y-coordinates of the data points.

    Returns
    -------
    res: array_like, 1-D
        The forward differences list.

    Notes
    -----
    The output is the list with following structure:

    .. math::

        [\Delta^0 f(x_0), \Delta^1 f(x_0), \Delta^2 f(x_0), \ldots, \Delta^n f(x_0)]

    where `n = len(yvals) - 1`.

    The forward differences list is used to compute the coefficients
    of the Newton Forward-Difference Formula

    Examples
    --------
    >>> import numpy as np
    >>> from cmlabs.interpolate import forward_differences
    >>> yvals = np.array([1, 3, 2, 5])
    >>> forward_differences(yvals)
    >>> [1, 2, -3, 7]

    """
    table = finite_differences(yvals)
    return np.array([table[k][0] for k in range(len(table))])


def backward_differences(yvals):
    r"""Return the backward differences list for :math:`f(x_n)`.

    .. math::

        \begin{gather}
            \nabla^k f(x_i) = \nabla^{k-1} f(x_{i+1}) - \nabla^{k-1} f(x_i)
        \end{gather}

    Parameters
    ----------
    yvals : array_like, 1-D
        The y-coordinates of the data points.

    Returns
    -------
    res: array_like, 1-D
        The backward differences list.

    Notes
    -----
    The output is the list with following structure:

    .. math::

        [\nabla^0 f_n, \nabla^1 f_n, \nabla^2 f_n, \ldots, \nabla^n f_n]

    where `n = len(yvals) - 1`.

    The backward differences list is used to compute the coefficients
    of the Newton Backward-Difference Formula

    Examples
    --------
    >>> import numpy as np
    >>> from cmlabs.interpolate import backward_differences
    >>> yvals = np.array([1, 3, 2, 5])
    >>> backward_differences(yvals)
    >>> [5, 3, 4, 7]

    """
    table = finite_differences(yvals)
    return np.array([table[k][-1] for k in range(len(table))])


def newtonfd(xvals, x, yvals=None, coef=None):
    r"""Newton's forward interpolation formula.

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
    coef : array_like, 1-D, optional
        The forward differences list.
        If not provided, the function will compute the forward differences table
        using the `forward_differences`.

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
            \quad h = x_{i+1} - x_i = const
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
            P_n(x) = \sum_{k=0}^{n} C^k_t \Delta^k f(x_0) \\
        \end{gather}

    This method is beneficial for determining values of :math:`x` at the
    beginning of a tabulated equally spaced set of values, i.e. :math:`0 < t < 1`.
    Also for extrapolating values of :math:`x` a bit backward (i.e. to the left)
    of :math:`x_0`.

    Examples
    --------
    >>> import numpy as np
    >>> from cmlabs.interpolate import newtonfd
    >>> xvals = np.array([0, 1, 2, 3])
    >>> yvals = np.array([1, 3, 2, 5])
    >>> x = np.float32(0.5)
    >>> newtonfd(xvals, x, yvals=yvals)
    >>> 2.8125

    """
    n = len(xvals) - 1
    h = xvals[1] - xvals[0]
    t = (x - xvals[0]) / h

    if coef is not None and n != len(coef):
        raise ValueError("coef must have the same length as xvals and yvals")

    if yvals is not None:
        coef = forward_differences(yvals)
    elif coef is None:
        raise ValueError("Either yvals or coef must be provided")

    res = coef[0]
    for k in range(1, n + 1):
        res += sps.binom(t, k) * coef[k]
    return res


def newtonbd(xvals, x, yvals=None, coef=None):
    r"""Newton's backward interpolation formula.

    .. math::

        \begin{gather}
            P_n(x) = \sum_{k=0}^{n} \frac{\nabla^k f(x_n)}{k!h^k} \omega_k(x) \\
            P_n(x) = \sum_{k=0}^{n} C^k_t \nabla^k f(x_n)
        \end{gather}

    where :math:`\nabla^k f_n` is the backward difference of the function.

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
        The backward differences list.
        If not provided, the function will compute the backward differences table
        using the `backward_differences`.

    Returns
    -------
    res: float
        The value of the polynomial at :math:`x`.

    See Also
    --------
    backward_differences

    Notes
    -----
    The relation between divided difference and backward difference is
    proved by induction.

    .. math::

        \begin{gather}
            f(x_0, x_1, \ldots, x_k) = \frac{\nabla^k f(x_n)}{k! h^k},
            \quad h = x_i - x_{i-1} = const
        \end{gather}

    So the Newton polynomial can be written as:

    .. math::

        \begin{gather}
            P_n(x) = \sum_{k=0}^{n} \frac{\nabla^k f(x_n)}{k!h^k} \omega_k(x) \\
        \end{gather}

    Using a variable :math:`s = \frac{x - x_n}{h}` we can rewrite the polynomial as:

    .. math::

        \begin{gather}
            P_n(x) = \sum_{k=0}^{n}
            \frac{s(s+1)\ldots(s+k-1)}{k!} \nabla^k f(x_n) \\
            C^k_{-n} = \frac{-n(-n-1)\ldots(-n-k+1)}{k!} = (-1)^k
            \frac{n(n+1)\ldots(n+k-1)}{k!} \\
            P_n(x) = \sum_{k=0}^{n} (-1)^k C^k_{-s} \nabla^k f(x_n) \\
        \end{gather}

    This method is beneficial for determining values of :math:`x` at the end of the
    tabulated equally spaced set of values, i.e. :math:`-1 < t < 0`. Also for
    extrapolating values of :math:`x` a little ahead (to the right) of :math:`x_n`.

    Examples
    --------
    >>> import numpy as np
    >>> from cmlabs.interpolate import newtonbd
    >>> xvals = np.array([0, 1, 2, 3])
    >>> yvals = np.array([1, 3, 2, 5])
    >>> x = np.float32(2.5)
    >>> newtonbd(xvals, x, yvals=yvals)
    2.5625

    """
    n = len(xvals) - 1
    h = xvals[1] - xvals[0]
    t = (x - xvals[-1]) / h

    if coef is not None and n != len(coef):
        raise ValueError("coef must have the same length as xvals and yvals")

    if yvals is not None:
        coef = backward_differences(yvals)
    elif coef is None:
        raise ValueError("Either yvals or coef must be provided")

    res = coef[0]
    for k in range(1, n + 1):
        res += (-1) ** k * sps.binom(-t, k) * coef[k]
    return res


def gaussfd(xvals, x, yvals=None, coef=None, m=None):
    r"""Gauss’s forward interpolation formula.

    .. math::

        \begin{aligned}
            P_n(x) &= \Delta^0 f(x_m) + \\
            &+ t \Delta^1 f(x_m) + \\
            &+ \frac{t(t-1)}{2!} \Delta^2 f(x_{m-1}) + \\
            &+ \frac{(t+1)t(t-1)}{3!} \Delta^3 f(x_{m-1}) + \\
            &+ \frac{(t+1)t(t-1)(t-2)}{4!} \Delta^4 f(x_{m-2}) + \\
            &+ \frac{(t+2)(t+1)t(t-1)(t-2)}{5!} \Delta^5 f(x_{m-2}) + \\
            &+ \frac{(t+2)(t+1)t(t-1)(t-2)(t-3)}{6!} \Delta^6 f(x_{m-3}) + \ldots
        \end{aligned}

    where :math:`x_m` is the midpoint of the data point (if even then :math:`x_{n/2}`)
    and :math:`t = \frac{x-x_m}{h}`.

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
        If not provided, the function will compute the finite differences table
        using the `finite_differences` function.
    m : int, optional
        The index of the midpoint of the data points.
        If not provided, the function will compute it using the formula:
        :math:`m = \frac{n}{2}` if `n` is even, otherwise :math:`m = \frac{n+1}{2}`.

    Returns
    -------
    res: float
        The value of the polynomial at :math:`x`.

    See Also
    --------
    finite_differences

    Notes
    -----
    Gauss’s forward interpolation formula is best applicable for determining
    the values near the middle of the table.

    .. math::

        x = x_m + t h, \quad 0 < t < \frac{1}{2}

    Examples
    --------
    >>> import numpy as np
    >>> from cmlabs.interpolate import gaussfd
    >>> xvals = np.array([0, 1, 2, 3])
    >>> yvals = np.array([1, 3, 2, 5])
    >>> x = np.float32(1.5)
    >>> gaussfd(xvals, x, yvals=yvals)
    """
    n = len(xvals) - 1

    if m is not None and m >= n:
        raise ValueError("m must be less than n")
    elif m is None:
        m = (n + 1) // 2 if n % 2 == 0 else n // 2

    h = xvals[1] - xvals[0]
    t = (x - xvals[m]) / h

    if coef is not None and n != len(coef):
        raise ValueError("coef must have the same length as xvals and yvals")

    if yvals is not None:
        coef = finite_differences(yvals)
    elif coef is None:
        raise ValueError("Either yvals or coef must be provided")

    res = coef[m][0]

    prod = 1
    # sign = 1

    # TODO: dodelat'
    for k in range(1, len(coef)):
        index = m - (k - 1) // 2

        # prod *= t + sign * ((k - 1) // 2)
        # sign *= -1
        prod *= (t - (k - 1) // 2) if k % 2 == 0 else (t + (k // 2))

        term = (prod / math.factorial(k)) * coef[index][k]
        res += term

    return res
