import numpy as np

__all__ = ["find_root_brackets", "bisect", "newton", "secant"]


def find_root_brackets(f: callable, a: float, b: float, bins: int = 10) -> list:
    r"""Find the root brackets of a function.

    Parameters
    ----------
    f : callable
        The function to find the root brackets for.
    a : float
        The lower bound of the interval.
    b : float
        The upper bound of the interval.
    bins : int, optional
        The number of bins to use for the search.

    Returns
    -------
    list of tuples
        A list of tuples containing the lower and upper bounds of each root bracket.

    Notes
    -----
    The function passed to this method must be continuous on the interval
    :math:`[a, b]`. According to the Bolzano–Cauchy theorem, if a continuous
    function :math:`f` satisfies :math:`f(a) \cdot f(b) < 0`, then there exists
    at least one point :math:`c \in (a, b)` such that :math:`f(c) = 0`.

    Examples
    --------
    >>> from cmlabs.optimize import find_root_brackets
    >>> f = lambda x: x**2 - 4
    >>> a = 0
    >>> b = 10
    >>> bins = 10
    >>> find_root_brackets(f, a, b, bins)
    >>> # [(np.float64(1.1111111111111112), np.float64(2.2222222222222223))]
    """
    x = np.linspace(a, b, bins)

    intervals = []
    for i in range(len(x) - 1):
        if np.sign(f(x[i])) != np.sign(f(x[i + 1])):
            intervals.append((x[i], x[i + 1]))

    return intervals


def bisect(f: callable, bracket: list, xtol: float = None, ytol: float = None) -> float:
    r"""Bisection method for finding roots.

    Parameters
    ----------
    f : callable
        The function to find the root of.
    bracket : A sequence of 2 floats
        THe root interval. The function must have different signs at the endpoints.
    xtol : float, optional
        The absolute error in x required to declare convergence.
    ytol : float, optional
        The absolute error in f(x) required to declare convergence.

    See Also
    --------
    find_root_brackets

    Notes
    -----
    The function passed to this method must satisfy The Bolzano–Cauchy theorem.

    Parameter `xtol` is the absolute error in x required to declare convergence.

    .. math::

        \begin{aligned}
            |x^* - x| < \text{xtol}
        \end{aligned}

    where :math:`x^*` is the root of the function.

    Parameter `ytol` is the absolute error in f(x) required to declare convergence.

    .. math::

        \begin{aligned}
            |f(x^*) - f(x)| < \text{ytol}
        \end{aligned}

    where :math:`x^*` is the root of the function.

    You can define both `xtol` and `ytol` to declare convergence. The first one
    that is satisfied will be used to declare convergence.

    Examples
    --------
    >>> from cmlabs.optimize import bisect
    >>> f = lambda x: x**2 - 4
    >>> bracket = [0, 10]
    >>> bisect(f, bracket)
    >>> # 1.9999980926513672
    """
    if xtol is None and ytol is None:
        raise ValueError("At least one of xtol or ytol must be provided.")

    x_n, x_k = np.array(bracket)

    if np.sign(f(x_n)) == np.sign(f(x_k)):
        raise ValueError("The function must have different signs at the endpoints.")

    if f(x_n) == 0 or (ytol and abs(f(x_n)) < ytol):
        return x_n

    if f(x_k) == 0 or (ytol and abs(f(x_k)) < ytol):
        return x_k

    while True:
        if xtol and abs(x_k - x_n) < xtol:
            break

        x_m = np.float64((x_n + x_k) / 2)

        f_m = np.float64(f(x_m))

        if np.sign(f(x_n)) != np.sign(f_m):
            x_k = x_m
        else:
            x_n = x_m

        if ytol and abs(f_m) < ytol:
            break

    return x_m


def newton(
    f: callable, df: callable, x0: float, xtol: float = None, ytol: float = None
) -> float:
    r"""Newton's method for finding roots.

    Parameters
    ----------
    f : callable
        The function to find the root of.
    df : callable
        The derivative of the function.
    x0 : float
        The initial guess for the root.
    xtol : float, optional
        The absolute error in x required to declare convergence.
    ytol : float, optional
        The absolute error in f(x) required to declare convergence.

    See Also
    --------
    find_root_brackets
    bisect

    Notes
    -----
    This method is an iterative numerical technique for finding a root of a real-valued
    function :math:`f(x)`, based on linear approximation using the first derivative.
    The method requires the function to be sufficiently smooth and its derivative
    behavior to meet certain conditions to ensure convergence.

    .. math::

        \begin{aligned}
            \overline{x}_{n+1} = \overline{x}_n -
            \frac{f(\overline{x}_n)}{f'(\overline{x}_n)}
        \end{aligned}

    where :math:`\overline{x}_n` is the current approximation of the root,
    and :math:`\overline{x}_{n+1}` is the next approximation.

    The function :math:`f(x)` must be continuous and differentiable in the neighborhood
    of the root. The derivative :math:`f'(x)` must not be zero at the root.

    The starting point :math:`x_0` should be chosen close to the actual root to ensure
    convergence. A practical sufficient condition for local convergence is
    :math:`f(x_0) \cdot f''(x_0) > 0`.

    Examples
    --------
    >>> from cmlabs.optimize import newton
    >>> f = lambda x: x**2 - 4
    >>> df = lambda x: 2 * x
    >>> x0 = 5.0
    >>> newton(f, df, x0, xtol=1e-5, ytol=1e-5)
    >>> # 2.0000051812194735
    """
    if xtol is None and ytol is None:
        raise ValueError("At least one of xtol or ytol must be provided.")

    x_n = np.float64(x0)
    f_n = np.float64(f(x_n))
    df_n = np.float64(df(x_n))

    if df_n == 0:
        raise ValueError("The derivative must not be zero at the root.")

    if f_n == 0 or (ytol and abs(f_n) < ytol):
        return x_n

    while True:
        if df_n == 0:
            raise ValueError("The derivative must not be zero at the root.")

        x_n1 = np.float64(x_n - f_n / df_n)

        if xtol and abs(x_n1 - x_n) < xtol:
            break

        f_n1 = np.float64(f(x_n1))
        df_n1 = np.float64(df(x_n1))

        if ytol and abs(f_n1) < ytol:
            break

        x_n = x_n1
        f_n = f_n1
        df_n = df_n1

    return x_n


def secant(
    f: callable, x0: float, x1: float, xtol: float = None, ytol: float = None
) -> float:
    r"""Secant method for finding roots.

    Parameters
    ----------
    f : callable
        The function to find the root of.
    x0 : float
        The first initial guess for the root.
    x1 : float
        The second initial guess for the root.
    xtol : float, optional
        The absolute error in x required to declare convergence.
    ytol : float, optional
        The absolute error in f(x) required to declare convergence.

    See Also
    --------
    find_root_brackets
    bisect

    Notes
    -----
    This method is an iterative numerical technique for finding a root of a real-valued
    function :math:`f(x)` using two initial guesses. It is a generalization of the
    Newton-Raphson method and does not require the computation of the derivative.

    .. math::

        \begin{aligned}
            \overline{x}_{n+1} = \overline{x}_n -
            \frac{f(\overline{x}_n) (\overline{x}_n - \overline{x}_{n-1})}
            {f(\overline{x}_n) - f(\overline{x}_{n-1})}
        \end{aligned}

    where :math:`\overline{x}_n` is the current approximation of the root,
    and :math:`\overline{x}_{n+1}` is the next approximation.

    The function :math:`f(x)` must be continuous in the neighborhood of the root.

    Examples
    --------
    >>> from cmlabs.optimize import secant
    >>> f = lambda x: x**2 - 4
    >>> x0 = 0.0
    >>> x1 = 10.0
    >>> secant(f, x0, x1, xtol=1e-5, ytol=1e-5)
    >>> # 2.000000000826115
    """
    if xtol is None and ytol is None:
        raise ValueError("At least one of xtol or ytol must be provided.")

    x_n = np.float64(x0)
    x_n1 = np.float64(x1)
    f_n = np.float64(f(x_n))
    f_n1 = np.float64(f(x_n1))

    if f_n == 0 or (ytol and abs(f_n) < ytol):
        return x_n

    if f_n1 == 0 or (ytol and abs(f_n1) < ytol):
        return x_n1

    while True:
        if f_n == f_n1:
            raise ValueError(
                "The function values at the current guesses must not be equal."
            )

        x_n2 = np.float64(x_n1 - f_n1 / (f_n1 - f_n) * (x_n1 - x_n))

        if xtol and abs(x_n2 - x_n1) < xtol:

            break

        f_n2 = np.float64(f(x_n2))

        if ytol and abs(f_n2) < ytol:
            break

        x_n, x_n1 = x_n1, x_n2
        f_n, f_n1 = f_n1, f_n2

    return x_n2
