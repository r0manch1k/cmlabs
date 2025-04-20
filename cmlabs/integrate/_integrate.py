__all__ = [
    "rectangle",
    "midpoint",
    "trapezoid",
    "simpsonq",
    "simpsonc",
    "weddles",
    "newton_cotes",
]

from typing import List
import numpy as np


def rectangle(
    xvals: List[float] | np.ndarray,
    yvals: List[float] | np.ndarray,
    method: str = "left",
) -> float:
    r"""Composite rectangle method for numerical integration.

    .. math::

        \begin{gather}
            \int_{a}^{b} f(x) \, dx \approx \sum_{i=0}^{n-1} f(x_i)
            \cdot (x_{i+1} - x_i) \\
            \int_{a}^{b} f(x) \, dx \approx \sum_{i=1}^{n} f(x_i) \cdot (x_i - x_{i-1})
        \end{gather}

    Parameters
    ----------
    xvals : array_like, 1-D
        The sorted x-coordinates of the data points.
    yvals : array_like, 1-D
        The y-coordinates of the data points, i.e., :math:`f(x)`.
    method : str, optional, default: 'left'
        The method to use for the rectangle rule. Options are:

          * 'left'
          * 'right'

    Returns
    -------
    float
        The approximate value of the integral.

    Notes
    -----
    The rectangle rule approximates the area under the curve by dividing it into
    rectangles. The height of each rectangle is determined by the function value
    at the left or right endpoint of the interval, depending on the chosen method.

    These methods are first-order accurate. The error is approximately:

    .. math::

        \begin{gather}
            |R_n(f)| \leq \frac{M(b-a)}{2} h, \\
            \max_{x \in [a, b]} |f'(x)| \leq M
        \end{gather}

    where :math:`h` is the width of the rectangles and depends on the number of
    intervals :math:`n`.

    Examples
    --------
    >>> # import numpy as np
    >>> # from cmlabs.integrate import rectangle
    >>> # xvals = np.array([0, 1, 2, 3])
    >>> # yvals = np.array([0, 1, 4, 9])
    >>> rectangle(xvals, yvals, method='left')
    5.0
    """
    if len(xvals) != len(yvals):
        raise ValueError("xvals and yvals must have the same length.")

    if len(xvals) < 2:
        raise ValueError("At least two points are required for integration.")

    if method not in ["left", "right"]:
        raise ValueError("Method must be 'left' or 'right'.")

    dx = np.diff(xvals)

    if method == "left":
        heights = yvals[:-1]
    elif method == "right":
        heights = yvals[1:]

    areas = heights * dx
    integral = np.sum(areas)

    return float(integral)


def midpoint(
    xvals: List[float] | np.ndarray,
    yvals: List[float] | np.ndarray,
) -> float:
    r"""Composite midpoint method for numerical integration.

    .. math::

        \begin{gather}
            \int_{a}^{b} f(x) \, dx \approx \sum_{i=0}^{n-1}
            f\left(\frac{x_i + x_{i+1}}{2}\right) (x_{i+1} - x_i)
        \end{gather}

    Parameters
    ----------
    xvals : array_like, 1-D
        The sorted x-coordinates of the data points.
    yvals : array_like, 1-D
        The y-coordinates of the data points, i.e., :math:`f(x)`.

    Returns
    -------
    float
        The approximate value of the integral.

    Notes
    -----
    The midpoint rule approximates the area under the curve by dividing it into
    rectangles. The height of each rectangle is determined by the function value
    at the midpoint of the interval.

    The :math:`\textit{xvals}` array is treated as

    .. math::

        [x_0, x_{\frac{1}{2}}, x_1, x_{\frac{3}{2}}, x_2, \ldots, x_{\frac{n}{2}}]

    so :math:`n` must be even. The same with the :math:`\textit{yvals}` array.

    This method is second-order accurate. The error is approximately:

    .. math::

        |R_n(f)| \leq \frac{M(b-a)}{24} h^2, \quad \max_{x \in [a, b]} |f''(x)| \leq M

    where :math:`h` is the width of the rectangles and depends on the number of
    intervals :math:`n`.

    Examples
    --------
    >>> # import numpy as np
    >>> # from cmlabs.integrate import midpoint
    >>> # xvals = np.array([0, 1, 2, 3, 4])
    >>> # yvals = np.array([0, 1, 4, 9, 16])
    >>> midpoint(xvals, yvals)
    20.0
    """
    if len(xvals) != len(yvals):
        raise ValueError("xvals and yvals must have the same length.")

    if len(xvals) < 2:
        raise ValueError("At least two points are required for integration.")

    n = len(xvals) - 1

    if n % 2 != 0:
        raise ValueError("xvals must fit midpoint rule formula.")

    integral = 0.0

    for i in range(1, n + 1, 2):
        integral += (xvals[i + 1] - xvals[i - 1]) * yvals[i]

    return float(integral)


def trapezoid(
    xvals: List[float] | np.ndarray,
    yvals: List[float] | np.ndarray,
) -> float:
    r"""Composite trapezoid method for numerical integration.

    .. math::

        \begin{gather}
            \int_{a}^{b} f(x) \, dx \approx \sum_{i=0}^{n-1}
            \frac{f(x_i) + f(x_{i+1})}{2} (x_{i+1} - x_i)
        \end{gather}

    Parameters
    ----------
    xvals : array_like, 1-D
        The sorted x-coordinates of the data points.
    yvals : array_like, 1-D
        The y-coordinates of the data points, i.e., :math:`f(x)`.

    Returns
    -------
    float
        The approximate value of the integral.

    Notes
    -----
    The trapezoid rule approximates the area under the curve by dividing it into
    trapezoids. The height of each trapezoid is determined by the function value
    at the endpoints of the interval.

    This method is second-order accurate. The error is approximately:

    .. math::

        |R_n(f)| \leq \frac{M(b-a)}{12} h^2, \quad \max_{x \in [a, b]} |f''(x)| \leq M

    where :math:`h` is the width of the rectangles and depends on the number of
    intervals :math:`n`.

    Examples
    --------
    >>> # import numpy as np
    >>> # from cmlabs.integrate import trapezoid
    >>> # xvals = np.array([0, 1, 2, 3])
    >>> # yvals = np.array([0, 1, 4, 9])
    >>> trapezoid(xvals, yvals)
    9.5
    """
    if len(xvals) != len(yvals):
        raise ValueError("xvals and yvals must have the same length.")

    if len(xvals) < 2:
        raise ValueError("At least two points are required for integration.")

    n = len(xvals) - 1

    integral = 0.0

    for i in range(n):
        integral += (xvals[i + 1] - xvals[i]) * (yvals[i] + yvals[i + 1]) / 2.0

    return float(integral)


def simpsonq(
    xvals: List[float] | np.ndarray,
    yvals: List[float] | np.ndarray,
) -> float:
    r"""Composite Simpson's 1/3 method using quadratic polynomial.

    .. math::

        \begin{gather}
            \int_{a}^{b} f(x) \, dx \approx \sum_{i=0}^{n-1}
            \frac{h}{3} \left[
            f(x_i) + 4f\left(\frac{x_i + x_{i+1}}{2}\right) + f(x_{i+1})\right]
        \end{gather}

    Parameters
    ----------
    xvals : array_like, 1-D
        The sorted x-coordinates of the data points.
    yvals : array_like, 1-D
        The y-coordinates of the data points, i.e., :math:`f(x)`.

    Returns
    -------
    float
        The approximate value of the integral.

    Notes
    -----
    Simpson's 1/3 rule approximates the area under the curve by dividing it into
    parabolas. The height of each parabola is determined by the function value
    at the endpoints and the midpoint of the interval.

    This method is fourth-order accurate. The error is approximately:

    .. math::

        \begin{gather}
            R_{2,i}(f) = \int_{x_i}^{x_{i+1}} \left(f(x) - L_{2, i}(x) \right) dx
            = -\frac{h^5}{90}f^{(4)}(\xi_i) \\
            R_n(f) = \sum_{i=0}^{n-1} R_{2,i}(f) =
            \sum_{i=0}^{n-1} - \frac{h^5}{90}f^{(4)}(\xi_i),
            \quad \xi_i \in [x_i, x_{i+1}] \\
            R_n(f) = -\frac{h^5}{90} \cdot \frac{b-a}{2h} f^{(4)}(\xi) =
            -\frac{f(\xi)(b-a)}{180} h^4, \quad \xi \in [a, b] \\
            |R_n(f)| \leq \frac{M(b-a)}{180} h^4,
            \quad \max_{x \in [a, b]} |f^{(4)}(x)| \leq M
        \end{gather}

    where :math:`h` is the width of the rectangles and depends on the number of
    intervals :math:`n`.

    The :math:`\textit{xvals}` array is treated as

    .. math::

        [x_0, x_{\frac{1}{2}}, x_1, x_{\frac{3}{2}}, x_2, \ldots, x_{\frac{n}{2}}]

    so :math:`n` must be even. The same with the :math:`\textit{yvals}` array.

    Examples
    --------
    >>> # import numpy as np
    >>> # from cmlabs.integrate import simpsonq
    >>> # xvals = np.array([0, 1, 2, 3, 4])
    >>> # yvals = np.array([0, 1, 4, 9, 16])
    >>> simpsonq(xvals, yvals)
    21.333333333333332
    """
    if len(xvals) != len(yvals):
        raise ValueError("xvals and yvals must have the same length.")

    if len(xvals) < 2:
        raise ValueError("At least two points are required for integration.")

    n = len(xvals) - 1

    if n % 2 != 0:
        raise ValueError("xvals must fit Simpson's 1/3 rule formula.")

    h = (xvals[-1] - xvals[0]) / n

    integral = 0.0

    for i in range(0, n, 2):
        integral += yvals[i] + 4 * yvals[i + 1] + yvals[i + 2]

    integral *= h / 3

    return float(integral)


def simpsonc(
    xvals: List[float] | np.ndarray,
    yvals: List[float] | np.ndarray,
) -> float:
    r"""Composite Simpson's 3/8 method using cubic polynomial.

    .. math::

        \begin{gather}
            \int_{a}^{b} f(x) \, dx \approx \sum_{i=0}^{n-1}
            \frac{3h}{8} \left[
            f(x_i) + 3f\left(\frac{x_i + x_{i+1}}{3}\right) +
            3f\left(\frac{2(x_i + x_{i+1})}{3}\right) + f(x_{i+1})\right]
        \end{gather}

    Parameters
    ----------
    xvals : array_like, 1-D
        The sorted x-coordinates of the data points.
    yvals : array_like, 1-D
        The y-coordinates of the data points, i.e., :math:`f(x)`.

    Returns
    -------
    float
        The approximate value of the integral.

    Notes
    -----
    Simpson's 3/8 rule approximates the area under the curve by dividing it into
    cubic polynomials. The height of each polynomial is determined by the function
    value at the endpoints and midpoints of the interval.

    This method is fourth-order accurate. The error is approximately:

    .. math::

        \begin{gather}
            R_{3,i}(f) = \int_{x_i}^{x_{i+1}} \left(f(x) - L_{3, i}(x) \right) dx
            = -\frac{3h^5}{80}f^{(4)}(\xi_i) \\
            R_n(f) = \sum_{i=0}^{n-1} R_{3,i}(f) =
            \sum_{i=0}^{n-1} - \frac{3h^5}{80}f^{(4)}(\xi_i),
            \quad \xi_i \in [x_i, x_{i+1}] \\
            R_n(f) = -\frac{3h^5}{80} \cdot \frac{b-a}{3h} f^{(4)}(\xi) =
            -\frac{f^{(4)}(\xi)(b-a)}{80} h^4, \quad \xi \in [a, b] \\
            \quad \xi \in [a, b] \\
            |R_n(f)| \leq \frac{M(b-a)}{80} h^4,
            \quad \max_{x \in [a, b]} |f^{(4)}(x)| \leq M
        \end{gather}

    where :math:`h` is the width of the rectangles and depends on the number of
    intervals :math:`n`.

    The :math:`\textit{xvals}` array is treated as

    .. math::

        [x_0, x_{\frac{1}{3}}, x_{\frac{2}{3}}, x_1, x_{\frac{4}{3}},
        x_{\frac{5}{3}}, x_2, \ldots, x_{\frac{n}{3}}]

    so :math:`n` must be a multiple of 3. The same with the :math:`\textit{yvals}`

    Examples
    --------
    >>> # import numpy as np
    >>> # from cmlabs.integrate import simpsonc
    >>> # xvals = np.array([0, 1, 2, 3, 4, 5, 6])
    >>> # yvals = np.array([0, 1, 4, 9, 16, 25, 36])
    >>> simpsonc(xvals, yvals)
    72.0
    """
    if len(xvals) != len(yvals):
        raise ValueError("xvals and yvals must have the same length.")

    if len(xvals) < 2:
        raise ValueError("At least two points are required for integration.")

    n = len(xvals) - 1

    if n % 3 != 0:
        raise ValueError("xvals must fit Simpson's 3/8 rule formula.")

    h = (xvals[-1] - xvals[0]) / n

    integral = 0.0

    for i in range(0, n, 3):
        integral += yvals[i] + 3 * yvals[i + 1] + 3 * yvals[i + 2] + yvals[i + 3]

    integral *= 3 * h / 8

    return float(integral)


def weddles(
    xvals: List[float] | np.ndarray,
    yvals: List[float] | np.ndarray,
) -> float:
    r"""Composite Weddle's method for numerical integration.

    .. math::

        \begin{aligned}
            \int_{a}^{b} f(x) \, dx & \approx \sum_{i=0}^{n-1}
            \frac{3h}{10} \bigg[f(x_i) + 5f\left(\frac{x_i + x_{i+1}}{6}\right) +
            f\left(\frac{2(x_i + x_{i+1})}{6}\right) + \\
            & \quad + 6f\left(\frac{3(x_i + x_{i+1})}{6}\right) +
            f\left(\frac{4(x_i + x_{i+1})}{6}\right) +
            5f\left(\frac{5(x_i + x_{i+1})}{6}\right) +
            f(x_{i+1})\bigg]
        \end{aligned}

    Parameters
    ----------
    xvals : array_like, 1-D
        The sorted x-coordinates of the data points.
    yvals : array_like, 1-D
        The y-coordinates of the data points, i.e., :math:`f(x)`.

    Returns
    -------
    float
        The approximate value of the integral.

    Notes
    -----
    Weddle's rule approximates the area under the curve by dividing it into
    parabolas. The height of each parabola is determined by the function value
    at the endpoints and the midpoint of the interval.

    This method is sixth-order accurate. The error is approximately:

    .. math::

        |R_n(f)| \leq \frac{M(b-a)}{1400} h^6,
        \quad \max_{x \in [a, b]} |f^{(6)}(x)| \leq M

    where :math:`h` is the width of the rectangles and depends on the number of
    intervals :math:`n`.

    The :math:`\textit{xvals}` array is treated as

    .. math::

        [x_0, x_{\frac{1}{6}}, x_{\frac{2}{6}}, x_{\frac{3}{6}},
        x_{\frac{4}{6}}, x_{\frac{5}{6}}, x_1, \ldots, x_{\frac{n}{6}}]

    so :math:`n` must be a multiple of 6. The same with the :math:`\textit{yvals}`

    Examples
    --------
    >>> # import numpy as np
    >>> # from cmlabs.integrate import weddles
    >>> # xvals = np.array([0, 1, 2, 3, 4, 5, 6])
    >>> # yvals = np.array([0, 1, 4, 9, 16, 25, 36])
    >>> weddles(xvals, yvals)
    72.0
    """
    if len(xvals) != len(yvals):
        raise ValueError("xvals and yvals must have the same length.")

    if len(xvals) < 2:
        raise ValueError("At least two points are required for integration.")

    n = len(xvals) - 1

    if n % 6 != 0:
        raise ValueError("xvals must fit Weddle's rule formula.")

    h = (xvals[-1] - xvals[0]) / n

    integral = 0.0

    for i in range(0, n, 6):
        integral += (
            yvals[i]
            + 5 * yvals[i + 1]
            + yvals[i + 2]
            + 6 * yvals[i + 3]
            + yvals[i + 4]
            + 5 * yvals[i + 5]
            + yvals[i + 6]
        )

    integral *= 3 * h / 10

    return float(integral)


def newton_cotes(
    xvals: List[float] | np.ndarray,
    yvals: List[float] | np.ndarray,
    coef: List[float] | np.ndarray,
) -> float:
    r"""Composite Newton-Cotes method for numerical integration.

    .. math::

        \begin{gather}
            I_n = (b - a) \cdot \sum_{i=0}^{n} c_i f(x_i) \\
        \end{gather}

    Parameters
    ----------
    xvals : array_like, 1-D
        The sorted x-coordinates of the data points.
    yvals : array_like, 1-D
        The y-coordinates of the data points, i.e., :math:`f(x)`.
    coef : array_like, 1-D
        The coefficients of the Newton-Cotes formula.

    Returns
    -------
    float
        The approximate value of the integral.

    See Also
    --------
    trapezoid, simpsonq, simpsonc, weddles

    Notes
    -----
    The Newton-Cotes method approximates the area under the curve by dividing it
    into polynomials. The height of each polynomial is determined by the function
    value at the endpoints and midpoints of the interval.

    :math:`\textit{coef}` is the coefficients of the Newton-Cotes formula. The
    coefficients are determined by the order of the polynomial used in the
    approximation. The coefficients are usually derived from the Lagrange
    interpolation polynomial.
    """
    if len(xvals) != len(yvals):
        raise ValueError("xvals and yvals must have the same length.")

    if len(xvals) < 2:
        raise ValueError("At least two points are required for integration.")

    if len(coef) != len(yvals):
        raise ValueError("coef must have the same length as yvals.")

    n = len(xvals) - 1

    integral = np.sum([coef[i] * yvals[i] for i in range(n + 1)])
    integral *= xvals[-1] - xvals[0]

    return float(integral)
