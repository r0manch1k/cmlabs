import numpy as np
from cmlabs.linalg import thomas

__all__ = ["CubicSpline"]


class CubicSpline:
    r"""Cubic spline data interpolator.

    .. math::

        \begin{gather}
            S(x) = c_i + d_i (x - x_i) + a_i (x - x_i)^2 + b_i (x - x_i)^3, \\
            S(x) \in P_{3i}, \quad S(x) \in C^2[a, b], \quad S(x_i) = y_i
        \end{gather}

    where :math:`P_{3i}` is the set of cubic polynomials, :math:`C^2[a, b]` is the
    set of twice continuously differentiable functions, and :math:`y_i` is the
    interpolated value at :math:`x_i`.

    Parameters
    ----------
    xvals : array-like
        The x-coordinates of the data points.
    yvals : array-like
        The y-coordinates of the data points.
    bc_type : str, optional
        Boundary condition type. Two additional equations, given by the
        boundary conditions, are required to determine all coefficients of
        polynomials on each segment.

        * 'clamped': The first derivative at curves ends are equal, i.e.,
          :math:`S'(a) = f'(a)` and :math:`S'(b) = f'(b)`.
        * 'second': The second derivative at curves ends are equal, i.e.,
          :math:`S''(a) = f''(a)` and :math:`S''(b) = f''(b)`.
        * 'periodic': The interpolated functions is assumed to be periodic
          of period, i.e., :math:`S^i(a) = S^i(b)` for :math:`i = 0, 1, 2`.
        * 'not-a-knot' (default): The first and second segment at a curve end
          are the same polynomial. It is a good default when there is no
          information on boundary conditions, i.e., :math:`S'''(x_1 - 0) =
          S'''(x_1 + 0)` and :math:`S'''(x_{n-1} - 0) = S'''(x_{n-1} + 0)`.

    See Also
    --------
    cmlabs.linalg.thomas

    Notes
    -----
    Сonsider the construction of a cubic spline interpolant based on given data points.
    The spline will be represented piecewise on each subinterval.

    Assume that :math:`S'(x_i) = y'_i = m_i` - first derivative of the
    interpolated function at the node :math:`x_i`. So the polynomial looks like this:

    .. math::

        \begin{gather}
            S(x) = y_i + m_i (x - x_i) + a_i \frac{(x - x_i)^2}{2}
            + b_i \frac{(x - x_i)^3}{6}, \\
            a_i = \frac{6}{h_{i+1}} \left(\frac{y_{i+1} - y_i}{h_{i+1}} -
            \frac{m_{i+1} + 2m_i}{3}\right), \quad
            b_i = \frac{12}{h_{i+1}^2} \left(\frac{m_{i+1} + m_i}{2} -
            \frac{y_{i+1} - y_i}{h_{i+1}}\right)
        \end{gather}

    where :math:`h_{i+1} = x_{i+1} - x_i` is the distance between the nodes.

    Using second condition :math:`S''(x_i - 0) = S''(x_i + 0)` we can
    set up a system of equations for the unknown slopes :math:`m_i`:

    .. math::

        \begin{gather}
            \mu_i m_{i-1} + 2 m_i + \lambda_i m_{i+1} = 3 \left(\lambda_i
            \frac{y_{i+1} - y_i}{h_{i+1}} + \mu_i \frac{y_i - y_{i-1}}{h_i}\right),
            \quad i = \overline{2, N-2}
        \end{gather}

    where :math:`\lambda_i = \frac{h_{i+1}}{h_i + h_{i+1}}` and
    :math:`\mu_i = \frac{h_i}{h_i + h_{i+1}}` are the coefficients of the
    system. The first and last equations are given by the boundary conditions.

    We impose the not-a-knot condition, which requires that the third derivative
    be continuous at the second and penultimate nodes. The not-a-knot condition
    effectively treats the first two segments and the last two segments as parts
    of the same cubic polynomial:

    .. math::

        \begin{gather}
            S'''(x_1 - 0) = S'''(x_1 + 0), \quad
            S'''(x_{n-1} - 0) = S'''(x_{n-1} + 0) \\
            b_0 = b_1, \quad b_{n-2} = b_{n-1} \\
            (1 + \gamma_1) m_1 + \gamma_1 m_2 = \lambda_1 (3 + 2 \gamma_1)
            \frac{y_2 - y_1}{h_2} + \mu_1 \frac{y_1 - y_0}{h_1} \\
            \gamma_N m_{N-2} + (1 + \gamma_N) m_{N-1} = \mu_{N-1}
            (3 + 2 \gamma_N) \frac{y_{N-1} - y_{N-2}}{h_{N-1}} +
            \lambda_{N-1} \frac{y_N - y_{N-1}}{h_N} \\
        \end{gather}

    where :math:`\gamma_1 = \frac{h_1}{h_2}` and :math:`\gamma_N = \frac{h_N}{h_{N-1}}`.

    The system of equations can be solved using the Thomas algorithm. The tridiagonal
    system is represented in the form:

    .. math::

        \left(
            \begin{array}{cccccc}
                1 + \gamma_1 & \gamma_1 & 0 & \cdots & 0 & 0 \\
                \mu_2 & 2 & \lambda_2 & \ddots & \vdots & \vdots \\
                0 & \mu_3 & 2 & \ddots & 0 & 0 \\
                \vdots & \ddots & \ddots & \ddots & \lambda_{N-3} & 0 \\
                0 & \cdots & 0 & \mu_{N-2} & 2 & \lambda_{N-2} \\
                0 & \cdots & 0 & 0 & \gamma_N & 1 + \gamma_N
            \end{array}
        \right)
        \left(
            \begin{array}{c}
                m_1 \\ m_2 \\ m_3 \\ \vdots \\ m_{N-1} \\ m_N
            \end{array}
        \right)
        =
        \left(
            \begin{array}{c}
                \hat{g}_1 \\ g_2 \\ g_3 \\ \vdots \\ g_{N-2} \\ \hat{g}_{N-1}
            \end{array}
        \right)

    where

    .. math::

        \begin{gather}
            \hat{g}_1 = \lambda_1 (3 + 2 \gamma_1) \frac{y_2 - y_1}{h_2} +
            \mu_1 \frac{y_1 - y_0}{h_1} \\
            g_i = 3 \left(\lambda_i \frac{y_{i+1} - y_i}{h_{i+1}} +
            \mu_i \frac{y_i - y_{i-1}}{h_i}\right), \quad i = \overline{2, N-2} \\
            \hat{g}_{N-1} = \mu_{N-1} (3 + 2 \gamma_N) \frac{y_{N-1} - y_{N-2}}
            {h_{N-1}} + \lambda_{N-1} \frac{y_N - y_{N-1}}{h_N}
        \end{gather}

    Now consider the second derivative of the spline:

    .. math::

        \begin{gather}
            S'(x) = m_i + \frac{6(x - x_i)}{h_{i+1}} \left(\frac{y_{i+1} - y_i}
            {h_{i+1}} - \frac{m_{i+1} + 2m_i}{3}\right) + \\
            + \frac{6(x - x_i)^2}{h_{i+1}^2}
            \left(\frac{m_{i+1} + m_i}{2} - \frac{y_{i+1} - y_i}{h_{i+1}}\right) \\
            S''(x) = \frac{6}{h_{i+1}} \left(\frac{y_{i+1} - y_i}{h_{i+1}} -
            \frac{m_{i+1} + 2m_i}{3}\right) + \frac{12(x - x_i)}{h_{i+1}^2}
            \left(\frac{m_{i+1} + m_i}{2} - \frac{y_{i+1} - y_i}{h_{i+1}}\right)
        \end{gather}

    Examples
    --------
    >>> import numpy as np
    >>> from cmlabs.interpolate import CubicSpline
    >>> x = np.linspace(0, 10, 5)
    >>> y = np.sin(x)
    >>> cs = CubicSpline(x, y)
    >>> cs.interpolate(5)
    -0.9589242746631386
    """

    def __init__(self, xvals, yvals, bc_type="not-a-knot"):
        """
        Initialize the cubic spline with given x and y data points.

        Parameters
        ----------
        xvals : array-like
            The x-coordinates of the data points.
        yvlas : array-like
            The y-coordinates of the data points.
        bc_type : str, optional
            Boundary condition type. Options are 'clamped', 'second',
            'periodic', and 'not-a-knot'. Default is 'not-a-knot'.
        """
        self.xvals = np.asarray(xvals)
        self.yvals = np.asarray(yvals)
        self.h = np.diff(self.xvals)

        # only implemented using slopes
        self.slopes = None

        if bc_type not in ["clamped", "second", "periodic", "not-a-knot"]:
            raise ValueError(
                "Invalid boundary condition type. Choose from 'clamped', 'second', "
                "'periodic', or 'not-a-knot'."
            )

        if bc_type in ["clamped", "second", "periodic"]:
            raise NotImplementedError("Nah you alone in this one lil bro...💀🙏")

        if bc_type == "not-a-knot":
            self._not_a_knot()

    def _not_a_knot(self):
        N = len(self.xvals) - 1
        g = np.zeros(N - 1)

        mus = np.zeros(N - 1)
        for i in range(N - 1):
            mus[i] = self.h[i + 1] / (self.h[i] + self.h[i + 1])

        lambdas = np.zeros(N - 1)
        for i in range(N - 1):
            lambdas[i] = self.h[i] / (self.h[i] + self.h[i + 1])

        gamma_1 = self.h[0] / self.h[1]
        gamma_N = self.h[-1] / self.h[-2]

        g[0] = (
            lambdas[0] * (3 + 2 * gamma_1) * (self.yvals[2] - self.yvals[1]) / self.h[1]
            + mus[0] * (self.yvals[1] - self.yvals[0]) / self.h[0]
        )

        g[-1] = (
            mus[-1] * (3 + 2 * gamma_N) * (self.yvals[-2] - self.yvals[-3]) / self.h[-2]
            + lambdas[-1] * (self.yvals[-1] - self.yvals[-2]) / self.h[-1]
        )

        for i in range(2, N - 1):
            g[i - 1] = 3 * (
                lambdas[i - 1] * (self.yvals[i + 1] - self.yvals[i]) / self.h[i]
                + mus[i - 1] * (self.yvals[i] - self.yvals[i - 1]) / self.h[i - 1]
            )

        A = np.concatenate([mus[1:-1], [gamma_N]])
        B = np.concatenate([[1 + gamma_1], np.full(N - 3, 2), [1 + gamma_N]])
        C = np.concatenate([[gamma_1], lambdas[1:-1]])

        self.slopes = thomas(A, B, C, g)

        m_0 = (
            2 * (self.yvals[1] - self.yvals[0]) / self.h[0]
            - 2 * gamma_1**2 * (self.yvals[2] - self.yvals[1]) / self.h[1]
            - (1 - gamma_1**2) * self.slopes[0]
            + gamma_1**2 * self.slopes[1]
        )

        m_N = (
            -2
            * (
                gamma_N**2 * (self.yvals[-2] - self.yvals[-3]) / self.h[-2]
                - (self.yvals[-1] - self.yvals[-2]) / self.h[-1]
            )
            - gamma_N**2 * self.slopes[-2]
            + (1 - gamma_N**2) * self.slopes[-1]
        )

        self.slopes = np.concatenate([[m_0], self.slopes, [m_N]])

    def interpolate(self, x):
        """
        Interpolate the :math:`S(x)` using cubic spline interpolation.

        Parameters
        ----------
        x : float
            The x-coordinate at which to interpolate.

        Returns
        -------
        float
            The interpolated :math:`S(x)` value.
        """
        idx = np.searchsorted(self.xvals, x)
        i = max(idx - 1, 0)

        y = (
            self.yvals[i]
            + self.slopes[i] * (x - self.xvals[i])
            + (6 / self.h[i])
            * (
                (self.yvals[i + 1] - self.yvals[i]) / self.h[i]
                - (self.slopes[i + 1] + 2 * self.slopes[i]) / 3
            )
            * ((x - self.xvals[i]) ** 2)
            / 2
            + (12 / self.h[i] ** 2)
            * (
                (self.slopes[i + 1] + self.slopes[i]) / 2
                - (self.yvals[i + 1] - self.yvals[i]) / self.h[i]
            )
            * ((x - self.xvals[i]) ** 3 / 6)
        )

        return y

    def derivative(self, x, order=1):
        """
        Calculate the derivative of the cubic spline at a given point.

        Parameters
        ----------
        x : float
            The x-coordinate at which to calculate the derivative.

        Returns
        -------
        float
            The value of the derivative at :math:`x`.
        """
        idx = np.searchsorted(self.xvals, x)
        i = max(idx - 1, 0)

        if order == 0:
            return self.interpolate(x)
        elif order == 1:
            return (
                self.slopes[i]
                + (6 / self.h[i])
                * (
                    (self.yvals[i + 1] - self.yvals[i]) / self.h[i]
                    - (self.slopes[i + 1] + 2 * self.slopes[i]) / 3
                )
                * (x - self.xvals[i])
                + (6 / self.h[i] ** 2)
                * (
                    (self.slopes[i + 1] + self.slopes[i]) / 2
                    - (self.yvals[i + 1] - self.yvals[i]) / self.h[i]
                )
                * (x - self.xvals[i]) ** 2
            )
        elif order == 2:
            return (6 / self.h[i]) * (
                (self.yvals[i + 1] - self.yvals[i]) / self.h[i]
                - (self.slopes[i + 1] + 2 * self.slopes[i]) / 3
            ) + (12 / self.h[i] ** 2) * (
                (self.slopes[i + 1] + self.slopes[i]) / 2
                - (self.yvals[i + 1] - self.yvals[i]) / self.h[i]
            ) * (
                x - self.xvals[i]
            )

        return np.nan
