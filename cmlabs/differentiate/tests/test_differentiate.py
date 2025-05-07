import math
import numpy as np
from cmlabs.differentiate import lagrange_derivative

__all__ = ["test_lagrange_derivative", "test_lagrange_derivative_from_docs_example"]


def f(x):
    r"""Test function for Lagrange interpolation.

    .. math::

        f(x) = x - \log_{10}(x + 2)

    """
    return x - np.log10(x + 2)


X = np.linspace(0.5, 1, 10)

df = {
    "f": f,
    "X": X,
    "Y": f(X),
    "x*": 0.77,
    "x**": 0.52,
    "x***": 0.97,
    "x****": 0.73,
}

np.set_printoptions(threshold=5, precision=3, suppress=True)


def test_lagrange_derivative():
    r"""Lagrange derivative.

    .. math::

        \begin{aligned}
            L_4(x) = \sum_{i=0}^{4} l_i(x) f(x_i) \\
            l_i(x) = \prod_{j=0, j \neq i}^{4} \frac{x - x_j}{x_i - x_j}
        \end{aligned}

    :math:`k=2, \quad m=0, \quad n=4`

    .. math::

        \begin{gather}
            l_0''(x_0) = \frac{35}{12h^2}, \quad l_1''(x_0) = -\frac{26}{3h^2},
            \quad l_2''(x_0) = \frac{19}{2h^2} \\
            l_3''(x_0) = -\frac{14}{3h^2}, \quad l_4''(x_0) = \frac{11}{12h^2} \\
            L_4''(x_0) = \frac{1}{h^2} \left( \frac{35}{12} f(x_0) - \frac{26}{3} f(x_1)
            + \frac{19}{2} f(x_2) - \frac{14}{3} f(x_3) + \frac{11}{12} f(x_4) \right)\\
            R_{4,2}''(x_0) = \frac{f^{(5)}(\xi_1)}{5!}\omega_{5}''(x_0)
            + 2 \frac{f^{(6)}(\xi_2)}{6!}\omega_{5}'(x_0) \\
            R_{4,2}''(x_0) = \frac{f^{(5)}}{5!} \cdot (-100h^3)
            + \frac{f^{(6)}}{6!} \cdot 48h^4 \\
            \left(x - \lg(x+2)\right)'' = \frac{1}{(x + 2)^2 \cdot \ln(10)} \\
            \left(x - \lg(x+2)\right)^{(5)} = -\frac{24}{(x + 2)^5 \cdot \ln(10)} \\
            \left(x - \lg(x+2)\right)^{(6)} = \frac{120}{(x + 2)^6 \cdot \ln(10)} \\
            min(R_{4,2}) = \left(-\frac{24}{(x_n + 2)^5 \cdot \ln(10)}\right)
            \cdot \frac{1}{5!} \cdot (-100h^3) + \\
            + \left(\frac{120}{(x_n + 2)^6 \cdot
            \ln(10)}\right) \cdot \frac{1}{6!} \cdot 48h^4 \\
            max(R_{4,2}) = \left(-\frac{24}{(x_0 + 2)^5 \cdot \ln(10)}\right)
            \cdot \frac{1}{5!} \cdot (-100h^3) + \\
            + \left(\frac{120}{(x_0 + 2)^6 \cdot
            \ln(10)}\right) \cdot \frac{1}{6!} \cdot 48h^4 \\
            |min(R_{4,2})| \leq |R_{4,2}''(x)| \leq |max(R_{4,2})|
        \end{gather}

    See Also
    --------
    cmlabs.interpolate.lagrange
    cmlabs.interpolate.remainder
    lagrange_derivative

    Notes
    -----
    -

    Examples
    --------
    >>> # Test 1: Lagrange Derivative
    >>> # - X:  [0.5   0.556 0.611 ... 0.889 0.944 1.   ]
    >>> # - x_0 =  0.5
    >>> # - h = 0.05555555555555558
    >>> # - Y:  [0.102 0.148 0.194 ... 0.428 0.475 0.523]
    >>> # L_4_2(x_0) = 0.06947426144114938
    >>> # OR WE CAN USE CUSTOM lagrange_derivative FUNCTION
    >>> lagrange_derivative([0.5 ... 0.722], [0.102 ... 0.287], 0.5, k=2)
    >>> # L_4_2(x_0) = 0.06947426144084545
    >>> # R_4_2_min = 6.1744127485294504e-06
    >>> # R_4_2_max = 1.5386508722671422e-05
    >>> # f''(x_0) = 0.06948711710452028
    >>> # R_4_2 = 1.2855663370905934e-05
    >>> 6.1744127485294504e-06 <= 1.2855663370905934e-05 <= 1.5386508722671422e-05
    >>> # True

    """
    print("\n")
    print("Test 1: Lagrange Derivative")

    print("- X: ", df["X"])
    print("- Y: ", df["Y"])
    print("- x_0 = ", df["X"][0])

    h = df["X"][1] - df["X"][0]
    print(f"- h = {h}")

    L_4_2 = (1 / h**2) * (
        (35 / 12) * df["Y"][0]
        - (26 / 3) * df["Y"][1]
        + (19 / 2) * df["Y"][2]
        - (14 / 3) * df["Y"][3]
        + (11 / 12) * df["Y"][4]
    )
    print(f"L_4_2(x_0) = {L_4_2}")

    print("OR WE CAN USE CUSTOM lagrange_derivative FUNCTION")
    print(f">>> lagrange_derivative({df['X'][:5]}, {df['Y'][:5]}, {df['X'][0]}, k=2)")

    L_4_2 = lagrange_derivative(df["X"][:5], df["Y"][:5], df["X"][0], k=2)
    print(f"L_4_2(x_0) = {L_4_2}")

    r_4_2_min = abs(
        (-24 / ((df["X"][-1] + 2) ** 5 * np.log(10)))
        * (1 / math.factorial(5))
        * (-100 * h**3)
        + (120 / ((df["X"][-1] + 2) ** 6 * np.log(10)))
        * (1 / math.factorial(6))
        * (48 * h**4)
    )
    print(f"R_4_2_min = {r_4_2_min}")

    r_4_2_max = abs(
        (-24 / ((df["X"][0] + 2) ** 5 * np.log(10)))
        * (1 / math.factorial(5))
        * (-100 * h**3)
        + (120 / ((df["X"][0] + 2) ** 6 * np.log(10)))
        * (1 / math.factorial(6))
        * (48 * h**4)
    )
    print(f"R_4_2_max = {r_4_2_max}")

    f_2 = 1 / ((df["X"][0] + 2) ** 2 * np.log(10))
    print(f"f''(x_0) = {f_2}")

    r_4_2 = abs(L_4_2 - f_2)
    print(f"R_4_2 = {r_4_2}")

    print(f">>> {r_4_2_min} <= {r_4_2} <= {r_4_2_max}")
    print(r_4_2_min <= r_4_2 <= r_4_2_max)

    assert r_4_2_min <= r_4_2 <= r_4_2_max, "Remainder is out of bounds"


def test_lagrange_derivative_from_docs_example():
    r"""Lagrange derivative from docs example.

    Results
    --------
    >>> # Test N: Lagrange Derivative From Docs Example
    >>> # - X:  [0 1 2 3 4]
    >>> # - Y:  [ 0  1  4  9 16]
    >>> # - x_0 =  2.5
    >>> # - k =  2
    >>> lagrange_derivative([0 1 2 3 4], [ 0  1  4  9 16], 2.5, k=2)
    >>> # 2.0

    See Also
    --------
    lagrange_derivative
    """
    print("\n")
    print("Test N: Lagrange Derivative From Docs Example")

    xvals = np.array([0, 1, 2, 3, 4])
    yvals = np.array([0, 1, 4, 9, 16])

    print("- X: ", xvals)
    print("- Y: ", yvals)

    x = 2.5
    print("- x_0 = ", x)

    k = 2
    print("- k = ", k)

    print(f">>> lagrange_derivative({xvals}, {yvals}, {x}, k={k})")

    L_4_2 = lagrange_derivative(xvals, yvals, x, k)
    print(L_4_2)
