import pytest
import numpy as np
import math
from cmlabs.interpolate import (
    lagrange,
    remainder,
    newton,
    finite_differences,
    forward_differences,
    backward_differences,
    newtonfd,
    newtonbd,
    gaussfd,
    gaussbd,
    stirling,
    bessel,
    interpolate,
)

__all__ = [
    "test_lagrange_degree",
    "test_lagrange_remainder_1",
    "test_lagrange_degree_2",
    "test_lagrange_remainder_2",
    "test_lagrange_compare_with_newton",
    "test_interpolate_remainder",
    "test_lagrange_derative",
    "test_lagrange_from_docs_example",
    "test_remainder_from_docs_example",
    "test_newton_from_docs_example",
    "test_finite_differences_from_docs_example",
    "test_forward_differences_from_docs_example",
    "test_backward_differences_from_docs_example",
    "test_newtonfd_from_docs_example",
    "test_newtonbd_from_docs_example",
    "test_gaussfd_from_docs_example",
    "test_gaussbd_from_docs_example",
]


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


def test_lagrange_degree():
    r"""Calculate Lagrange interpolation polynomial of degree 1 at :math:`x^*`.

    .. math::

        L_1(x^*) = f(x_i) \cdot \frac{x^* - x_{i+1}}{x_i - x_{i+1}} +
        f(x_{i+1}) \cdot \frac{x^* - x_i}{x_{i+1} - x_i}

    Results
    -------
    >>> # Test 1: Lagrange Interpolation Of Degree 1
    >>> # - f(x) = x - lg(x + 2)
    >>> # - X:  [0.5   0.556 0.611 ... 0.889 0.944 1.   ]
    >>> # - Y:  [0.102 0.148 0.194 ... 0.428 0.475 0.523]
    >>> # - x* =  0.77
    >>> f(0.77)
    >>> # 0.32752023093555144
    >>> # Nearest points: [0.722 0.778]
    >>> # Nearest f(x) values: [0.287 0.334]
    >>> lagrange([0.722 0.778], [0.287 0.334], 0.77)
    >>> # 0.327530850170338

    See Also
    --------
    lagrange

    """
    print("\n")
    print("Test 1: Lagrange Interpolation Of Degree 1")
    print("- f(x) = x - lg(x + 2)")
    print("- X: ", df["X"])
    print("- Y: ", df["Y"])
    print("- x* = ", df["x*"])

    print(f">>> f({df["x*"]})")

    f_exp = df["f"](df["x*"])
    print(f_exp)

    distances = np.abs(df["X"] - df["x*"])
    sorted_indices = np.argsort(distances)
    nearest_x = np.sort(df["X"][sorted_indices[:2]])
    print(f"Nearest points: {nearest_x}")

    nearest_y = df["f"](nearest_x)
    print(f"Nearest f(x) values: {nearest_y}")

    print(f">>> lagrange({nearest_x}, {nearest_y}, {df['x*']})")

    f_obs = lagrange(nearest_x, nearest_y, df["x*"])
    print(f_obs)

    assert isinstance(f_obs, float), "Result is not a float"


def test_lagrange_remainder_1():
    r"""Estimate the derivative and remainder degree 1.

    .. math::

        \begin{gather}
            R_1(x) = f(x) - L_1(x) \\
            \\
            f(x) = x - \log_{10}(x + 2) \\
            f''(x) = \frac{1}{(x + 2)^2 \cdot \ln(10)} \\
            \min{f''_{[x_i, x_{i+1}]}} = \min_{x \in [x_i, x_{i+1}]} |f''(x)| =
            \frac{1}{(x_{i+1} + 2)^2 \cdot \ln(10)} \\
            \max{f''_{[x_i, x_{i+1}]}} = \max_{x \in [x_i, x_{i+1}]} |f''(x)| =
            \frac{1}{(x_i + 2)^2 \cdot \ln(10)} \\
            \\
            \left|\frac{\min{f''(x)}}{2} \cdot
            \omega_2(x)\right| \leq |R_1(x)| \leq
            \left|\frac{\max{f''(x)}}{2} \cdot
            \omega_2(x)\right|, \quad x \in [x_i, x_{i+1}] \\
        \end{gather}

    Results
    -------
    >>> # Test 2: Estimating Remainder Degree 1 In Lagrange Interpolation Formula
    >>> # - X:  [0.5   0.556 0.611 ... 0.889 0.944 1.   ]
    >>> # - x* =  0.77
    >>> # Nearest points: [0.722 0.778]
    >>> # Nearest f(x) values: [0.287 0.334]
    >>> # f''(x) min: 0.056284564854661434
    >>> # f''(x) max: 0.05860533616686946
    >>> f(0.77) - lagrange([0.722 0.778], [0.287 0.334], 0.77)
    >>> # 1.0619234786568565e-05
    >>> # R_min: 1.0457811124230303e-05
    >>> # R_max: 1.0889016164338089e-05
    >>> |1.0457811124230303e-05| <= |1.0619234786568565e-05| <= |1.0889016164338089e-05|
    >>> # True
    >>> abs(1.0889016164338089e-05) <= 1e-4
    >>> # True

    See Also
    --------
    lagrange_remainder

    """
    print("\n")
    print("Test 2: Estimating Remainder Degree 1 In Lagrange Interpolation Formula")
    print("- X: ", df["X"])
    print("- x* = ", df["x*"])

    distances = np.abs(df["X"] - df["x*"])
    sorted_indices = np.argsort(distances)
    nearest_x = np.sort(df["X"][sorted_indices[:2]])
    print(f"Nearest points: {nearest_x}")

    nearest_y = df["f"](nearest_x)
    print(f"Nearest f(x) values: {nearest_y}")

    f_der_2_min = 1 / ((nearest_x[-1] + 2) ** 2 * np.log(10))
    f_der_2_max = 1 / ((nearest_x[0] + 2) ** 2 * np.log(10))
    print(f"f''(x) min: {f_der_2_min}")
    print(f"f''(x) max: {f_der_2_max}")

    print(f">>>f({df['x*']}) - lagrange({nearest_x}, {nearest_y}, {df['x*']})")

    r_exp = lagrange(nearest_x, nearest_y, df["x*"]) - df["f"](df["x*"])
    print(r_exp)

    r_exp_min = remainder(nearest_x, f_der_2_min, df["x*"])
    print(f"R_min: {r_exp_min}")

    r_exp_max = remainder(nearest_x, f_der_2_max, df["x*"])
    print(f"R_max: {r_exp_max}")

    print(f">>> |{r_exp_min}| <= |{r_exp}| <= |{r_exp_max}|")

    res = r_exp_min <= r_exp <= r_exp_max
    print(res)

    assert r_exp_min <= r_exp <= r_exp_max, "Remainder is out of bounds"

    print(f">>> abs({r_exp_max}) <= 1e-4")
    print(abs(r_exp_max) <= 1e-4)


def test_lagrange_degree_2():
    r"""Calculate Lagrange interpolation polynomial of degree 2 at :math:`x^*`.

    .. math::

        L_2(x^*) = f(x_{i-1}) \cdot
        \frac{(x^* - x_i)(x^* - x_{i+1})}{(x_{i-1} - x_i)(x_i - x_{i+1})} +
        f(x_i) \cdot
        \frac{(x^* - x_{i-1})(x^* - x_{i+1})}{(x_i - x_{i-1})(x_i - x_{i+1})} +
        f(x_{i+1}) \cdot
        \frac{(x^* - x_{i-1})(x^* - x_i)}{(x_{i+1} - x_{i-1})(x_{i+1} - x_i)}

    Results
    -------
    >>> # Test 3: Lagrange Interpolation Of Degree 2
    >>> # - f(x) = x - lg(x + 2)
    >>> # - X:  [0.5   0.556 0.611 ... 0.889 0.944 1.   ]
    >>> # - Y:  [0.102 0.148 0.194 ... 0.428 0.475 0.523]
    >>> # - x* =  0.77
    >>> f(0.77)
    >>> # 0.32752023093555144
    >>> # Nearest points: [0.722 0.778 0.833]
    >>> # Nearest f(x) values: [0.287 0.334 0.381]
    >>> lagrange([0.722 0.778 0.833], [0.287 0.334 0.381], 0.77)
    >>> # 0.3275203902670937

    See Also
    --------
    lagrange

    """
    print("\n")
    print("Test 3: Lagrange Interpolation Of Degree 2")
    print("- f(x) = x - lg(x + 2)")
    print("- X: ", df["X"])
    print("- Y: ", df["Y"])
    print("- x* = ", df["x*"])

    print(f">>> f({df['x*']})")

    f_exp = df["f"](df["x*"])
    print(f_exp)

    distances = np.abs(df["X"] - df["x*"])
    sorted_indices = np.argsort(distances)
    nearest_x = np.sort(df["X"][sorted_indices[:3]])
    print(f"Nearest points: {nearest_x}")

    nearest_y = df["f"](nearest_x)
    print(f"Nearest f(x) values: {nearest_y}")

    print(f">>> lagrange({nearest_x}, {nearest_y}, {df['x*']})")

    f_obs = lagrange(nearest_x, nearest_y, df["x*"])
    print(f_obs)

    assert isinstance(f_obs, float), "Result is not a float"


def test_lagrange_remainder_2():
    r"""Estimate the derivative and remainder degree 2.

    .. math::

        \begin{gather}
            R_2(x) = f(x) - L_2(x) \\
            \\
            f(x) = x - \log_{10}(x + 2) \\
            f^{(3)}(x) = \frac{-2}{(x + 2)^3 \cdot \ln(10)} \\
            \min{f^{(3)}_{[x_{i-1}, x_{i+1}]}} = \min_{x \in [x_{i-1}, x_{i+1}]}
            |f^{(3)}(x)| = \frac{-2}{(x_{i+1} + 2)^3 \cdot \ln(10)} \\
            \max{f^{(3)}_{[x_{i-1}, x_{i+1}]}} = \max_{x \in [x_{i-1}, x_{i+1}]}
            |f^{(3)}(x)| = \frac{-2}{(x_{i-1} + 2)^3 \cdot \ln(10)} \\
            \\
            \left|\frac{\min{f^{(3)}(x)}}{6} \cdot
            \omega_3(x)\right| \leq |R_2(x)| \leq
            \left|\frac{\max{f^{(3)}(x)}}{6} \cdot
            \omega_3(x)\right|, \quad x \in [x_{i-1}, x_{i+1}]
        \end{gather}

    Results
    -------
    >>> # Test 4: Estimating Remainder Degree 2 In Lagrange Interpolation Formula
    >>> # - X:  [0.5   0.556 0.611 ... 0.889 0.944 1.   ]
    >>> # - x* =  0.77
    >>> # Nearest points: [0.722 0.778 0.833]
    >>> # Nearest f(x) values: [0.287 0.334 0.381]
    >>> # f'''(x) min: -0.03818750583802256
    >>> # f'''(x) max: -0.04305698167361838
    >>> f(0.77) - lagrange([0.722 0.778 0.833], [0.287 0.334 0.381], 0.77)
    >>> # 1.5933154223768398e-07
    >>> # R_min: -1.4979036069111843e-07
    >>> # R_max: -1.6889086295708034e-07
    >>> |-1.4979036069111843e-07| <= |1.5933154223768398e-07| <= |-1.688908629...|
    >>> # True
    >>> abs(-1.6889086295708034e-07) <= 1e-5
    >>> # True

    See Also
    --------
    lagrange_remainder

    """
    print("\n")
    print("Test 4: Estimating Remainder Degree 2 In Lagrange Interpolation Formula")
    print("- X: ", df["X"])
    print("- x* = ", df["x*"])

    distances = np.abs(df["X"] - df["x*"])
    sorted_indices = np.argsort(distances)
    nearest_x = np.sort(df["X"][sorted_indices[:3]])
    print(f"Nearest points: {nearest_x}")

    nearest_y = df["f"](nearest_x)
    print(f"Nearest f(x) values: {nearest_y}")

    f_der_3_min = -2 / ((nearest_x[-1] + 2) ** 3 * np.log(10))
    f_der_3_max = -2 / ((nearest_x[0] + 2) ** 3 * np.log(10))
    print(f"f'''(x) min: {f_der_3_min}")
    print(f"f'''(x) max: {f_der_3_max}")

    print(f">>> f({df['x*']}) - lagrange({nearest_x}, {nearest_y}, {df['x*']})")

    r_exp = lagrange(nearest_x, nearest_y, df["x*"]) - df["f"](df["x*"])
    print(r_exp)

    r_exp_min = remainder(nearest_x, f_der_3_min, df["x*"])
    print(f"R_min: {r_exp_min}")

    r_exp_max = remainder(nearest_x, f_der_3_max, df["x*"])
    print(f"R_max: {r_exp_max}")

    print(f">>> |{r_exp_min}| <= |{r_exp}| <= |{r_exp_max}|")

    res = abs(r_exp_min) <= abs(r_exp) <= abs(r_exp_max)
    print(res)

    assert abs(r_exp_min) <= abs(r_exp) <= abs(r_exp_max), "Remainder is out of bounds"

    print(f">>> abs({r_exp_max}) <= 1e-5")
    print(abs(r_exp_max) <= 1e-5)


@pytest.mark.parametrize("degree", [1, 2])
def test_lagrange_compare_with_newton(degree):
    r"""Compare Lagrange and Newton interpolation.

    .. math::

        \begin{gather}
            L_n(x) = \sum_{i=0}^{n} l_i(x) f(x_i)\\
            \\
            L_n(x) = \sum_{i=0}^{n} f(x_0, x_1, \ldots, x_i) \omega_i(x)
        \end{gather}

    Results
    -------
    >>> # Test 5: Compare Lagrange And Newton Interpolation Degree 1
    >>> # - X:  [0.5   0.556 0.611 ... 0.889 0.944 1.   ]
    >>> # - Y:  [0.102 0.148 0.194 ... 0.428 0.475 0.523]
    >>> # - x* =  0.77
    >>> # Nearest points: [0.722 0.778]
    >>> # Nearest f(x) values: [0.287 0.334]
    >>> lagrange([0.722 0.778], [0.287 0.334], 0.77)
    >>> # 0.327530850170338
    >>> newton([0.722 0.778], 0.77, yvals=[0.287 0.334])
    >>> # 0.327530850170338

    >>> # Test 6: Compare Lagrange And Newton Interpolation Degree 2
    >>> # - X:  [0.5   0.556 0.611 ... 0.889 0.944 1.   ]
    >>> # - Y:  [0.102 0.148 0.194 ... 0.428 0.475 0.523]
    >>> # - x* =  0.77
    >>> # Nearest points: [0.722 0.778 0.833]
    >>> # Nearest f(x) values: [0.287 0.334 0.381]
    >>> lagrange([0.722 0.778 0.833], [0.287 0.334 0.381], 0.77)
    >>> # 0.3275203902670937
    >>> newton([0.722 0.778 0.833], 0.77, yvals=[0.287 0.334 0.381])
    >>> # 0.3275203902670936

    See Also
    --------
    lagrange
    newton

    """
    print("\n")
    print(
        f"Test {4 + degree}: Compare Lagrange And Newton Interpolation Degree {degree}"
    )
    print("- X: ", df["X"])
    print("- Y: ", df["Y"])
    print("- x* = ", df["x*"])

    distances = np.abs(df["X"] - df["x*"])
    sorted_indices = np.argsort(distances)
    nearest_x = np.sort(df["X"][sorted_indices[: (degree + 1)]])
    print(f"Nearest points: {nearest_x}")

    nearest_y = df["f"](nearest_x)
    print(f"Nearest f(x) values: {nearest_y}")

    print(f">>> lagrange({nearest_x}, {nearest_y}, {df['x*']})")

    f_obs_l = lagrange(nearest_x, nearest_y, df["x*"])
    print(f_obs_l)

    print(f">>> newton({nearest_x}, {df['x*']}, yvals={nearest_y})")

    f_obs_n = newton(nearest_x, df["x*"], yvals=nearest_y)
    print(f_obs_n)

    assert np.isclose(
        f_obs_l, f_obs_n, atol=1e-6
    ), "Lagrange and Newton results are not close"


def test_interpolate_remainder():
    r"""Estimate the remainder of interpolation.

    .. math::

        \begin{aligned}
            R_n(x) &= f(x) - L_n(x) \\
            &= \frac{f^{(n+1)}(\xi)}{(n+1)!} \cdot \omega_{n+1}(x) \\
        \end{aligned}

    for :math:`n = 9` have

    .. math::

        \begin{aligned}
            (x + \log_{10}(x + 2))^{(10)} = \frac{362880}{\ln(10) \cdot (x + 2)^{10}}
        \end{aligned}

    Results
    -------
    >>> # Test 7: Estimate Remainder Of Interpolation
    >>> # - X:  [0.5   0.556 0.611 ... 0.889 0.944 1.   ]
    >>> # - Y:  [0.102 0.148 0.194 ... 0.428 0.475 0.523]
    >>> # - x** =  0.52
    >>> # - x*** =  0.97
    >>> # - x**** =  0.73
    >>> # M_min = 2.6689153346043457
    >>> # M_max = 16.52522028557161
    >>> interpolate([0.5   0.556 0.611 ... 0.889 0.944 1.   ], 0.52, yvals=[...])
    >>> # Using Newton's forward interpolation formula for 0.52
    >>> # 0.1185994592184786
    >>> # R_min = 8.579275443070178e-15
    >>> # R_max = 5.312061223865951e-14
    >>> 8.579275443070178e-15 <= 2.2662427490161008e-14 <= 5.312061223865951e-14
    >>> True
    >>> interpolate([0.5   0.556 0.611 ... 0.889 0.944 1.   ], 0.97, yvals=...)
    >>> # Using Newton's backward interpolation formula for 0.97
    >>> # 0.4972435506828018
    >>> # R_min = 6.312938757056343e-15
    >>> # R_max = 3.9088052834446335e-14
    >>> 6.312938757056343e-15 <= 1.4155343563970746e-14 <= 3.9088052834446335e-14
    >>> True
    >>> interpolate([0.5   0.556 0.611 ... 0.889 0.944 1.   ], 0.73, yvals=...)
    >>> # Using Gauss's forward interpolation formula for 0.73
    >>> # 0.29383735295924407
    >>> # R_min = 7.849181383895046e-17
    >>> # R_max = 4.860006226068698e-16
    >>> 7.849181383895046e-17 <= 1.1102230246251565e-16 <= 4.860006226068698e-16
    >>> True

    See Also
    --------
    lagrange
    newton
    newtonfd
    newtonbd
    gaussfd
    gaussbd
    stirling
    bessel
    interpolate
    """
    print("\n")
    print("Test 7: Estimate Remainder Of Interpolation")
    print("- X: ", df["X"])
    print("- Y: ", df["Y"])
    print("- x** = ", df["x**"])
    print("- x*** = ", df["x***"])
    print("- x**** = ", df["x****"])

    M_min = 362880 / ((df["X"][-1] + 2) ** 10 * np.log(10))
    M_max = 362880 / ((df["X"][0] + 2) ** 10 * np.log(10))
    print(f"M_min = {M_min}")
    print(f"M_max = {M_max}")

    print(f">>> interpolate({df['X']}, {df['x**']}, yvals={df['Y']})")

    l_1 = interpolate(df["X"], df["x**"], yvals=df["Y"])
    print(l_1)

    r_min = abs(remainder(df["X"], M_min, df["x**"]))
    r_max = abs(remainder(df["X"], M_max, df["x**"]))
    print(f"R_min = {r_min}")
    print(f"R_max = {r_max}")

    r_exp = abs(df["f"](df["x**"]) - l_1)
    print(f">>> {r_min} <= {r_exp} <= {r_max}")
    print(r_min <= r_exp <= r_max)

    assert r_min <= r_exp <= r_max, "Remainder is out of bounds"

    print(f">>> interpolate({df['X']}, {df['x***']}, yvals={df['Y']})")

    l_2 = interpolate(df["X"], df["x***"], yvals=df["Y"])
    print(l_2)

    r_min = abs(remainder(df["X"], M_min, df["x***"]))
    r_max = abs(remainder(df["X"], M_max, df["x***"]))
    print(f"R_min = {r_min}")
    print(f"R_max = {r_max}")

    r_exp = abs(df["f"](df["x***"]) - l_2)
    print(f">>> {r_min} <= {r_exp} <= {r_max}")
    print(r_min <= r_exp <= r_max)

    assert r_min <= r_exp <= r_max, "Remainder is out of bounds"

    print(f">>> interpolate({df['X']}, {df['x****']}, yvals={df['Y']})")

    l_3 = interpolate(df["X"], df["x****"], yvals=df["Y"])
    print(l_3)

    r_min = abs(remainder(df["X"], M_min, df["x****"]))
    r_max = abs(remainder(df["X"], M_max, df["x****"]))
    print(f"R_min = {r_min}")
    print(f"R_max = {r_max}")

    r_exp = abs(df["f"](df["x****"]) - l_3)
    print(f">>> {r_min} <= {r_exp} <= {r_max}")
    print(r_min <= r_exp <= r_max)

    assert r_min <= r_exp <= r_max, "Remainder is out of bounds"


def test_lagrange_derative():
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
    lagrange
    remainder

    Notes
    -----
    -

    Examples
    -------
    >>> # Test 8: Lagrange Derivative
    >>> # - X:  [0.5   0.556 0.611 ... 0.889 0.944 1.   ]
    >>> # - x_0 =  0.5
    >>> # - h = 0.05555555555555558
    >>> # - Y:  [0.102 0.148 0.194 ... 0.428 0.475 0.523]
    >>> # L_4_2(x_0) = 0.06947426144114938
    >>> # R_4_2_min = 6.1744127485294504e-06
    >>> # R_4_2_max = 1.5386508722671422e-05
    >>> # f''(x_0) = 0.06948711710452028
    >>> # R_4_2 = 1.2855663370905934e-05
    >>> 6.1744127485294504e-06 <= 1.2855663370905934e-05 <= 1.5386508722671422e-05
    >>> # True

    """
    print("\n")
    print("Test 8: Lagrange Derivative")

    print("- X: ", df["X"])
    print("- Y: ", df["Y"])
    print("- x_0 = ", df["X"][0])

    h = df["X"][1] - df["X"][0]
    print(f"- h = {h}")

    l_4_2 = (1 / h**2) * (
        (35 / 12) * df["Y"][0]
        - (26 / 3) * df["Y"][1]
        + (19 / 2) * df["Y"][2]
        - (14 / 3) * df["Y"][3]
        + (11 / 12) * df["Y"][4]
    )
    print(f"L_4_2(x_0) = {l_4_2}")

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

    r_4_2 = abs(l_4_2 - f_2)
    print(f"R_4_2 = {r_4_2}")

    print(f">>> {r_4_2_min} <= {r_4_2} <= {r_4_2_max}")
    print(r_4_2_min <= r_4_2 <= r_4_2_max)

    assert r_4_2_min <= r_4_2 <= r_4_2_max, "Remainder is out of bounds"


def test_lagrange_from_docs_example():
    r"""Lagrange interpolation from docs example.

    See Also
    --------
    lagrange

    """
    print("\n")
    print("Test N: Lagrange Interpolation From Docs Example")

    xvals = np.array([0, 2, 3, 5])
    yvals = np.array([1, 3, 2, 5])
    print("- X: ", xvals)
    print("- Y: ", yvals)

    x = 1.5
    print(f">>> lagrange({xvals}, {yvals}, {x})")

    f_obs = lagrange(xvals, yvals, x)
    print(f_obs)

    assert isinstance(f_obs, float), "Result is not a float"


def test_remainder_from_docs_example():
    r"""Remainder from docs example.

    See Also
    --------
    remainder

    """
    print("\n")
    print("Test N: Remainder From Docs Example")
    print("- f(x) = sin(x)")

    xvals = np.array([0, np.pi / 6, np.pi / 2])
    yvals = np.array([0, 1 / 2, 1])

    print("- X: ", xvals)
    print("- Y: ", yvals)

    M = 1.0
    print(f"- n = {len(xvals) - 1}")
    print(f"- M_3 = max |f'''(x)| = {M}")

    print(">>> np.sin(x) - lagrange(xvals, yvals, x)")

    x = np.pi / 8
    r_exp = abs(np.sin(x) - lagrange(xvals, yvals, x))
    print(r_exp)

    print(f">>> lagrange_remainder(X, {M}, {x})")

    r_obs = remainder(xvals, M, x)
    print(r_obs)

    assert abs(r_exp) <= r_obs, "Remainder is out of bounds"

    print(f">>> lagrange_remainder(X, {M})")

    r_obs = remainder(xvals, M)
    print(r_obs)

    assert abs(r_exp) <= r_obs, "Remainder is out of bounds"


def test_newton_from_docs_example():
    r"""Newton interpolation from docs example.

    See Also
    --------
    newton

    """
    print("\n")
    print("Test N: Newton Interpolation From Docs Example")

    xvals = np.array([0, 2, 3, 5])
    yvals = np.array([1, 3, 2, 5])
    print("- X: ", xvals)
    print("- Y: ", yvals)

    x = 1.5
    print(f">>> newton({xvals}, {x}, yvals={yvals})")

    f_obs = newton(xvals, x, yvals=yvals)
    print(f_obs)

    assert isinstance(f_obs, float), "Result is not a float"


def test_finite_differences_from_docs_example():
    r"""Finite differences from docs example.

    See Also
    --------
    finite_differences

    """
    print("\n")
    print("Test N: Finite Differences From Docs Example")

    yvals = np.array([1, 3, 2, 5])
    print("- Y: ", yvals)

    print(f">>> finite_differences({yvals})")

    fd = finite_differences(yvals)
    print(fd)

    assert isinstance(fd, list), "Result is not a list"


def test_forward_differences_from_docs_example():
    r"""Forward differences from docs example.

    See Also
    --------
    forward_differences

    """
    print("\n")
    print("Test N: Forward Differences From Docs Example")

    yvals = np.array([1, 3, 2, 5])
    print("- Y: ", yvals)

    print(f">>> forward_differences({yvals})")

    fd = forward_differences(yvals)
    print(fd)

    assert isinstance(fd, np.ndarray), "Result is not a numpy array"


def test_backward_differences_from_docs_example():
    r"""Backward differences from docs example.

    See Also
    --------
    backward_differences

    """
    print("\n")
    print("Test N: Backward Differences From Docs Example")

    yvals = np.array([1, 3, 2, 5])
    print("- Y: ", yvals)

    print(f">>> backward_differences({yvals})")

    bd = backward_differences(yvals)
    print(bd)

    assert isinstance(bd, np.ndarray), "Result is not a numpy array"


def test_newtonfd_from_docs_example():
    r"""Newton's forward interpolation formula from docs example.

    See Also
    --------
    newtonfd

    """
    print("\n")
    print("Test N: Newton Forward Differences From Docs Example")

    xvals = np.array([0, 1, 2, 3])
    yvals = np.array([1, 3, 2, 5])
    print("- X: ", xvals)
    print("- Y: ", yvals)

    x = 0.5
    print(f">>> x = {x}")

    print(f">>> newtonfd({xvals}, {x}, yvals={yvals})")

    f_obs = newtonfd(xvals, x, yvals=yvals)
    print(f_obs)

    assert isinstance(f_obs, float), "Result is not a float"


def test_newtonbd_from_docs_example():
    r"""Newton's backward interpolation formula from docs example.

    See Also
    --------
    newtonbd

    """
    print("\n")
    print("Test N: Newton Backward Differences From Docs Example")

    xvals = np.array([0, 1, 2, 3])
    yvals = np.array([1, 3, 2, 5])
    print("- X: ", xvals)
    print("- Y: ", yvals)

    x = 2.5
    print(f">>> x = {x}")

    print(f">>> newtonbd({xvals}, {x}, yvals={yvals})")

    f_obs = newtonbd(xvals, x, yvals=yvals)
    print(f_obs)

    assert isinstance(f_obs, float), "Result is not a float"


def test_gaussfd_from_docs_example():
    r"""Gauss’s forward interpolation formula from docs example.

    See Also
    --------
    gaussfd

    """
    print("\n")
    print("Test N: Gauss Forward Differences From Docs Example")

    xvals = np.array([0, 1, 2, 3])
    yvals = np.array([1, 3, 2, 5])
    print("- X: ", xvals)
    print("- Y: ", yvals)

    x = 1.25
    print(f">>> x = {x}")

    print(f">>> gaussfd({xvals}, {x}, yvals={yvals})")

    f_obs = gaussfd(xvals, x, yvals=yvals)
    print(f_obs)

    assert isinstance(f_obs, float), "Result is not a float"


def test_gaussbd_from_docs_example():
    r"""Gauss’s backward interpolation formula from docs example.

    See Also
    --------
    gaussbd

    """
    print("\n")
    print("Test N: Gauss Backward Differences From Docs Example")

    xvals = np.array([0, 1, 2, 3])
    yvals = np.array([1, 3, 2, 5])
    print("- X: ", xvals)
    print("- Y: ", yvals)

    x = 0.75
    print(f">>> x = {x}")

    print(f">>> gaussbd({xvals}, {x}, yvals={yvals})")

    f_obs = gaussbd(xvals, x, yvals=yvals)
    print(f_obs)

    assert isinstance(f_obs, float), "Result is not a float"


def test_stirling_from_docs_example():
    r"""Stirling's interpolation formula from docs example.

    See Also
    --------
    stirling

    """
    print("\n")
    print("Test N: Stirling Interpolation From Docs Example")

    xvals = np.array([0, 1, 2, 3, 4])
    yvals = np.array([1, 3, 2, 5, 3])
    print("- X: ", xvals)
    print("- Y: ", yvals)

    x = 2.15
    print(f">>> x = {x}")

    print(f">>> stirling({xvals}, {x}, yvals={yvals})")

    f_obs = stirling(xvals, x, yvals=yvals)
    print(f_obs)

    assert isinstance(f_obs, float), "Result is not a float"


def test_bessel_from_docs_example():
    r"""Bessel's interpolation formula from docs example.

    See Also
    --------
    bessel

    """
    print("\n")
    print("Test N: Bessel Interpolation From Docs Example")

    xvals = np.array([0, 1, 2, 3])
    yvals = np.array([1, 3, 2, 5])
    print("- X: ", xvals)
    print("- Y: ", yvals)

    x = 1.15
    print(f">>> x = {x}")

    print(f">>> bessel({xvals}, {x}, yvals={yvals})")

    f_obs = bessel(xvals, x, yvals=yvals)
    print(f_obs)

    assert isinstance(f_obs, float), "Result is not a float"


def test_interpolate_from_docs_example():
    r"""Interpolation from docs example.

    See Also
    --------
    interpolate

    """
    print("\n")
    print("Test N: Interpolation From Docs Example")

    xvals = np.array([0, 1, 2, 3])
    yvals = np.array([1, 3, 2, 5])
    print("- X: ", xvals)
    print("- Y: ", yvals)

    x = 1.15
    print(f">>> x = {x}")

    print(f">>> interpolate({xvals}, {x}, yvals={yvals})")

    f_obs = interpolate(xvals, x, yvals=yvals)
    print(f_obs)

    assert isinstance(f_obs, float), "Result is not a float"
