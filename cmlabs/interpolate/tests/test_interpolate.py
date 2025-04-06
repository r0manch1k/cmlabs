import pytest
import numpy as np
from cmlabs.interpolate import lagrange, lagrange_remainder, newton

__all__ = [
    "test_lagrange_degree",
    "test_lagrange_remainder_1",
    "test_lagrange_degree_2",
    "test_lagrange_remainder_2",
    "test_lagrange_compare_with_newton",
    "test_lagrange_from_docs_example",
    "test_lagrange_remainder_from_docs_example",
    "test_newton_from_docs_example",
]


def f(x):
    r"""Test function for Lagrange interpolation.

    .. math::

        f(x) = x - \log_{10}(x + 2)

    """
    return x - np.log10(x + 2)


X = np.linspace(0.5, 1.0, 10)

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

    r_exp_min = lagrange_remainder(nearest_x, f_der_2_min, df["x*"])
    print(f"R_min: {r_exp_min}")

    r_exp_max = lagrange_remainder(nearest_x, f_der_2_max, df["x*"])
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
    >>> |-1.4979036069111843e-07| <= |1.5933154223768398e-07| <= |-1.6889086295708034e-07|
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

    r_exp_min = lagrange_remainder(nearest_x, f_der_3_min, df["x*"])
    print(f"R_min: {r_exp_min}")

    r_exp_max = lagrange_remainder(nearest_x, f_der_3_max, df["x*"])
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


def test_lagrange_remainder_from_docs_example():
    r"""Lagrange remainder from docs example.

    See Also
    --------
    lagrange_remainder

    """
    print("\n")
    print("Test N: Lagrange Remainder From Docs Example")
    print("- f(x) = sin(x)")

    xvals = np.array([0, np.pi / 6, np.pi / 2])
    yvals = np.array([0, 1 / 2, 1])

    print("- X: ", xvals)
    print("- Y: ", yvals)

    M = 1.0
    print(f"- n = {len(xvals) - 1}")
    print(f"- M_3 = max |f'''(x)| = {M}")

    print(">>> np.sin(x) - lagrange(xvals, yvals, x)")

    x = np.pi / 4
    r_exp = np.sin(x) - lagrange(xvals, yvals, x)
    print(r_exp)

    print(f">>> lagrange_remainder(X, {M}, {x})")

    r_obs = lagrange_remainder(xvals, M, x)
    print(r_obs)

    assert abs(r_exp) <= r_obs, "Remainder is out of bounds"

    print(f">>> lagrange_remainder(X, {M})")

    r_obs = lagrange_remainder(xvals, M)
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
