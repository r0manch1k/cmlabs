import pytest
import numpy as np
from cmlabs.interpolate import lagrange, lagrange_remainder


def f(x):
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


TOL = 1e-4

np.set_printoptions(threshold=5, precision=3, suppress=True)


def test_lagrange_degree_1():
    r"""Calculate Lagrange interpolation polynomial of degree 1 at :math:`x^*`.

    .. math::

        L_1(x^*) = f(x_i) \cdot \frac{x^* - x_{i+1}}{x_i - x_{i+1}} +
        f(x_{i+1}) \cdot \frac{x^* - x_i}{x_{i+1} - x_i}
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

    assert isinstance(f_obs, float)


# def test_lagrange_remainder():
#     r"""Estimate the derative and remainder in Lagrange interpolation formula.

#     .. math::

#         \begin{gather}
#             R_n(x) = f(x) - L_n(x) \\
#             |f^{(n+1)}(\xi)| \leq \left|\frac{(n+1)!
#             \cdot R_n(x)}{\omega_{n+1}}\right|
#         \end{gather}
#     """
#     print("\n")
#     print("Test 2: Estimating Remainder In Lagrange Interpolation Formula\n")
#     print("- X: ", df["X"].values)
#     print("- x* = ", df["x*"])

#     distances = np.abs(df["X"] - df["x*"])
#     sorted_indices = np.argsort(distances)
#     nearest_x = np.sort(df["X"][sorted_indices[:2]])
#     print(f"Nearest points: {nearest_x}")

#     nearest_y = df["f"](nearest_x)
#     print(f"Nearest f(x) values: {nearest_y}")

#     print(""
#     res = lagrange_remainder(df["X"].values, df["Y"].values, df["x*"].values)

#     print("Result: ", res)

#     expected = lagrange_remainder(
#         df["X"].values, df["Y"].values, df["x*"].values, method=method
#     )
#     print("Expected: ", expected)

#     assert np.isclose(res, expected, atol=TOL)


def test_lagrange_remainder_from_docs_example():
    print("\n")
    print("Test 3: Lagrange Remainder From Docs Example")
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

    print(f">>> lagrange_remainder(X, {M})")

    r_obs = lagrange_remainder(xvals, M)
    print(r_obs)
    # assert np.isclose(r_obs, r_exp, atol=TOL)
