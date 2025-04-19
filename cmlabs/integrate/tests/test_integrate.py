import numpy as np
from cmlabs.integrate import rectangle, midpoint, trapezoid, simpsonq, simpsonc, weddles

__all__ = [
    "test_rectangle_error",
    "test_rectangle_from_docs_example",
    "test_midpoint_from_docs_example",
    "test_trapezoid_from_docs_example",
    "test_simpsonq_from_docs_example",
    "test_simpsonc_from_docs_example",
    "test_weddles_from_docs_example",
]


def f(x):
    r"""Test function for Lagrange interpolation.

    .. math::

        f(x) = x - \log_{10}(x + 2)

    """
    return x - np.log10(x + 2)


df = {
    "f": f,
    "a": 0.5,
    "b": 1,
    "eps": 0.01,
    "x*": 0.77,
    "x**": 0.52,
    "x***": 0.97,
    "x****": 0.73,
}

np.set_printoptions(threshold=5, precision=3, suppress=True)


def test_rectangle_error():
    r"""Reach accuracy :math:`\epsilon` error for rectangle rule.

    .. math::

        \begin{gather}
            |I_n - I_{2n}| \leq \epsilon, \\
            |I_{2n} - I_{4n}| \leq \epsilon, \\
            \ldots \\
            |I_{2^k n} - I_{2^{k+1} n}| \leq \epsilon, \\
        \end{gather}

    where :math:`I_n` is the integral of the function :math:`f(x)` over the interval
    :math:`[a, b]` using the rectangle rule with :math:`n`
    subintervals, and :math:`\epsilon` is the desired accuracy.

    Notes
    -----

    .. math::

        \begin{gather}
            \int_a^b f(x) \, dx = \int_{0.5}^{1} x - \log_{10}(x + 2) \, dx \\
            = \frac{1\ln{10} + 20\ln{5} - 24\ln{3} - 20\ln{2} + 4}{8\ln{10}} \\
            \approx 0.1556335
        \end{gather}

    Results
    -------
    >>> # Test 1: Rectangle Rule Error
    >>> # - f(x) = x - lg(x + 2)
    >>> # - X_n:  [0.5   0.625 0.75  0.875 1.   ]
    >>> # - Y_1:  [0.102 0.206 0.311 0.416 0.523]
    >>> # - X_2n:  [0.5   0.556 0.611 ... 0.889 0.944 1.   ]
    >>> # - Y_2n:  [0.102 0.148 0.194 ... 0.428 0.475 0.523]
    >>> rectangle(X_n, Y_n, method='left')
    >>> # - I_n:  0.12937001759125935
    >>> rectangle(X_2n, Y_2n, method='left')
    >>> # - I_2n:  0.14395153508130418
    >>> # |I_n - I_2n|:  0.014581517490044826
    >>> # 0.014581517490044826 <= 0.01
    >>> # False
    >>> # - X_4n:  [0.5   0.526 0.553 ... 0.947 0.974 1.   ]
    >>> # - Y_4n:  [0.102 0.124 0.146 ... 0.478 0.5   0.523]
    >>> rectangle(X_4n, Y_4n, method='left')
    >>> # - I_4n:  0.15009808046684736
    >>> # |I_2n - I_4n|:  0.006146545385543184
    >>> # 0.006146545385543184 <= 0.01
    >>> # True
    >>> # |I_2n - I_4n|:  0.006146545385543184

    See Also
    --------
    rectangle
    """
    print("\n")
    print("Test 1: Rectangle Rule Error")
    print("- f(x) = x - lg(x + 2)")

    X_n = np.linspace(df["a"], df["b"], 5)
    print("- X_n: ", X_n)

    Y_n = df["f"](X_n)
    print("- Y_1: ", Y_n)

    X_2n = np.linspace(df["a"], df["b"], 10)
    print("- X_2n: ", X_2n)

    Y_2n = df["f"](X_2n)
    print("- Y_2n: ", Y_2n)

    print(">>> rectangle(X_n, Y_n, method='left')")

    I_n = rectangle(X_n, Y_n)
    print("- I_n: ", I_n)

    print(">>> rectangle(X_2n, Y_2n, method='left')")

    I_2n = rectangle(X_2n, Y_2n)
    print("- I_2n: ", I_2n)

    print("|I_n - I_2n|: ", abs(I_n - I_2n))

    print(f"{abs(I_n - I_2n)} <= {df['eps']}")
    print(abs(I_n - I_2n) <= df["eps"])

    if abs(I_n - I_2n) <= df["eps"]:
        assert abs(I_n - I_2n) <= df["eps"], f"|I_n - I_2n| > {df['eps']}"
        return

    X_4n = np.linspace(df["a"], df["b"], 20)
    print("- X_4n: ", X_4n)

    Y_4n = df["f"](X_4n)
    print("- Y_4n: ", Y_4n)

    print(">>> rectangle(X_4n, Y_4n, method='left')")

    I_4n = rectangle(X_4n, Y_4n)
    print("- I_4n: ", I_4n)

    print("|I_2n - I_4n|: ", abs(I_2n - I_4n))
    print(f"{abs(I_2n - I_4n)} <= {df['eps']}")
    print(abs(I_2n - I_4n) <= df["eps"])

    print("|I_2n - I_4n|: ", abs(I_2n - I_4n))
    if abs(I_2n - I_4n) <= df["eps"]:
        assert abs(I_2n - I_4n) <= df["eps"], f"|I_2n - I_4n| > {df['eps']}"
        return


def test_rectangle_from_docs_example():
    r"""Test rectangle rule from docs example.

    See Also
    --------
    rectangle
    """
    print("\n")
    print("Test N: Rectangle Rule From Docs Example")
    xvals = np.array([0, 1, 2, 3])
    yvals = np.array([0, 1, 4, 9])
    print("- X:", xvals)
    print("- Y:", yvals)

    print(f">>> rectangle({xvals}, {yvals}, method='left')")
    result = rectangle(xvals, yvals, method="left")
    print(result)

    assert isinstance(result, float), "Result is not a float"


def test_midpoint_from_docs_example():
    r"""Test midpoint rule from docs example.

    See Also
    --------
    midpoint
    """
    print("\n")
    print("Test N: Midpoint Rule From Docs Example")
    xvals = np.array([0, 1, 2, 3, 4])
    yvals = np.array([0, 1, 4, 9, 16])
    print("- X:", xvals)
    print("- Y:", yvals)

    print(f">>> midpoint({xvals}, {yvals})")
    result = midpoint(xvals, yvals)
    print(result)

    assert isinstance(result, float), "Result is not a float"


def test_trapezoid_from_docs_example():
    r"""Test trapezoid rule from docs example.

    See Also
    --------
    trapezoid
    """
    print("\n")
    print("Test N: Trapezoid Rule From Docs Example")
    xvals = np.array([0, 1, 2, 3])
    yvals = np.array([0, 1, 4, 9])
    print("- X:", xvals)
    print("- Y:", yvals)

    print(f">>> trapezoid({xvals}, {yvals})")
    result = trapezoid(xvals, yvals)
    print(result)

    assert isinstance(result, float), "Result is not a float"


def test_simpsonq_from_docs_example():
    r"""Test Simpson's rule (quadratic) from docs example.

    See Also
    --------
    simpsonq
    """
    print("\n")
    print("Test N: Simpson's Rule (Quadratic) From Docs Example")
    xvals = np.array([0, 1, 2, 3, 4])
    yvals = np.array([0, 1, 4, 9, 16])
    print("- X:", xvals)
    print("- Y:", yvals)

    print(f">>> simpsonq({xvals}, {yvals})")
    result = simpsonq(xvals, yvals)
    print(result)

    assert isinstance(result, float), "Result is not a float"


def test_simpsonc_from_docs_example():
    r"""Test Simpson's rule (cubic) from docs example.

    See Also
    --------
    simpsonc
    """
    print("\n")
    print("Test N: Simpson's Rule (Cubic) From Docs Example")
    xvals = np.array([0, 1, 2, 3, 4, 5, 6])
    yvals = np.array([0, 1, 4, 9, 16, 25, 36])
    print("- X:", xvals)
    print("- Y:", yvals)

    print(f">>> simpsonc({xvals}, {yvals})")
    result = simpsonc(xvals, yvals)
    print(result)

    assert isinstance(result, float), "Result is not a float"


def test_weddles_from_docs_example():
    r"""Test Weddle's rule from docs example.

    See Also
    --------
    weddles
    """
    print("\n")
    print("Test N: Weddle's Rule From Docs Example")
    xvals = np.array([0, 1, 2, 3, 4, 5, 6])
    yvals = np.array([0, 1, 4, 9, 16, 25, 36])
    print("- X:", xvals)
    print("- Y:", yvals)

    print(f">>> weddles({xvals}, {yvals})")
    result = weddles(xvals, yvals)
    print(result)

    assert isinstance(result, float), "Result is not a float"
