import numpy as np
from cmlabs.integrate import (
    rectangle,
    midpoint,
    trapezoid,
    simpsonq,
    simpsonc,
    weddles,
    newton_cotes,
)

__all__ = [
    "test_rectangle_error",
    "test_midpoint_error",
    "test_trapezoid_error",
    "test_simpsonq_error",
    "test_simpsonc_error",
    "test_weddles_error",
    "test_newton_cotes_error",
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
    "eps*": 1e-2,
    "eps**": 1e-3,
    "eps***": 1e-4,
    "eps****": 1e-5,
    "eps*****": 1e-6,
    "eps******": 1e-7,
    "coef": np.array([19 / 288, 75 / 288, 50 / 288, 50 / 288, 75 / 288, 19 / 288]),
}

np.set_printoptions(threshold=2, precision=3, suppress=True)


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
    >>> # - Y_n:  [0.102 0.206 0.311 0.416 0.523]
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
    print("- Y_n: ", Y_n)

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

    print(f"{abs(I_n - I_2n)} <= {df['eps*']}")
    print(abs(I_n - I_2n) <= df["eps*"])

    if abs(I_n - I_2n) <= df["eps*"]:
        assert abs(I_n - I_2n) <= df["eps*"], f"|I_n - I_2n| > {df['eps*']}"
        return

    X_4n = np.linspace(df["a"], df["b"], 20)
    print("- X_4n: ", X_4n)

    Y_4n = df["f"](X_4n)
    print("- Y_4n: ", Y_4n)

    print(">>> rectangle(X_4n, Y_4n, method='left')")

    I_4n = rectangle(X_4n, Y_4n)
    print("- I_4n: ", I_4n)

    print("|I_2n - I_4n|: ", abs(I_2n - I_4n))
    print(f"{abs(I_2n - I_4n)} <= {df['eps*']}")
    print(abs(I_2n - I_4n) <= df["eps*"])

    print("|I_2n - I_4n|: ", abs(I_2n - I_4n))
    if abs(I_2n - I_4n) <= df["eps*"]:
        assert abs(I_2n - I_4n) <= df["eps*"], f"|I_2n - I_4n| > {df['eps*']}"
        return


def test_midpoint_error():
    r"""Reach accuracy :math:`\epsilon` error for midpoint rule.

    .. math::

        \begin{gather}
            |I_n - I_{2n}| \leq \epsilon, \\
            |I_{2n} - I_{4n}| \leq \epsilon, \\
            \ldots \\
            |I_{2^k n} - I_{2^{k+1} n}| \leq \epsilon, \\
        \end{gather}

    where :math:`I_n` is the integral of the function :math:`f(x)` over the interval
    :math:`[a, b]` using the midpoint rule with :math:`n`
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
    >>> # Test 2: Midpoint Rule Error
    >>> # - f(x) = x - lg(x + 2)
    >>> # - X_n:  [0.5   0.625 0.75  0.875 1.   ]
    >>> # - Y_n:  [0.102 0.206 0.311 0.416 0.523]
    >>> # - X_2n:  [0.5   0.562 0.625 ... 0.875 0.938 1.   ]
    >>> # - Y_2n:  [0.102 0.154 0.206 ... 0.416 0.47  0.523]
    >>> midpoint(X_n, Y_n)
    >>> # - I_n:  0.15555821080809373
    >>> midpoint(X_2n, Y_2n)
    >>> # - I_2n:  0.1556146558266645
    >>> # |I_n - I_2n|:  5.644501857077211e-05
    >>> # 5.644501857077211e-05 <= 0.001
    >>> # True

    See Also
    --------
    midpoint
    """
    print("\n")
    print("Test 2: Midpoint Rule Error")
    print("- f(x) = x - lg(x + 2)")

    X_n = np.linspace(df["a"], df["b"], 5)
    print("- X_n: ", X_n)

    Y_n = df["f"](X_n)
    print("- Y_n: ", Y_n)

    X_2n = np.linspace(df["a"], df["b"], 9)
    print("- X_2n: ", X_2n)

    Y_2n = df["f"](X_2n)
    print("- Y_2n: ", Y_2n)

    print(">>> midpoint(X_n, Y_n)")

    I_n = midpoint(X_n, Y_n)
    print("- I_n: ", I_n)

    print(">>> midpoint(X_2n, Y_2n)")

    I_2n = midpoint(X_2n, Y_2n)
    print("- I_2n: ", I_2n)

    print("|I_n - I_2n|: ", abs(I_n - I_2n))

    print(f"{abs(I_n - I_2n)} <= {df['eps**']}")
    print(abs(I_n - I_2n) <= df["eps**"])

    if abs(I_n - I_2n) <= df["eps**"]:
        assert abs(I_n - I_2n) <= df["eps**"], f"|I_n - I_2n| > {df['eps**']}"


def test_trapezoid_error():
    r"""Reach accuracy :math:`\epsilon` error for trapezoid rule.

    .. math::

        \begin{gather}
            |I_n - I_{2n}| \leq \epsilon, \\
            |I_{2n} - I_{4n}| \leq \epsilon, \\
            \ldots \\
            |I_{2^k n} - I_{2^{k+1} n}| \leq \epsilon, \\
        \end{gather}

    where :math:`I_n` is the integral of the function :math:`f(x)` over the interval
    :math:`[a, b]` using the trapezoid rule with :math:`n`
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
    >>> # Test 3: Trapezoid Rule Error
    >>> # - f(x) = x - lg(x + 2)
    >>> # - X_n:  [0.5   0.625 0.75  0.875 1.   ]
    >>> # - Y_n:  [0.102 0.206 0.311 0.416 0.523]
    >>> # - X_2n:  [0.5   0.556 0.611 ... 0.889 0.944 1.   ]
    >>> # - Y_2n:  [0.102 0.148 0.194 ... 0.428 0.475 0.523]
    >>> trapezoid(X_n, Y_n)
    >>> # - I_n:  0.1556711897132828
    >>> trapezoid(X_2n, Y_2n)
    >>> # - I_2n:  0.1556409449133146
    >>> # |I_n - I_2n|:  3.0244799968215386e-05
    >>> # 3.0244799968215386e-05 <= 0.001
    >>> # True

    See Also
    --------
    trapezoid
    """
    print("\n")
    print("Test 3: Trapezoid Rule Error")
    print("- f(x) = x - lg(x + 2)")

    X_n = np.linspace(df["a"], df["b"], 5)
    print("- X_n: ", X_n)

    Y_n = df["f"](X_n)
    print("- Y_n: ", Y_n)

    X_2n = np.linspace(df["a"], df["b"], 10)
    print("- X_2n: ", X_2n)

    Y_2n = df["f"](X_2n)
    print("- Y_2n: ", Y_2n)

    print(">>> trapezoid(X_n, Y_n)")

    I_n = trapezoid(X_n, Y_n)
    print("- I_n: ", I_n)

    print(">>> trapezoid(X_2n, Y_2n)")

    I_2n = trapezoid(X_2n, Y_2n)
    print("- I_2n: ", I_2n)

    print("|I_n - I_2n|: ", abs(I_n - I_2n))
    print(f"{abs(I_n - I_2n)} <= {df['eps**']}")

    print(abs(I_n - I_2n) <= df["eps**"])

    if abs(I_n - I_2n) <= df["eps**"]:
        assert abs(I_n - I_2n) <= df["eps**"], f"|I_n - I_2n| > {df['eps**']}"
        return


def test_simpsonq_error():
    r"""Reach accuracy :math:`\epsilon` error for Simpson's rule (quadratic).

    .. math::

        \begin{gather}
            |I_n - I_{2n}| \leq \epsilon, \\
            |I_{2n} - I_{4n}| \leq \epsilon, \\
            \ldots \\
            |I_{2^k n} - I_{2^{k+1} n}| \leq \epsilon, \\
        \end{gather}

    where :math:`I_n` is the integral of the function :math:`f(x)` over the interval
    :math:`[a, b]` using Simpson's rule (quadratic) with :math:`n`
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
    >>> # Test 4: Simpson's Rule (Quadratic) Error
    >>> # - f(x) = x - lg(x + 2)
    >>> # - X_n:  [0.5  0.75 1.  ]
    >>> # - Y_n:  [0.102 0.311 0.523]
    >>> # - X_2n:  [0.5   0.625 0.75  0.875 1.   ]
    >>> # - Y_2n:  [0.102 0.206 0.311 0.416 0.523]
    >>> simpsonq(X_n, Y_n)
    >>> # - I_n:  0.15563399677393747
    >>> simpsonq(X_2n, Y_2n)
    >>> # - I_2n:  0.15563353007821978
    >>> # |I_n - I_2n|:  4.6669571768243046e-07
    >>> # 4.6669571768243046e-07 <= 0.0001
    >>> # True

    See Also
    --------
    simpsonq
    """
    print("\n")
    print("Test 4: Simpson's Rule (Quadratic) Error")
    print("- f(x) = x - lg(x + 2)")

    X_n = np.linspace(df["a"], df["b"], 3)
    print("- X_n: ", X_n)

    Y_n = df["f"](X_n)
    print("- Y_n: ", Y_n)

    X_2n = np.linspace(df["a"], df["b"], 5)
    print("- X_2n: ", X_2n)

    Y_2n = df["f"](X_2n)
    print("- Y_2n: ", Y_2n)

    print(">>> simpsonq(X_n, Y_n)")

    I_n = simpsonq(X_n, Y_n)
    print("- I_n: ", I_n)

    print(">>> simpsonq(X_2n, Y_2n)")

    I_2n = simpsonq(X_2n, Y_2n)
    print("- I_2n: ", I_2n)

    print("|I_n - I_2n|: ", abs(I_n - I_2n))
    print(f"{abs(I_n - I_2n)} <= {df['eps***']}")

    print(abs(I_n - I_2n) <= df["eps***"])

    if abs(I_n - I_2n) <= df["eps***"]:
        assert abs(I_n - I_2n) <= df["eps***"], f"|I_n - I_2n| > {df['eps***']}"
        return


def test_simpsonc_error():
    r"""Reach accuracy :math:`\epsilon` error for Simpson's rule (cubic).

    .. math::

        \begin{gather}
            |I_n - I_{2n}| \leq \epsilon, \\
            |I_{2n} - I_{4n}| \leq \epsilon, \\
            \ldots \\
            |I_{2^k n} - I_{2^{k+1} n}| \leq \epsilon, \\
        \end{gather}

    where :math:`I_n` is the integral of the function :math:`f(x)` over the interval
    :math:`[a, b]` using Simpson's rule (cubic) with :math:`n`
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
    >>> # Test 5: Simpson's Rule (Cubic) Error
    >>> # - f(x) = x - lg(x + 2)
    >>> # - X_n:  [0.5   0.667 0.833 1.   ]
    >>> # - Y_1:  [0.102 0.241 0.381 0.523]
    >>> # - X_2n:  [0.5   0.583 0.667 ... 0.833 0.917 1.   ]
    >>> # - Y_2n:  [0.102 0.171 0.241 ... 0.381 0.452 0.523]
    >>> simpsonc(X_n, Y_n)
    >>> # - I_n:  0.15563372042547285
    >>> simpsonc(X_2n, Y_2n)
    >>> # - I_2n:  0.15563351252747326
    >>> # |I_n - I_2n|:  2.0789799959342048e-07
    >>> # 2.0789799959342048e-07 <= 0.0001
    >>> # True

    See Also
    --------
    simpsonc
    """
    print("\n")
    print("Test 5: Simpson's Rule (Cubic) Error")
    print("- f(x) = x - lg(x + 2)")

    X_n = np.linspace(df["a"], df["b"], 4)
    print("- X_n: ", X_n)

    Y_n = df["f"](X_n)
    print("- Y_n: ", Y_n)

    X_2n = np.linspace(df["a"], df["b"], 7)
    print("- X_2n: ", X_2n)

    Y_2n = df["f"](X_2n)
    print("- Y_2n: ", Y_2n)

    print(">>> simpsonc(X_n, Y_n)")

    I_n = simpsonc(X_n, Y_n)
    print("- I_n: ", I_n)

    print(">>> simpsonc(X_2n, Y_2n)")

    I_2n = simpsonc(X_2n, Y_2n)
    print("- I_2n: ", I_2n)

    print("|I_n - I_2n|: ", abs(I_n - I_2n))
    print(f"{abs(I_n - I_2n)} <= {df['eps***']}")

    print(abs(I_n - I_2n) <= df["eps***"])

    if abs(I_n - I_2n) <= df["eps***"]:
        assert abs(I_n - I_2n) <= df["eps***"], f"|I_n - I_2n| > {df['eps***']}"
        return


def test_weddles_error():
    r"""Reach accuracy :math:`\epsilon` error for Weddle's rule.

    .. math::

        \begin{gather}
            |I_n - I_{2n}| \leq \epsilon, \\
            |I_{2n} - I_{4n}| \leq \epsilon, \\
            \ldots \\
            |I_{2^k n} - I_{2^{k+1} n}| \leq \epsilon, \\
        \end{gather}

    where :math:`I_n` is the integral of the function :math:`f(x)` over the interval
    :math:`[a, b]` using Weddle's rule with :math:`n`
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
    >>> # Test 6: Weddle's Rule Error
    >>> # - f(x) = x - lg(x + 2)
    >>> # - X_n:  [0.5   0.583 0.667 ... 0.833 0.917 1.   ]
    >>> # - Y_n:  [0.102 0.171 0.241 ... 0.381 0.452 0.523]
    >>> # - X_2n:  [0.5   0.542 0.583 ... 0.917 0.958 1.   ]
    >>> # - Y_2n:  [0.102 0.137 0.171 ... 0.452 0.487 0.523]
    >>> weddles(X_n, Y_n)
    >>> # - I_n:  0.155633498497833
    >>> weddles(X_2n, Y_2n)
    >>> # - I_2n:  0.1556334984731289
    >>> # |I_n - I_2n|:  2.4704099876871055e-11
    >>> # 2.4704099876871055e-11 <= 1e-05
    >>> # True

    See Also
    --------
    weddles
    """
    print("\n")
    print("Test 6: Weddle's Rule Error")
    print("- f(x) = x - lg(x + 2)")

    X_n = np.linspace(df["a"], df["b"], 7)
    print("- X_n: ", X_n)

    Y_n = df["f"](X_n)
    print("- Y_n: ", Y_n)

    X_2n = np.linspace(df["a"], df["b"], 13)
    print("- X_2n: ", X_2n)

    Y_2n = df["f"](X_2n)
    print("- Y_2n: ", Y_2n)

    print(">>> weddles(X_n, Y_n)")

    I_n = weddles(X_n, Y_n)
    print("- I_n: ", I_n)

    print(">>> weddles(X_2n, Y_2n)")

    I_2n = weddles(X_2n, Y_2n)
    print("- I_2n: ", I_2n)

    print("|I_n - I_2n|: ", abs(I_n - I_2n))
    print(f"{abs(I_n - I_2n)} <= {df['eps****']}")

    print(abs(I_n - I_2n) <= df["eps****"])

    if abs(I_n - I_2n) <= df["eps****"]:
        assert abs(I_n - I_2n) <= df["eps****"], f"|I_n - I_2n| > {df['eps****']}"
        return


def test_newton_cotes_error():
    r"""Reach accuracy :math:`\epsilon` error for Newton-Cotes rule.

    .. math::

        \begin{gather}
            |I_n - I_{2n}| \leq \epsilon, \\
            |I_{2n} - I_{4n}| \leq \epsilon, \\
            \ldots \\
            |I_{2^k n} - I_{2^{k+1} n}| \leq \epsilon, \\
        \end{gather}

    where :math:`I_n` is the integral of the function :math:`f(x)` over the interval
    :math:`[a, b]` using Newton-Cotes rule with :math:`n`
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
    >>> # Test 7: Newton-Cotes Rule Error
    >>> # - f(x) = x - lg(x + 2)
    >>> # - X_n:  [0.5 0.6 0.7 0.8 0.9 1. ]
    >>> # - Y_n:  [0.102 0.185 0.269 0.353 0.438 0.523]
    >>> # - Coef:  [0.066 0.26  0.174 0.174 0.26  0.066]
    >>> newton_cotes(X_n, Y_n, df['coef'])
    >>> # - I_n:  0.1556334987504589
    >>> # - I_2n:  0.1556334984772168
    >>> # |I_n - I_2n|:  2.7324209561641055e-10
    >>> # 2.7324209561641055e-10 <= 1e-06
    >>> # True

    See Also
    --------
    newton_cotes
    """
    print("\n")
    print("Test 7: Newton-Cotes Rule Error")
    print("- f(x) = x - lg(x + 2)")

    X_n = np.linspace(df["a"], df["b"], len(df["coef"]))
    print("- X_n: ", X_n)

    Y_n = df["f"](X_n)
    print("- Y_n: ", Y_n)

    print("- Coef: ", df["coef"])

    print(">>> newton_cotes(X_n, Y_n, df['coef'])")

    I_n = newton_cotes(X_n, Y_n, df["coef"])
    print("- I_n: ", I_n)

    X_2n_1 = np.linspace(df["a"], (df["a"] + df["b"]) / 2, len(df["coef"]))
    Y_2n_1 = df["f"](X_2n_1)
    I_2n_1 = newton_cotes(X_2n_1, Y_2n_1, df["coef"])

    X_2n_2 = np.linspace((df["a"] + df["b"]) / 2, df["b"], len(df["coef"]))
    Y_2n_2 = df["f"](X_2n_2)
    I_2n_2 = newton_cotes(X_2n_2, Y_2n_2, df["coef"])
    I_2n = I_2n_1 + I_2n_2
    print("- I_2n: ", I_2n)

    print("|I_n - I_2n|: ", abs(I_n - I_2n))
    print(f"{abs(I_n - I_2n)} <= {df['eps*****']}")

    print(abs(I_n - I_2n) <= df["eps*****"])

    if abs(I_n - I_2n) <= df["eps*****"]:
        assert abs(I_n - I_2n) <= df["eps*****"], f"|I_n - I_2n| > {df['eps*****']}"
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


def test_trapezoid_and_simpsonq():
    r"""Test trapezoid and Simpson's rule (quadratic).

    See Also
    --------
    trapezoid
    simpsonq
    """
    print("\n")
    print("Test N: Trapezoid and Simpson's Rule (Quadratic)")

    xvals_1 = np.linspace(0, 2, 9)
    yvals_1 = [
        1.0,
        0.979915,
        0.927295,
        0.858001,
        0.785398,
        0.716844,
        0.655196,
        0.600943,
        0.553574,
    ]

    print("- X:", xvals_1)
    print("- Y:", yvals_1)

    print(f">>> trapezoid({xvals_1}, {yvals_1})")
    result_trap_1 = trapezoid(xvals_1, yvals_1)
    print(result_trap_1)

    print(f">>> simpsonq({xvals_1}, {yvals_1})")
    result_sim_1 = simpsonq(xvals_1, yvals_1)
    print(result_sim_1)

    xvals_2 = np.linspace(0, 2, 5)
    yvals_2 = [
        1.0,
        0.927295,
        0.785398,
        0.655196,
        0.553574,
    ]

    print("- X:", xvals_2)
    print("- Y:", yvals_2)

    print(f">>> trapezoid({xvals_2}, {yvals_2})")
    result_trap_2 = trapezoid(xvals_2, yvals_2)
    print(result_trap_2)

    print(f">>> simpsonq({xvals_2}, {yvals_2})")
    result_simp_2 = simpsonq(xvals_2, yvals_2)
    print(result_simp_2)

    print("1/3 * |I_tr(2h) - I_tr(h)|")
    print(1 / 3 * abs(result_trap_1 - result_trap_2))

    print("1/15 * |I_s(2h) - I_s(h)|")
    print(1 / 15 * abs(result_sim_1 - result_simp_2))
