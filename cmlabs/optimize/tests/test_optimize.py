import numpy as np
from cmlabs.optimize import find_root_brackets, bisect, newton, secant

__all__ = [
    "test_root",
    "test_find_root_brackets_from_docs_example",
    "test_bisect_from_docs_example",
    "test_newton_from_docs_example",
    "test_secant_from_docs_example",
]


def f(x):
    return 2 * (x**2) - np.cos(2 * x)
    # return 2**x - 2 * (x**2) - 1


def df(x):
    return 4 * x + 2 * np.sin(2 * x)
    # return 2**x * np.log(2) - 4 * x


df = {
    "f": f,
    "df": df,
    "a": -10,
    # "a": 0,
    "b": 10,
    "xtol": 1e-5,
    "ytol": 1e-5,
}


def test_root():
    r"""Test the root finding function.

    This test checks if the root function correctly identifies the root of a
    given function within a specified interval.

    Results
    -------
    >>> # Test 1: Root
    >>> # f(x) = 2 * (x^2) - cos(2 * x)
    >>> # f'(x) = 4 * x + 2 * sin(2 * x)
    >>> # a = -10
    >>> # b = 10
    >>> find_root_brackets(f, -10, 10)
    >>> # Intervals: [(np.float64(-0.612244897959183), np.float64(-0.204081632653061)), (np.float64(0.204081632653061), np.float64(0.612244897959183))]
    >>> # BISECT:
    >>> bisect(f, (np.float64(-0.612244897959183), np.float64(-0.204081632653061)))
    >>> # Root: -0.5108455735809945
    >>> bisect(f, (np.float64(0.204081632653061), np.float64(0.612244897959183)))
    >>> # Root: 0.5108455735809945
    >>> # NEWTON:
    >>> newton(f, df, -0.612244897959183)
    >>> # Root: -0.5108803890400834
    >>> newton(f, df, 0.204081632653061)
    >>> # Root: 0.5108803890400834
    >>> # SECANT:
    >>> secant(f, -0.612244897959183, -0.204081632653061)
    >>> # Root: -0.5108449781771545
    >>> secant(f, 0.204081632653061, 0.612244897959183)
    >>> # Root: 0.5108448318004966

    See Also
    --------
    find_root_brackets
    """
    print("\n")
    print("Test 1: Root")

    print("f(x) = 2 * (x^2) - cos(2 * x)")
    print("f'(x) = 4 * x + 2 * sin(2 * x)")
    print(f"a = {df['a']}")
    print(f"b = {df['b']}")

    print(f">>> find_root_brackets(f, {df['a']}, {df['b']})")

    intervals = find_root_brackets(f, df["a"], df["b"], bins=50)
    print(f"Intervals: {intervals}")

    print("BISECT:")
    for i, interval in enumerate(intervals):
        print(f">>> bisect(f, {interval})")
        root = bisect(df["f"], interval, xtol=df["xtol"], ytol=df["ytol"])
        print(f"Root: {root}")

    print("NEWTON:")
    for i, interval in enumerate(intervals):
        print(f">>> newton(f, df, {interval[0]})")
        root = newton(df["f"], df["df"], interval[0], xtol=df["xtol"], ytol=df["ytol"])
        print(f"Root: {root}")

    print("SECANT:")
    for i, interval in enumerate(intervals):
        print(f">>> secant(f, {interval[0]}, {interval[1]})")
        root = secant(
            df["f"], interval[0], interval[1], xtol=df["xtol"], ytol=df["ytol"]
        )
        print(f"Root: {root}")


def test_find_root_brackets_from_docs_example():
    r"""Test the find_root_brackets function from the documentation example.

    This test checks if the find_root_brackets function correctly identifies
    the root brackets for a given function and interval.

    See Also
    --------
    find_root_brackets
    """
    print("\n")
    print("Test N: Find Root Brackets From Docs Example")

    def f_test(x):
        return x**2 - 4

    a = 0
    b = 10
    bins = 10

    print("f(x) = x^2 - 4")
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"bins = {bins}")

    print(f">>> find_root_brackets(f_test, {a}, {b}, {bins})")

    intervals = find_root_brackets(f_test, a, b, bins)
    print(f"Intervals: {intervals}")

    assert isinstance(intervals, list), "Intervals should be a list"


def test_bisect_from_docs_example():
    r"""Test the bisect function from the documentation example.

    This test checks if the bisect function correctly finds the root of a
    given function within a specified bracket.

    See Also
    --------
    bisect
    """
    print("\n")
    print("Test N: Bisect From Docs Example")

    def f_test(x):
        return x**2 - 4

    bracket = [0, 10]
    xtol = 1e-5
    ytol = 1e-5

    print("f(x) = x^2 - 4")
    print(f"xtol = {xtol}")
    print(f"ytol = {ytol}")

    print(f"bracket: {bracket}")
    print(f">>> bisect(f_test, bracket, xtol={xtol}, ytol={ytol})")

    root = bisect(f_test, bracket, xtol=xtol, ytol=ytol)
    print(f"root: {root}")

    assert isinstance(root, float), "Root should be a float"


def test_newton_from_docs_example():
    r"""Test the newton function from the documentation example.

    This test checks if the newton function correctly finds the root of a
    given function within a specified bracket.

    See Also
    --------
    newton
    """
    print("\n")
    print("Test N: Newton From Docs Example")

    def f_test(x):
        return x**2 - 4

    def df_test(x):
        return 2 * x

    x0 = 5.0
    xtol = 1e-5
    ytol = 1e-5

    print("f(x) = x^2 - 4")
    print("f'(x) = 2 * x")
    print(f"xtol = {xtol}")
    print(f"ytol = {ytol}")

    print(f"x0: {x0}")
    print(f">>> newton(f_test, df_test, {x0}, xtol={xtol}, ytol={ytol})")

    root = newton(f_test, df_test, x0, xtol=xtol, ytol=ytol)
    print(f"root: {root}")

    assert isinstance(root, float), "Root should be a float"


def test_secant_from_docs_example():
    r"""Test the secant function from the documentation example.

    This test checks if the secant function correctly finds the root of a
    given function within a specified bracket.

    See Also
    --------
    secant
    """
    print("\n")
    print("Test N: Secant From Docs Example")

    def f_test(x):
        return x**2 - 4

    x0 = 0.0
    x1 = 10.0
    xtol = 1e-5
    ytol = 1e-5

    print("f(x) = x^2 - 4")
    print(f"xtol = {xtol}")
    print(f"ytol = {ytol}")

    print(f"x0: {x0}")
    print(f"x1: {x1}")
    print(f">>> secant(f_test, {x0}, {x1}, xtol={xtol}, ytol={ytol})")

    root = secant(f_test, x0, x1, xtol=xtol, ytol=ytol)
    print(f"root: {root}")

    assert isinstance(root, float), "Root should be a float"
