import numpy as np
from scipy import linalg

from cmlabs.linalg import thomas

__all__ = ["test_thomas", "test_thomas_from_docs_example"]

np.set_printoptions(threshold=5, precision=3, suppress=True)


def test_thomas():
    r"""Test Thomas Algorithm.

    Example
    -------
    >>> # Test N: Thomas Algorithm
    >>> # A:  [1 2 3 4 5]
    >>> # B:  [1 2 3 4 5 6]
    >>> # C:  [1 2 3 4 5]
    >>> # F:  [1 2 3 4 5 6]
    >>> thomas(A, B, C, F)
    >>> # res:  [-8.774  9.774 -4.387 -1.129  5.419 -3.516]
    >>> # res_scipy:  [-8.774  9.774 -4.387 -1.129  5.419 -3.516]

    See Also
    --------
    thomas
    """
    print("\n")
    print("Test N: Thomas Algorithm")

    A = np.array([1, 2, 3, 4, 5])
    B = np.array([1, 2, 3, 4, 5, 6])
    C = np.array([1, 2, 3, 4, 5])
    F = np.array([1, 2, 3, 4, 5, 6])

    print("A: ", A)
    print("B: ", B)
    print("C: ", C)
    print("F: ", F)

    print(">>> thomas(A, B, C, F)")

    res = thomas(A, B, C, F)
    print("res: ", res)

    res_scipy = linalg.solve_banded(
        (1, 1), np.array([np.concatenate([[0], A]), B, np.concatenate([C, [0]])]), F
    )
    print("res_scipy: ", res_scipy)

    assert np.allclose(res, res_scipy)


def test_thomas_from_docs_example():
    r"""Thomas From Docs.

    See Also
    --------
    thomas
    """
    print("\n")
    print("Test N: Thomas From Docs Example")

    A = [1, 1]
    B = [4, 4, 4]
    C = [1, 1]
    F = [5, 5, 5]

    print("A: ", A)
    print("B: ", B)
    print("C: ", C)
    print("F: ", F)

    print(">>> thomas(A, B, C, F)")

    res = thomas(A, B, C, F)
    print("res: ", res)

    assert isinstance(res, np.ndarray)
