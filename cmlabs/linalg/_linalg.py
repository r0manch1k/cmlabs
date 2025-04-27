__all__ = ["thomas"]

from typing import List
import numpy as np


def thomas(
    A: List[float] | np.ndarray,
    B: List[float] | np.ndarray,
    C: List[float] | np.ndarray,
    F: List[float] | np.ndarray,
) -> np.ndarray:
    r"""
    Solve a tridiagonal system of equations using the Thomas algorithm.

    Parameters
    ----------
    A : array_like
        Sub-diagonal elements (length N-1).
    B : list of float
        Diagonal elements (length N).
    C : list of float
        Super-diagonal elements (length N-1).
    F : list of float
        Right-hand side vector (length N).

    Returns
    -------
    X : list of float
        Solution vector (length N).

    Notes
    -----
    The Thomas algorithm is an efficient way of solving tridiagonal matrix systems.

    .. math::

        \left(
            \begin{array}{cccccc}
                B_1 & C_1 & 0 & \cdots & 0 & 0 \\
                A_2 & B_2 & C_2 & \ddots & \vdots & \vdots \\
                0 & A_3 & B_3 & \ddots & 0 & 0 \\
                \vdots & \ddots & \ddots & \ddots & C_{N-2} & 0 \\
                0 & \cdots & 0 & A_{N-1} & B_{N-1} & C_{N-1} \\
                0 & \cdots & 0 & 0 & A_N & B_N
            \end{array}
        \right)
        \left(
            \begin{array}{c}
                x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_{N-1} \\ x_N
            \end{array}
        \right)
        =
        \left(
            \begin{array}{c}
                F_1 \\ F_2 \\ F_3 \\ \vdots \\ F_{N-1} \\ F_N
            \end{array}
        \right)

    Consider the first row of the matrix:

    .. math::

        \begin{gather}
            x_1 = \alpha_2 x_2 + \beta_2, \quad
            \alpha_2 = -\frac{C_1}{B_1}, \quad \beta_2 = \frac{F_1}{B_1}
        \end{gather}

    Let :math:`x_{k-1} = \alpha_k x_k + \beta_k`. Then, we can express the
    :math:`x_k` as:

    .. math::

        \begin{gather}
            x_k = \alpha_{k+1} x_{k+1} + \beta_{k+1}, \quad
            \alpha_{k+1} = -\frac{C_k}{B_k + A_k \alpha_k}, \quad
            \beta_{k+1} = \frac{F_k - A_k \beta_k}{B_k + A_k \alpha_k}
        \end{gather}

    Consider the last row of the matrix:

    .. math::

        \begin{gather}
            \begin{cases}
                A_N x_{N-1} + B_N x_N = F_N, \\
                x_{N-1} = \alpha_N x_N + \beta_N
            \end{cases}
            \quad \Longrightarrow \quad
            x_N = \frac{F_N - A_N \beta_N}{A_N \alpha_N + B_N}
        \end{gather}

    The algorithm consists of two steps: forward elimination and back substitution.
    The forward elimination step modifies the coefficients of the system to
    eliminate the sub-diagonal elements. The back substitution step solves the
    modified system to find the solution vector.

    The algorithm has a time complexity of O(N) and is particularly efficient
    for large systems of equations.

    Examples
    --------
    >>> from cmlabs.linalg import thomas
    >>> A = [1, 1]
    >>> B = [4, 4, 4]
    >>> C = [1, 1]
    >>> F = [5, 5, 5]
    >>> thomas(A, B, C, F)
    array([1.07142857 0.71428571 1.07142857])
    """
    N = len(B)

    if len(A) != N - 1:
        raise ValueError("Length of A must be N-1")
    if len(C) != N - 1:
        raise ValueError("Length of C must be N-1")
    if len(F) != N:
        raise ValueError("Length of F must be N")

    X = np.zeros(N)

    A = np.concatenate([[0], A])
    C = np.concatenate([C, [0]])

    # (alpha, beta)
    coef = np.zeros((N, 2))

    for i in range(0, N - 1):
        denom = B[i] + A[i] * coef[i, 0]
        coef[i + 1, 0] = -C[i] / denom
        coef[i + 1, 1] = (F[i] - A[i] * coef[i, 1]) / denom

    X[N - 1] = (F[N - 1] - A[N - 1] * coef[N - 1, 1]) / (
        A[N - 1] * coef[N - 1, 0] + B[N - 1]
    )

    for i in range(N - 2, -1, -1):
        X[i] = coef[i + 1, 0] * X[i + 1] + coef[i + 1, 1]

    return X
