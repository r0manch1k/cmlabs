# cmlabs

**cmlabs** is a computational mathematics library for Python

## Installation

```bash
pip install cmlabs
```

## Usage

### Interpolation

```python
from cmlabs.interpolate import lagrange, newton
from cmlabs.cubic import cubic_spline

# Sample data
x = [0, 1, 2]
y = [1, 2, 0]

# Lagrange interpolation
y = lagrange(x, y, 0.5)
print(y)

# Newton interpolation
y = newton(x, y, 0.5)
print(y)

# Cubic spline interpolation
spline = cubic_spline(x, y, bc_type='not-a-knot')
print(spline.interpolate(0.5))
```

### Differentiation

```python
from cmlabs.differentiate import lagrange_derivative

# Differentiate Lagrange polynomial at a point
x = [0, 1, 2]
y = [1, 2, 0]
value = lagrange_derivative(x, y, 1.5, 2)
print(value)
```

### Linear Algebra

```python
from cmlabs.linalg import thomas

# Solve tridiagonal system Ax = d
A = [1, 1]  # sub-diagonal
B = [4, 4, 4]  # main diagonal
C = [1, 1]  # super-diagonal
d = [7, 8, 7]  # right-hand side

solution = thomas(A, B, C, d)
print(solution)
```

If you find any errors in the formulas or code, please feel free to open a [github issue](https://github.com/r0manch1k/cmlabs/issues)
