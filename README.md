# cmlabs

**cmlabs** is a computational mathematics library for Python

## Installation

```bash
pip install cmlabs
```

## Usage

### Interpolation

```python
from cmlabs.interpolate import lagrange, newton, CubicSpline

xvals = np.array([0, 2, 3, 5])
yvals = np.array([1, 3, 2, 5])

y = lagrange(xvals, yvals, 1.5)
print(y)
# 3.3375000000000004

y = newton(xvals, yvlas, 1.5)
print(y)
# 3.3375

cs = CubicSpline(x, y, bc_type='not-a-knot')
print(cs.interpolate(0.5))
# 3.3375000000000004
```

### Integration

```python
from cmlabs.integrate import rectangle

xvals = np.array([0, 1, 2, 3])
yvals = np.array([0, 1, 4, 9])

I = rectangle(xvals, yvals, method="left")
print(I)
# 5.0
```

### Differentiation

```python
from cmlabs.differentiate import lagrange_derivative

xvals = np.array([0, 1, 2, 3, 4])
yvals = np.array([0, 1, 4, 9, 16])

L = lagrange_derivative(xvals, yvals, 2.5, 2)
print(L)
# 2.0
```

### Linear Algebra

```python
from cmlabs.linalg import thomas

A = [1, 1]
B = [4, 4, 4]
C = [1, 1]
F = [5, 5, 5]

solution = thomas(A, B, C, F)
print(solution)
# [1.071 0.714 1.071]
```

### Optimization

```python
from cmlabs.optimize import find_root_brackets, bisect

def f(x):
    return x**2 - 4

intervals = find_root_brackets(f, a=0, b=10, bins=10)
print(intervals)
# [(1.1111111111111112, 2.2222222222222223)]

root = bisect(f_test, intervals[0], xtol=1e-5, ytol=1e-5)
print(root)
# 1.9999980926513672
```

If you find any errors in the formulas or code, please feel free to open a [github issue](https://github.com/r0manch1k/cmlabs/issues)
