import sympy as sym
from sympy.abc import x, y

func = sym.diff(sym.poly(x**2 + 2*x + 3), x)
print(func)
print(func.subs(x, 2))
print(sym.diff(sym.poly(x**2 + 2*x*y + 3) + sym.cos(x + 2*y), x))
print(sym.diff(sym.poly(x**2 + 2*x*y + 3) + sym.cos(x + 2*y), y))
