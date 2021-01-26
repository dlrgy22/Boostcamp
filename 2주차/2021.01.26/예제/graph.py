import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

x, y = sym.symbols('x y') # symbol 설정
fun = x**2 + y**2 # func 설정

# 미분 함수 정의
gradfun = [sym.diff(fun, var) for var in (x,y)] # 미분
numgradfun = sym.lambdify([x, y], gradfun) # 미분 함수

X,Y = np.meshgrid(np.arange(-5,6, 0.5),np.arange(-5,6, 0.5)) # 범위 설정
graddat = numgradfun(X,Y) # gradient 계산

# graph
plt.figure()
plt.quiver(X,Y,graddat[0],graddat[1])
plt.show()