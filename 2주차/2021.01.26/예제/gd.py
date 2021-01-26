import numpy as np
import sympy as sym
from sympy.abc import x, y

def eval_(fun, val):
    """함수와 변수값를 매개변수로 받아 함수의 변수에 변수값을 대입하여 계산된 값을 반환

    Args:
        fun : 입력함수
        val : 변수값

    Returns:
        fun_eval : 변수에 값을 넣어 계산된 값
    """    
    val_x, val_y = val
    fun_eval = fun.subs(x, val_x).subs(y, val_y)
    return fun_eval


def func_multi(val):
    """변수값을 입력으로 받고 함수식과 함수에 변수를 대입한 값을 반환
    Args:
        val : 대입할 변수

    Returns:
        eval_(func, [x_, y_]) : 대입한 값
        func : 함수식
    """    
    x_, y_ = val
    func = sym.poly(x**2 + 2*y**2)
    return eval_(func, [x_, y_]), func


def func_gradient(fun, val):
    """매개변수로 받은 함수의 gradient 벡터를 반환한다.

    Args:
        fun : 함수
        val : 대입할 변수

    Returns:
        grad_vec : gradient 백터
        [diff_x, diff_y] : 편미분된 값
    """    
    x_, y_ = val
    _, function = fun(val)
    diff_x = sym.diff(function, x)
    diff_y = sym.diff(function, y)
    grad_vec = np.array([eval_(diff_x, [x_, y_]), eval_(diff_y, [x_, y_])], dtype=float)
    return grad_vec, [diff_x, diff_y]


def gradient_descent(fun, init_point, lr_rate=1e-2, epsilon=1e-5):
    """ 경사하강법 구현

    Args:
        fun : 함수
        init_point : 초기 좌표
        lr_rate : 학습율 Defaults to 1e-2.
        epsilon : 종료조건 Defaults to 1e-5.
    """    
    cnt=0
    val = init_point
    diff, _ = func_gradient(fun, val)
    while np.linalg.norm(diff) > epsilon:
        val = val- lr_rate * diff
        diff, _ = func_gradient(fun, val)
        cnt += 1

    print(f"함수: {fun(val)[1]}, 연산횟수: {cnt}, 최소점: ({val}, {fun(val)[0]})")

if __name__ == "__main__":
    pt=[np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
    gradient_descent(fun=func_multi, init_point=pt)