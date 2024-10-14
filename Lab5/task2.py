from scipy.integrate import solve_ivp
import numpy as np

def f(x, y):
    """Функция правой части ОДУ"""
    return 2 * x


if __name__ == "__main__":
    # Начальное условие
    y0 = 1
    # Интервал интегрирования
    t_span = (0, 3)

    # Решение ОДУ
    sol = solve_ivp(f, t_span, [y0], method='RK45')

    # Значение y(3)
    y_3 = sol.y[0][-1]
    print("y(3) =", y_3)
