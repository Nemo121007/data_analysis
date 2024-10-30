from scipy.integrate import solve_ivp, odeint
import numpy as np

# Определение уравнения y' = 2x
def f(y, x):
    return 2 * x  # Производная по x

if __name__ == "__main__":
    # Начальные условия
    x0 = 0
    y0 = [1]  # y(0) = 1
    x_end = 3
    tolerance = 0.01  # Заданная погрешность


    # print(f"Использованный метод: RK45")
    # '''
    # Результат: приближенное значение y(3)
    # [x0, x_end] — интервал решения от 0 до 3,
    # y0 — начальные условия 𝑦(0)=1
    # method — метод интегрирования, в данном случае используем метод Рунге-Кутта 4-го порядка (по умолчанию это 'RK45').
    # rtol и atol — относительная и абсолютная точности, соответственно. Устанавливаем atol в соответствии с требуемой точностью tolerance.
    # '''
    # sol = solve_ivp(f, [x0, x_end], y0, method='RK45', rtol=1e-3, atol=tolerance)
    #
    # y_approx = sol.y[0][-1]  # sol.y[0] — массив решений y, [-1] — последнее значение
    #
    # print(f"Приближенное значение y(3) = {y_approx}")

    print("Иттерационный подбор точности")

    y0 = y0[0]
    step = x_end - x0
    result = 0
    error = 1
    # Если погрешность больше 0.01, то увеличиваем количество точек
    while error > tolerance:
        step /= 2
        x_field = np.arange(x0, x_end + 0.01, step)
        sol = odeint(f, y0, x_field)
        y3 = sol[-1][0]
        print(f"y(3) = {y3}")
        result = y3

        x_field_fine = np.arange(x0, x_end + 0.01, step / 100)
        sol_fine = odeint(f, y0, x_field_fine)
        y3_fine = sol_fine[-1][0]

        # Сравниваем результаты
        error = abs(y3 - y3_fine)
        print(f"Погрешность: {error}")

    print(f"Итоговое значение y(3) = {result} с требуемой точностью: {error}")
    print(f"Шаг: {step}")

