from scipy.integrate import quad, dblquad
import numpy as np


def f1(x):
    return 2 * np.tan(x) + 1


def f2(y, x):
    return x + y


if __name__ == "__main__":
    print("Стандартная функция выдаёт некритическую ошибку:")
    # Первый интеграл
    result1 = quad(f1, -1, 2)
    print(f"Первый интеграл:{result1[0]}    Погрешность{result1[1]}")
    print("Корректировка с учётом точек разрыва:")
    # Первый интеграл: делим на два участка из-за разрыва в pi/2
    pi_half = np.pi / 2

    # Интеграл от -1 до pi/2 - ε
    result1_part1 = quad(f1, -1, pi_half - 1e-5)

    # Интеграл от pi/2 + ε до 2
    result1_part2 = quad(f1, pi_half + 1e-5, 2)

    error = max(result1_part1[1], result1_part2[1])
    # Суммируем части
    result1 = result1_part1[0] + result1_part2[0]
    print(f"Первый интеграл: {result1}    Погрешность {error}")

    # Второй интеграл
    result2 = dblquad(f2, 0, 1, lambda x: x**2, lambda x: x)
    print(f"Вттрй интеграл: {result2[0]}    Погрешность {result2[1]}")
