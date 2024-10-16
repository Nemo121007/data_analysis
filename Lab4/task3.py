import numpy as np


def function(a, b, c):
    # Коэффициенты системы уравнений
    A = np.array([[1, 0, 1, 0],  # Уравнение: x + z = 2
                  [4, 0, 1, -2],  # Уравнение: 4x + z - 2w = 0
                  [a, b, 0, c],  # Уравнение: a*x + b*y + c*w = 5
                  [-1, 1, -2, 1]])  # Уравнение: -x + y - 2z + w = -2

    # Правая часть системы уравнений
    B = np.array([2, 0, 5, -2])

    # Решаем систему линейных уравнений
    solution = np.linalg.solve(A, B)

    # Находим сумму переменных x, y, z, w
    total_sum = np.sum(solution)

    # Округляем результат до целого числа
    return int(total_sum)


if __name__ == "__main__":
    # Пример использования:
    print(function(12, 6, 1))  # Ожидаемый результат: 2
    print(function(3, 6, 11))  # Ожидаемый результат: 6
