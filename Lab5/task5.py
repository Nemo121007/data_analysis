from sympy import symbols, Eq, solve


if __name__ == "__main__":
    # Определяем символьные переменные
    x, y, z = symbols('x y z')

    # Задаем уравнения системы
    eq1 = Eq(x - y + z, 2)
    eq2 = Eq(2*x - y + z, 3)
    eq3 = Eq(3*x - 3*y + z, 0)

    # Решаем систему уравнений
    solution = solve((eq1, eq2, eq3), (x, y, z))

    # Выводим решение
    print(solution)
