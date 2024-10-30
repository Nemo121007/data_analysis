import pandas as pd


def main():
    # Исходный список
    transaction = [120, -31, '20.1', 12.3, 'bank', 12, -4, -7, 150, 'mr.', 23, 32, 21]

    # Создаем объект Series с индексами от 10 до 22
    t = pd.Series(transaction, index=range(10, 23))
    print(f"t: \n{t}")

    # Фильтруем только целые числа
    t_integers = t[t.apply(lambda x: isinstance(x, int))]
    print(f"t_integers: \n{t_integers}")

    # Вычисляем несмещенную выборочную дисперсию и среднее значение
    variance = t_integers.var(ddof=1)
    mean = t_integers.mean()

    # Вывод результатов
    print(f"Несмещенная выборочная дисперсия: {variance}")
    print(f"Среднее значение: {mean}")


if __name__ == "__main__":
    main()
