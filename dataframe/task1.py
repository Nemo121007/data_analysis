import pandas as pd


def series_data(name, index):
    # Генерируем серию всех нечётных положительных чисел, кратных 15 и меньше 10000
    odd_multiples_of_15 = [x for x in range(15, 10000, 15) if x % 2 == 1]
    index_pd = [x for x in range(1, len(odd_multiples_of_15) + 1)]
    s = pd.Series(odd_multiples_of_15, index=index_pd, name=name)

    # Применяем преобразования в зависимости от индекса
    for i in range(index, len(s) - 1):
        if i % 2 == 1:
            s[i] *= 2
        else:
            s[i] -= 70

    indices = [s[index], s[index] + 1, s[index - 5]]

    # Возвращаем максимальное значение
    return max([s[index], s[index + 1], s[index - 5]])


if __name__ == "__main__":
    # Пример вызова функции
    print(series_data('Название серии', 8))   # Ожидается результат 510
    print(series_data('Название серии', 18))  # Ожидается результат 1110
