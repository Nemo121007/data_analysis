import numpy as np

def function(mas):
    """
    Функция, выполняющая указанные операции над списком чисел.

    Args:
        mas: Список целых чисел.

    Returns:
        Сумма округленных среднего значения и среднеквадратичного отклонения.
    """

    # Создаем NumPy массив
    np_mas = np.array(mas)

    # Увеличиваем элементы с четными и нечетными индексами
    np_mas[::2] *= 2
    np_mas[1::2] += 5

    # Удаляем минимальное и максимальное значения
    np_mas = np.delete(np_mas, [np_mas.argmin(), np_mas.argmax()])

    # Вычисляем среднее значение и среднеквадратичное отклонение
    np_mas_avg = np.mean(np_mas)
    np_mas_std = np.std(np_mas)

    # Округляем и возвращаем сумму
    return int(np_mas_avg) + int(np_mas_std)

if __name__ == "__main__":
    # Примеры использования
    print(function([11, 22, 33, 44]))  # Вывод: 49
    print(function([110, 22, 33, 44]))  # Вывод: 65