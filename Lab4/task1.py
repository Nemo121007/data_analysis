import numpy as np


def function(int_list):
    # Создаем numpy массив из списка int_list
    n_list = np.array(int_list)

    # Применяем маски для изменения элементов массива
    n_list[n_list < 0] = 0  # Все отрицательные элементы приравниваем к 0
    n_list[n_list > 0] *= 4  # Все положительные элементы умножаем на 4

    # Сортируем массив по убыванию
    n_list = np.sort(n_list)[::-1]

    # Возвращаем сумму двух первых элементов
    return n_list[0] + n_list[1]


if __name__ == "__main__":
    print(function([1, 2, 3, -3, 0]))  # Вывод: 20
    print(function([1, 2, 13, -3]))  # Вывод: 60
