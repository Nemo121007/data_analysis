import numpy as np
import pandas as pd


def main():
    # 1. Генерация 200 значений нормально распределённой случайной величины
    s = pd.Series(np.random.normal(size=200))

    # 2. Возведение каждого значения s во 2 степень и увеличение индексов на 2
    s = s**2
    s.index = s.index + 2

    # 3. Подсчёт количества значений, которые больше 2
    count_greater_than_2 = (s > 2).sum()
    print("Количество значений s, которые больше 2:", count_greater_than_2)

    # 4. Вычисление суммы элементов, строго меньших 2 и имеющих нечётные индексы
    sum_less_than_2_odd_index = s[(s < 2) & (s.index % 2 == 1)].sum()
    print("Сумма элементов, строго меньших 2 и имеющих нечётные индексы:", sum_less_than_2_odd_index)


if __name__ == "__main__":
    main()
