import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == "__main__":
    # Возможные значения X
    x_values = np.array([1, 2, 3, 4, 5, 6])

    # Вероятности для каждого значения
    probabilities = np.full_like(x_values, 1/6, dtype=np.float64)

    # Дискретное распределение
    discrete_value = stats.rv_discrete(values=(x_values, probabilities))

    # Математическое ожидание
    expected_value = discrete_value.mean()
    print(f"Математическое ожидание: {expected_value}")

    # Дисперсия
    expected_value_sq = discrete_value.var()
    variance = expected_value_sq - expected_value**2
    print(f"Дисперсия: {variance}")

    # Медиана
    median = discrete_value.median()
    print(f"Медиана: {median}")

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # Построение графика закона распределения
    ax1.bar(x_values, probabilities, width=0.5, color='lightblue', edgecolor='black')
    ax1.set_title("Закон распределения случайной величины X")
    ax1.set_xlabel("Значения X")
    ax1.set_ylabel("Вероятность")
    ax1.grid(True)

    # Построение графика функции распределения
    cdf_values = discrete_value.cdf(x_values)  # Вычисление CDF для каждого значения

    ax2.step(x_values, cdf_values, where='mid', color='blue')
    ax2.set_title("Функция распределения случайной величины X")
    ax2.set_xlabel("Значения X")
    ax2.set_ylabel("Накопленная вероятность")
    ax2.grid(True)
    plt.show()
