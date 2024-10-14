import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Значения случайной величины
    x = np.arange(1, 7)
    # Вероятности
    p = np.ones(6) / 6

    # Построение графика функции распределения
    plt.step(x, np.cumsum(p), where='post')
    plt.title('Функция распределения')
    plt.xlabel('x')
    plt.ylabel('P(X <= x)')
    plt.grid(True)
    plt.show()

    # Построение столбчатой диаграммы закона распределения
    plt.bar(x, p)
    plt.title('Закон распределения')
    plt.xlabel('x')
    plt.ylabel('P(X = x)')
    plt.grid(True)
    plt.show()
    