import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Определяем функцию
def f(x):
    return(-np.sin(2*x)) + np.sqrt(x)


if __name__ == "__main__":
    # Создаем массив значений x
    x_data = np.arange(0, 16)

    # Вычисляем значения функции в заданных точках
    y_data = f(x_data)

    # Создаем массив точек для интерполяции
    x_interp = np.linspace(0, 15, 100)

    # Линейная интерполяция
    f_linear = interp1d(x_data, y_data, kind='linear')
    y_linear = f_linear(x_interp)

    # Кубическая интерполяция
    f_cubic = interp1d(x_data, y_data, kind='cubic')
    y_cubic = f_cubic(x_interp)

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # Построение графиков
    ax1.plot(x_data, y_data, 'o', label='Исходные данные')
    ax1.plot(x_interp, y_linear, label='Линейная интерполяция')
    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Линейная интерполяция')
    ax1.grid(True)

    ax2.plot(x_data, y_data, 'o', label='Исходные данные')
    ax2.plot(x_interp, y_cubic, label='Кубическая интерполяция')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title('Кубическая интерполяция')
    ax2.grid(True)

    plt.title('Графики функции и ее интерполяций')
    plt.show()
