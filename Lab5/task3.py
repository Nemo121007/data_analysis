import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Определяем функцию
def f(x):
    return x * (1 - np.sin(2*x)) + np.sqrt(x)


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

    # Построение графиков
    plt.plot(x_data, y_data, 'o', label='Исходные данные')
    plt.plot(x_interp, y_linear, label='Линейная интерполяция')
    plt.plot(x_interp, y_cubic, label='Кубическая интерполяция')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Графики функции и ее интерполяций')
    plt.grid(True)
    plt.show()
