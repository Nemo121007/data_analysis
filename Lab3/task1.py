import os
import re
import json
import datetime
import matplotlib.pyplot as plt

pattern_search = r".* \bA00000000001\b .*KEEP"
pattern_clip = r"(\d{2}:\d{2}:\d{2},\d+).*pressure=(\d+)"


def read_data(name_file='n_log1.txt'):
    result = []
    with open(name_file, 'r') as infile:
        for line in infile:
            # Проверяем наличие подстрок "KEEP" и "A00000000001" через pattern_search
            if re.search(pattern_search, line):
                match = re.search(pattern_clip, line)
                if match:
                    time, pressure = match.groups()
                    result.append([time, pressure])
    return result


def calculate_values(data_cut, interval_cut=10):
    # Преобразуем данные в более удобный формат (список кортежей)
    data_cut = [(datetime.datetime.strptime(time, "%H:%M:%S,%f"), int(pressure)) for time, pressure in data_cut]
    # Определяем текущий интервал
    current_interval = data_cut[0][0].minute // interval_cut

    result = []
    count, summ = 0, 0
    for time, pressure in data_cut:
        interval = time.minute // interval_cut
        if interval != current_interval:
            # Переходим к следующему интервалу
            result.append([current_interval, summ / count])
            current_interval = interval
            summ = 0
            count = 0
        # Добавляем давление к сумме и увеличиваем счетчик измерений
        summ += pressure
        count += 1
    result.append([current_interval, summ / count])

    times_data, pressures_data = zip(*result)

    return times_data, pressures_data


if __name__ == "__main__":
    data1 = []
    data2 = []

    if not (os.path.exists('n_log1.json') and os.path.exists('n_log2.json')):
        with open('n_log1.json', 'w') as f:
            data1 = read_data('n_log1.txt')
            json.dump(data1, f)
        with open('n_log2.json', 'w') as f:
            data2 = read_data('n_log2.txt')
            json.dump(data2, f)
    else:
        with open('n_log1.json', 'r') as f:
            data1 = json.load(f)
        with open('n_log2.json', 'r') as f:
            data2 = json.load(f)

    # Преобразуем данные в более удобный формат (список кортежей)
    data = [(datetime.datetime.strptime(time, "%H:%M:%S,%f"), int(pressure)) for time, pressure in data1]

    # Фильтруем данные по времени (при условии, что данные отсортированы по времени)
    start_time = datetime.datetime.strptime("14:10:00", "%H:%M:%S")
    end_time = datetime.datetime.strptime("14:15:00", "%H:%M:%S")
    filtered_data = [item for item in data if start_time <= item[0] <= end_time]

    # Извлекаем время и давление в отдельные списки
    times, pressures = zip(*filtered_data)

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)  # Две строки, два столбца. Выберем первую ячейку.
    ax2 = fig.add_subplot(2, 2, 2)  # Две строки, два столбца. Выберем вторую ячейку.
    ax3 = fig.add_subplot(2, 2, 3)  # Две строки, два столбца. Выберем третью ячейку.

    # Строим график
    ax1.plot(times, pressures)
    ax1.legend("A00000000001")
    ax1.set_xlabel("Время")
    ax1.set_ylabel("Давление")
    ax1.set_title("График 1. Давление 14:10 - 14:15 из n_log1.txt")
    ax1.grid(True)

    times, pressures = calculate_values(data2, 10)

    # Строим график
    ax2.plot(times, pressures)
    ax2.legend("A00000000001")
    ax2.set_xlabel('Время')
    ax2.set_ylabel('Давление')
    ax2.set_title('График2. Давление по 10 мин из n_log2.txt')
    ax2.grid(True)

    times_1, pressures_1 = calculate_values(data1, 20)

    times_2, pressures_2 = calculate_values(data2, 20)

    # Строим график
    ax3.plot(times_1, pressures_1)
    ax3.plot(times_2, pressures_2)
    ax3.legend(["n_log1.txt", "n_log2.txt"])
    ax3.set_xlabel('Время')
    ax3.set_ylabel('Давление')
    ax3.set_title('График3. Давление по 20 мин')
    ax3.grid(True)

    plt.show()
