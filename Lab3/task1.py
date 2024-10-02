import os
import re
import json
import datetime
import matplotlib.pyplot as plt

pattern_search = r".* \bA00000000001\b .*KEEP"
pattern_clip = r"(\d{2}:\d{2}:\d{2},\d+).*pressure=(\d+)"

def read_data(name_file = 'n_log1.txt'):
    result = []
    # Открываем исходный файл для чтения и выходной файл для записи
    with open(name_file, 'r') as infile, open('out1.txt', 'w') as outfile:

        # Проходим по каждой строке в исходном файле
        for line in infile:
            # Проверяем наличие подстрок "KEEP" и "A00000000001"
            if re.search(pattern_search, line):
                match = re.search(pattern_clip, line)
                if match:
                    time, pressure = match.groups()
                    result.append([time, pressure])

    return result

def calculate_values(data, interval = 10):
    # Преобразуем данные в список кортежей (время, давление)
    data = [(datetime.datetime.strptime(time, "%H:%M:%S,%f"), int(pressure)) for time, pressure in data]

    # Определяем текущий интервал
    current_interval = data[0][0].minute // interval

    result = []
    count, summ = 0, 0
    for time, pressure in data:
        interval = time.minute // interval
        if interval != current_interval:
            result.append([current_interval, summ / count])
            # Переходим к следующему интервалу
            current_interval = interval
            summ = 0
            count = 0
        # Добавляем давление к сумме и увеличиваем счетчик измерений
        summ += pressure
        count += 1
    result.append([current_interval, summ / count])

    times, pressures = zip(*result)

    return times, pressures




if __name__ == "__main__":
    data1 = []
    data2 = []

    if not(os.path.exists('n_log1.json') and os.path.exists('n_log2.json')):
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
    data1 = [(datetime.datetime.strptime(time, "%H:%M:%S,%f"), int(pressure)) for time, pressure in data1]

    # Фильтруем данные по времени (при условии, что данные отсортированы по времени)
    start_time = datetime.datetime.strptime("14:10:00", "%H:%M:%S")
    end_time = datetime.datetime.strptime("14:15:00", "%H:%M:%S")
    filtered_data = [item for item in data1 if start_time <= item[0] <= end_time]

    # Извлекаем время и давление в отдельные списки
    times, pressures = zip(*filtered_data)

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)  # Две строки, два столбца. Выберем первую ячейку.
    ax2 = fig.add_subplot(2, 2, 2)  # Две строки, два столбца. Выберем вторую ячейку.
    ax3 = fig.add_subplot(2, 2, 3)  # Две строки, два столбца. Выберем третью ячейку.

    # Строим график
    ax1.plot(times, pressures)
    ax1.set_xlabel("Время")
    ax1.set_ylabel("Давление")
    ax1.set_title("График 1. Давление 14:10 - 14:15 из n_log1.txt")
    ax1.grid(True)



    # Преобразуем данные в список кортежей (время, давление)
    data2 = [(datetime.datetime.strptime(time, "%H:%M:%S,%f"), int(pressure)) for time, pressure in data2]

    # Определяем текущий интервал
    current_interval = data2[0][0].minute // 10

    result = []
    count, summ = 0, 0
    for time, pressure in data2:
        interval = time.minute // 10
        if interval != current_interval:
            result.append([current_interval, summ / count])
            # Переходим к следующему интервалу
            current_interval = interval
            summ = 0
            count = 0
        # Добавляем давление к сумме и увеличиваем счетчик измерений
        summ += pressure
        count += 1
    result.append([current_interval, summ / count])

    times, pressures = zip(*result)



    # Строим график
    ax2.plot(times, pressures)
    ax2.set_xlabel('Номер 10-минутного интервала')
    ax2.set_ylabel('Среднее давление')
    ax2.set_title('Давление по 10 мин из n-log2.txt')
    ax2.grid(True)




    # Определяем текущий интервал
    current_interval = data1[0][0].minute // 20

    result = []
    count, summ = 0, 0
    for time, pressure in data1:
        interval = time.minute // 20
        if interval != current_interval:
            result.append([current_interval, summ / count])
            # Переходим к следующему интервалу
            current_interval = interval
            summ = 0
            count = 0
        # Добавляем давление к сумме и увеличиваем счетчик измерений
        summ += pressure
        count += 1
    result.append([current_interval, summ / count])

    times_1, pressures_1 = zip(*result)


    # Определяем текущий интервал
    current_interval = data2[0][0].minute // 20

    result = []
    count, summ = 0, 0
    for time, pressure in data2:
        interval = time.minute // 20
        if interval != current_interval:
            result.append([current_interval, summ / count])
            # Переходим к следующему интервалу
            current_interval = interval
            summ = 0
            count = 0
        # Добавляем давление к сумме и увеличиваем счетчик измерений
        summ += pressure
        count += 1
    result.append([current_interval, summ / count])

    times_2, pressures_2 = zip(*result)

    # Строим график
    ax3.plot(times_1, pressures_1, label='data1')
    ax3.plot(times_2, pressures_2, label='data2')
    ax3.set_xlabel('Номер 10-минутного интервала')
    ax3.set_ylabel('Среднее давление')
    ax3.set_title('Давление по 10 мин из n-log2.txt')
    ax3.grid(True)





    plt.show()