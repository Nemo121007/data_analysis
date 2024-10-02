import re
from datetime import datetime, timedelta

# Открываем исходный файл для чтения и выходной файл для записи
with open('n_log1.txt', 'r') as infile, open('out1.txt', 'w') as outfile:
    time_list = []

    # Проходим по каждой строке в исходном файле
    for line in infile:
        # Проверяем наличие подстрок "KEEP" и "capacity_type=1"
        if 'KEEP' in line and 'capacity_type=1' in line:
            # Извлекаем время в формате ЧЧ:ММ:СС, используя регулярное выражение
            time_match = re.search(r'\d{2}:\d{2}:\d{2}', line)
            if time_match:
                # Преобразуем строку времени в объект datetime
                time_str = time_match.group(0)
                time_obj = datetime.strptime(time_str, '%H:%M:%S')

                # Записываем строку в файл out1.txt
                outfile.write(line)

                # Добавляем время в список для последующего вычисления интервалов
                time_list.append(time_obj)

# Вычисляем средний интервал времени (avg_delta_t) между строками
time_deltas = [(time_list[i] - time_list[i - 1]) for i in range(1, len(time_list))]

# Если есть хотя бы один интервал
if time_deltas:
    avg_delta_t = sum(time_deltas, timedelta()) / len(time_deltas)

    # Находим целое количество секунд в avg_delta_t
    whole_number = int(avg_delta_t.total_seconds())

    # Выводим результат
    print(whole_number)
else:
    print('Не найдено достаточно строк для вычисления интервала.')
