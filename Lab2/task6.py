import os
import zipfile

def process_folder(folder_path):
    """
    Рекурсивно обрабатывает папку и все ее подпапки.

    Args:
        folder_path: Путь к обрабатываемой папке.

    Returns:
        Количество файлов в папке и ее подпапках.
    """

    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                os.remove(os.path.join(root, file))
            count += 1
    return count

# Подсчет файлов до удаления
folder_path = 'folder_main'  # Замените на ваш путь
n_file_before = process_folder(folder_path)

# Подсчет файлов после удаления
n_file_after = process_folder(folder_path)

# Вычисление суммы
sum_file = n_file_before + n_file_after

# Вывод результата
print("Суммарное количество файлов до и после удаления .txt: ", sum_file)