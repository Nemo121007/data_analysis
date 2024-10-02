def function(string):
    """Функция обрабатывает строку, отбирает гласные, удваивает их и возвращает объединенную строку."""
    vowels = "аеёиоуыэюя"

    # Отбираем все гласные буквы из строки с помощью функции filter()
    list_1 = list(filter(lambda char: char in vowels, string))

    # Удваиваем каждую букву из list_1 с помощью функции map()
    list_2 = list(map(lambda char: char * 2, list_1))

    # Создаем список кортежей из элементов list_1 и list_2 с помощью функции zip()
    list_3 = list(zip(list_1, list_2))

    # Конкатенируем все элементы списка list_3 и сохраняем в строку string_out
    string_out = ''.join([char1 + char2 for char1, char2 in list_3])

    return string_out

if __name__ == "__main__":
    # Пример использования
    print(function("Каталог"))
    print(function("Карл у Клары украл кораллы"))