import re

def function(string):
    """
    Подсчитывает количество аббревиатур в строке.

    Args:
        string: Исходная строка.

    Returns:
        Количество найденных аббревиатур.
    """

    # Регулярное выражение для поиска аббревиатур
    pattern = r"[А-ЯЁ]{2,}"

    # Поиск всех совпадений с шаблоном в строке
    matches = re.findall(pattern, string)

    # Возврат количества найденных совпадений
    return len(matches)

# Примеры использования функции
print(function(" А курс информатики в вузе соответствует ФГОС и ПООП, что подтверждено ФГУ"))  # Вывод: 3
print(function(" СССР и США"))  # Вывод: 2