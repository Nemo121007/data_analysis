import re

def function(name):
    """
    Преобразует имя переменной из CamelCase в snake_case с поддержкой русских букв.

    Args:
        name: Имя переменной в формате CamelCase.

    Returns:
        Имя переменной в формате snake_case.
    """

    # Ищем заглавные буквы, не стоящие в начале строки, включая русские
    s1 = re.sub('(.)([A-ZА-Я][a-zа-я]+)', r'\1_\2', name)
    # Разделяем слова, начинающиеся с заглавной буквы, включая русские
    return re.sub('([a-zа-я0-9])([A-ZА-Я])', r'\1_\2', s1).lower()

# Примеры использования функции
print(function("camelCaseVar"))  # Вывод: camel_case_var
print(function("myWonderfulVar"))  # Вывод: my_wonderful_var
print(function("этоРусскийПример"))  # Вывод: это_русский_пример
print(function("АббревиатураСРусскимиБуквами"))  # Вывод: аббревиатура_с_русскими_буквами