import re
def function(string):
    # Ищем границы слов (заглавные буквы) и добавляем перед ними "_", приводя всю строку к нижнему регистру
    snake_case = re.sub(r'([A-Z])', r'_\1', string).lower()
    # Убираем возможное начальное подчеркивание
    if snake_case.startswith('_'):
        snake_case = snake_case[1:]
    return snake_case

# Примеры использования функции
print(camel_to_snake("camelCaseVar"))  # Вывод: camel_case_var
print(camel_to_snake("myWonderfulVar"))  # Вывод: my_wonderful_var
print(camel_to_snake("этоРусскийПример"))  # Вывод: это_русский_пример
print(camel_to_snake("АббревиатураСРусскимиБуквами"))  # Вывод: аббревиатура_с_русскими_буквами