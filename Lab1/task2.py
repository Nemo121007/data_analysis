def work(n):
    """Функция выполняет операции с множествами и возвращает сумму sum_1 + sum_2 + sum_3."""
    if n <= 0 or n % 2 != 0:
        raise ValueError("n должно быть положительным числом, кратным 2")

    # Создаем множество set_1 = {2, 4, 6, ..., n}
    set_1 = set(range(2, n + 1, 2))

    # Создаем множество set_2 = {7, 14, 21, ..., 98}
    set_2 = set(range(7, 99, 7))

    # Найдем сумму элементов объединения множеств set_1 и set_2
    sum_1 = sum(set_1 | set_2)

    # Найдем сумму элементов пересечения множеств set_1 и set_2
    sum_2 = sum(set_1 & set_2)

    # Найдем сумму элементов разности множеств set_1 и set_2
    sum_3 = sum(set_1 - set_2)

    # Возвращаем сумму sum_1 + sum_2 + sum_3
    return sum_1 + sum_2 + sum_3

if __name__ == "__main__":
    # Пример использования
    print(work(20))
    print(work(40))