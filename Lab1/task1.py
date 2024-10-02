def is_prime(num):
    """Функция для проверки, является ли число простым."""
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

def my_list(n):
    """Функция для выполнения задач, описанных в задании."""
    if not (0 < n < 100):
        raise ValueError("n должно быть положительным числом меньше 100")

    # Создаем список простых чисел, которые больше n и меньше 400
    primes = [i for i in range(n + 1, 400) if is_prime(i)]

    # Создаем второй список, срез элементов первого списка с 24-го по 29-й индекс
    sliced_primes = primes[24:30]

    # Удаляем элемент с индексом 1 во втором списке
    if len(sliced_primes) > 1:
        del sliced_primes[1]

    # Находим среднее арифметическое элементов второго списка и округляем его
    if sliced_primes:
        average = round(sum(sliced_primes) / len(sliced_primes))
    else:
        average = 0

    return average

if __name__ == "__main__":
    # Пример использования
    print(my_list(99))
    print(my_list(13))
