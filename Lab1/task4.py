import sys

def check_generator():
    # Однострочный генератор списка mas_gen
    mas_gen = [i for i in range(1000000) if i % 7 == 0]

    # Получаем объем памяти, занимаемый списком mas_gen
    mem1 = sys.getsizeof(mas_gen)

    # Генератор на основе yield, реализующий ту же последовательность
    def gen():
        for i in range(1000000):
            if i % 7 == 0:
                yield i

    # Создаем объект генератора
    generator = gen()

    # Поочередно генерируем два элемента
    first_elem = next(generator)
    second_elem = next(generator)

    # Получаем объем памяти, занимаемый вторым сгенерированным элементом
    mem2 = sys.getsizeof(second_elem)

    # Возвращаем разность mem1 и mem2
    return mem1 - mem2

if __name__ == "__main__":
    # Пример использования
    print(check_generator())
