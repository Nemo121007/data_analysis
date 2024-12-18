1. task 1) Создайте функцию series_data(name,index), в которой сгенерируйте объект s 
типа Series, содержащий все целые положительные нечетные числа меньшие 10000 и кратные 15. 
Озаглавьте серию как name. Индексы s должны начинаться с 1 (1, 2, 3,...). 
Преобразуйте числа, начиная с числа с индексом index (7 < index < 40) и до конца списка 
следующим образом:

если индекс нечетный (1, 3…), то число умножается на 2,
если индекс четный (2, 4…), то число уменьшается на 70.

Функция series_data(name, index) с помощью return возвращает  наибольшее из чисел с индексами 
index, index+1, index-5.

Тест	Результат

print(series_data('Название серии', 8)) 510


print(series_data('Название серии', 18)) 1110

2. task2) Создайте функцию frame_data(list1, list2, indices), где  list1, list2 – 
двумерные списки одинакового размера n x n (n = 1, 2, 3, …), indices – список из 
односимвольных значений длиной n. Из списков list1 и list2 создайте объекты класса 
DataFrame df1 и df2 соответственно. В df1 и df2 задайте индексацию indices. 
Отсортируйте строки df1 по убыванию индекса. В df2 отсортируйте строки по увеличению 
их суммы (для этого создайте в df2 столбец с суммами элементов строк и используйте 
этот столбец в качестве индекса). В каждом из df1 и df2 получить элемент, находящийся 
в первой по порядку (нулевой) строке и левом столбце, эти два элемента сложить. Функция 
frame_data(list1, list2, indices) с помощью return должна возвращать полученную сумму.

Для примера:

Тест	Результат

print(frame_data([[4,5],[7,8]], [[20,30],[4,5]], ['C','D'])) 11

print(frame_data([[1,2,3],[4,5,6],[7,8,9]], [[100,20,30],[4,5,60],[7,80,9]], 
['C','A', 'B'])) 5

3. task3) В файле tranzaktions.csv приведена история продаж автомобилей тремя 
продавцами seller_1, seller_2, seller_3 автомагазина. Написать функцию sellers(n), 
в которую передается номер продавца n (1<= n<=3). В функции, используя методы модуля 
pandas:
- Преобразовать файл tranzaktions.csv в DataFrame.
- Используя агрегацию найти и сохранить в переменные sum_n и avg_n соответственно 
сумму и среднее значение вырученных средств продавца с номером n (seller_n). 
Среднее значение avg_n, используя функцию round(), округлить до целого числа.
- Используя агрегацию найти и сохранить в переменную avg2_n  среднее значение 
вырученных средств продавца с номером n (seller_n) при продаже автомобилей, цена 
которых >= 2 миллиона. Среднее значение avg2_n, используя функцию round(), 
округлить до целого числа.
- Функция sellers(n) с помощью return должна возвращать сумму значений sum_n, 
avg_n и avg2_n.
- Найти значение sel_n, которое возвращает sellers(n) при n=2.


4. task4) Зашумленные транзакционные данные представлены в списке 
transaction = [120, -31, ’20.1’, 12.3, ‘bank’, 12, -4, -7, 150, ‘mr.’, 23, 32, 21]. 
Необходимо:

Создать объект pd.Series, значения которого совпадают со значениями 
transaction, а индексы – целые числа >= 10 и < 23. 
Сохранить созданный объект в переменную t.
Из t получить только целые числа элементы (*) и вычислите их несмещенную 
выборочную дисперсию, среднее значение.


5. task5) 
- Используя функцию random.normal() из модуля numpy, cгенерируйте 
200 значений нормально распределённой случайной величины (*). Результат 
генерации сохраните как pd.Series в переменную s.
- Каждое значение s возведите во 2 степень, индексы элементов s увеличьте на 2.
- Выведите количество значений s, которые больше 2.
- Выведите сумму элементов, строго меньших 2 и имеют нечётные индексы.


6. task6) Скачать набор данных (таблицу) CreditCardTransaction.csv 
с https://www.kaggle.com/datasets/ybifoundation/credit-card-transaction. 
Прочитать и сохранить эту таблицу в pd.DataFrame с именем df.

Задав np.random.seed(12) и используя методsample(), из df получите выборку, 
содержащую 10000 строк. Выборку сохраните в переменную dfs.

Получите количество раз, которое встретился каждый департамент (значение столбца 
Department) в dfs. Выведите3 департамента, которые наиболее часто встречаются в dfs.

В dfs получить размер сумм транзакций (поле TrnxAmount) соответствующие январю 
или февралю 2022 года (поля Year и Month) и:
- Найдите медиану сумм транзакций полученных значений.
- Добавьте в dfs столбец с разностью значений модуля найденной медианы и модуля значения из поля TrnxAmount.