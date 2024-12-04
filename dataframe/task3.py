import pandas as pd


def sellers(seller_n):
    # Загружаем данные из файла tranzaktions.csv в DataFrame
    df = pd.read_csv('tranzaktions.csv', delimiter='\t')

    # Фильтруем строки, относящиеся к данному продавцу
    seller_data = df[df['Продавец'] == seller_n]

    # Находим сумму и среднее значение выручки
    sum_n = seller_data['Цена (млн)'].sum()
    avg_n = seller_data['Цена (млн)'].mean()

    # Находим среднее значение выручки для автомобилей с ценой >= 2 миллиона
    avg2_n = seller_data[seller_data['Цена (млн)'] >= 2]['Цена (млн)'].mean()

    print(f"{sum_n}     {avg_n}     {avg2_n}")
    # Возвращаем сумму значений sum_n, avg_n и avg2_n
    return sum_n + round(avg_n) + round(avg2_n)


if __name__ == "__main__":
    # Вызываем функцию для n=2 и сохраняем результат в переменную sel_n
    sel_n = sellers("seller_2")

    # Выводим результат
    print(sel_n)
