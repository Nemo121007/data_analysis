import pandas as pd
import numpy as np


def main():
    # Прочитаем таблицу CreditCardTransaction.csv и сохраним в DataFrame с именем df
    df = pd.read_csv('CreditCardTransaction.csv')

    # Установим фиксированное значение seed для воспроизводимости результатов
    np.random.seed(12)

    # Получим случайную выборку из 10000 строк
    dfs = df.sample(n=10000, random_state=12)

    # Найдем количество раз, которое встретился каждый департамент (значение столбца Department)
    department_counts = dfs['Department'].value_counts()
    print(department_counts)

    # Выведем 3 департамента, которые наиболее часто встречаются
    top_departments = department_counts.nlargest(3)
    print("Три наиболее частых департамента:\n", top_departments)

    # Получим размер сумм транзакций (поле TrnxAmount) за январь или февраль 2022 года
    filtered_transactions = dfs[(dfs['Year'] == 2022) & (dfs['Month'].isin([1, 2]))]['TrnxAmount']

    # Найдем медиану сумм транзакций из полученных значений
    median_transaction = filtered_transactions.median()
    print("Медиана сумм транзакций за январь и февраль 2022 года:", median_transaction)

    # Добавим в dfs столбец с разностью значений модуля найденной медианы и модуля значения из поля TrnxAmount
    dfs['TrnxAmount_Diff'] = abs((median_transaction) - (dfs['TrnxAmount']))

    # Выведем первые несколько строк для проверки
    print(dfs.head())


if __name__ == "__main__":
    main()
