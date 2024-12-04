Скачать набор данных creditcard.csv с https://www.kaggle.com/mlg-ulb/creditcardfraud, 
в котором: Time – время транзакции, V1-V28 – признаки, Amount – количество, 
Class – класс транзакции: мошенническая (Class=1) и легальная (Class=0).

Используйте из набора данных только первые 50 000 записей.

Используя train_test_split(…, test_size=0.3, stratify=df.Class) разбейте 
выборку (c_card.csv) на тренировочное и тестовое множества так, чтобы тестовая выборка 
составляла 30% от всей выборки и в тестовой выборке пропорция мошеннических и легальных 
транзакций была такой же как и в основной выборке (stratify).

При решении задачи классификации транзакций на мошеннические и легальные необходимо 
максимизировать метрику macro f1 на тестовой выборке.

Задания:

Задание 1. Сравнить значения метрики macro_f1 при тестировании методов: 
дерево принятия решений, бэггинга деревьев решений и случайного леса.

Задание 2. Оптимизировать/настроить параметры (GridSearchCV) случайного леса для 
максимизации значения f1_macro (scoring='f1_macro') на тестовой выборке. Для метода, 
обеспечившего максимум f1_macro, построить матрицу ошибок.

Задание 3. Примененеие ансамблей моделей. Обучить и для обученных моделей получить 
(вывести на экран) матрицу метрик и матрицу ошибок на тестовой выборке, сравнить 
значение метрики f1_macro. Модели:

а) LightGBM-классификтор,

б) стекинг с не менее тремя моделями ML:
    1-я модель) Случайный лес с полученными в задаче 2 значениями,
    2-я модель) xgboost,
    3-я модель) выбрать самостоятельно.