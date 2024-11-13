import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Загрузка данных и выбор первых 50 000 записей
    df = pd.read_csv("creditcard.csv")
    df = df.head(50000)

    # Подготовка данных: разделяем на признаки и целевую переменную
    y = df['Class']
    X = df.drop(columns=['Class'])

    # # Масштабирование признаков для лучшей сходимости
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # Отсутствие масштабирования
    X_scaled = X

    # Разделение на тренировочную и тестовую выборки с помощью stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )
    # Определяем сетку параметров для случайного леса
    # param_grid = {
    #     # Количество деревьев решений в случайном лесу
    #     'n_estimators': [50, 100, 200],
    #     # Максимальная глубина каждого дерева
    #     'max_depth': [None, 10, 20, 30],
    #     # Минимальное количество объектов в узле, необходимое для его дальнейшего разбиения
    #     'min_samples_split': [2, 5, 10],
    #     # Минимальное количество объектов, которое должно содержаться в каждом листе (конечном узле дерева)
    #     'min_samples_leaf': [1, 2, 4]
    # }
    param_grid = {
        # Количество деревьев решений в случайном лесу
        'n_estimators': [10, 20],
        # Максимальная глубина каждого дерева
        'max_depth': [2, 3],
        # Минимальное количество объектов в узле, необходимое для его дальнейшего разбиения
        'min_samples_split': [2, 5],
        # Минимальное количество объектов, которое должно содержаться в каждом листе (конечном узле дерева)
        'min_samples_leaf': [1, 2]
    }

    # Инициализация случайного леса и выполнение GridSearchCV для поиска оптимальных параметров
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    '''
        estimator - Метод, параметры которого настраиваем
        param_grid - Словарь партеров со значениями для перебора
        scoring - Стратегия оценки эффективности модели с перекрестной проверкой на тестовом наборе
        cv - количество фолдов кросс-валидации
        n_jobs - количество параллельных процессов вычислений (доступных ядер)
        verbose - целое число, количество выводимой информации при обучении
    '''
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='f1_macro', cv=5, n_jobs=-1, verbose=3)
    grid_search.fit(X_train, y_train)

    # Лучшая модель и параметры
    best_rf_model = grid_search.best_estimator_
    print("Лучшие параметры для случайного леса:", grid_search.best_params_)

    # Оценка производительности случайного леса на тестовой выборке
    y_predict_rf = best_rf_model.predict(X_test)
    rf_macro_f1 = f1_score(y_test, y_predict_rf, average='macro')
    print(f"Macro F1 для лучшего случайного леса на тестовой выборке: {rf_macro_f1:.4f}")

    # Вывод отчета классификации
    print("Отчет классификации для лучшего случайного леса:")
    print(classification_report(y_test, y_predict_rf))

    # Построение матрицы ошибок
    classes = [0, 1]  # Классы транзакций
    cm = confusion_matrix(y_test, y_predict_rf)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues')
    plt.title("Матрица ошибок для лучшего случайного леса")
    plt.show()
