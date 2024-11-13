import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    # Загрузка данных и выбор первых 50 000 записей
    df = pd.read_csv("creditcard.csv")
    df = df.head(50000)

    # Подготовка данных: разделяем на признаки и целевую переменную
    y = df['Class']
    X = df.drop(columns=['Class'])

    # # Масштабирование признаков для лучшей сходимости
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Отсутствие масштабирования
    # X_scaled = X

    # Разделение на тренировочную и тестовую выборки с помощью stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )

    # Инициализация моделей
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Bagging (Decision Tree)": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=10, random_state=42
            ),
        "Random Forest": RandomForestClassifier(n_estimators=10, random_state=42)
    }

    # Обучение моделей и оценка метрики macro_f1
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        macro_f1 = f1_score(y_test, y_predict, average='macro')
        results[model_name] = macro_f1

    # Вывод результатов
    print("\nСравнение моделей по метрике macro F1:")
    for model_name, macro_f1 in results.items():
        print(f"{model_name}: {macro_f1:.4f}")
