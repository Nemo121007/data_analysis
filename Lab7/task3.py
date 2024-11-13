import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Загрузка и подготовка данных
    df = pd.read_csv("creditcard.csv").head(50000)

    y = df['Class']
    X = df.drop(columns=['Class'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Создание сетки для матриц ошибок
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # # Загружаем данные на GPU
    # dtrain = xgb.DMatrix(X_train, label=y_train, device='cuda')
    # dtest = xgb.DMatrix(X_test, device='cuda')

    # LightGBM-классификатор
    lgb_model = lgb.LGBMClassifier(
        random_state=42,
        n_estimators=100,  # Количество иттераций бустинга (количество моделей)
        max_depth=-1,
        boosting_type='gbdt',  # Дерево решений с градиентным бустингом
        device='gpu',  # Использование GPU
        n_jobs=-1  # Использование всех доступных ядер процессора
    )
    lgb_model.fit(X_train, y_train)
    y_predict_lgb = lgb_model.predict(X_test)

    # Оценка LightGBM
    print("LightGBM - Отчет классификации")
    print(classification_report(y_test, y_predict_lgb))
    lgb_f1_macro = f1_score(y_test, y_predict_lgb, average='macro')
    print(f"Macro F1 для LightGBM: {lgb_f1_macro:.4f}")

    # Матрица ошибок для LightGBM
    cm_lgb = confusion_matrix(y_test, y_predict_lgb)
    disp_lgb = ConfusionMatrixDisplay(confusion_matrix=cm_lgb)
    disp_lgb.plot(cmap='Blues', ax=ax1)
    ax1.set_title("Матрица ошибок для LightGBM")

    # Стекинг-классификатор
    rf = RandomForestClassifier(
        n_estimators=10,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    # # Создание и обучение модели с использованием GPU
    # xgb_model = xgb.XGBClassifier(
    #     tree_method='hist',  # Используем метод "hist"
    #     device='cuda',  # Указываем, что используем GPU
    #     eval_metric='logloss',
    #     random_state=42
    # )
    xgb_model = xgboost.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    additional_model = CatBoostClassifier(random_state=42,
                                          task_type="GPU",
                                          learning_rate=1,
                                          verbose=0,
                                          depth=2,
                                          n_estimators=100)
    stacking_model = StackingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgboost.XGBClassifier()),
            ('gb', additional_model)
        ],
        final_estimator=RandomForestClassifier(
            n_estimators=10,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
            )
    )

    stacking_model.fit(X_train, y_train)
    y_predict_stacking = stacking_model.predict(X_test)

    # Оценка стекинг-модели
    print("Стекинг - Отчет классификации")
    print(classification_report(y_test, y_predict_stacking))
    stacking_f1_macro = f1_score(y_test, y_predict_stacking, average='macro')
    print(f"Macro F1 для стекинга: {stacking_f1_macro:.4f}")

    # Матрица ошибок для стекинга
    cm_stacking = confusion_matrix(y_test, y_predict_stacking)
    disp_stacking = ConfusionMatrixDisplay(confusion_matrix=cm_stacking)
    disp_stacking.plot(cmap='Blues', ax=ax2)
    ax2.set_title("Матрица ошибок для стекинга")

    plt.show()
