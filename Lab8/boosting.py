import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

# Фиксируем сиды
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
tf.keras.utils.set_random_seed(42)

def plot_confusion_matrix(cm, classes, filename, f_size=16, normalize=False, title='Матрица ошибок', cmap=plt.cm.Blues):
    plt.figure(figsize=(7, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=f_size + 2)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=f_size - 6)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=f_size)
    plt.yticks(tick_marks, classes, fontsize=f_size)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=f_size)
    plt.tight_layout()
    plt.ylabel('Действительный класс', fontsize=f_size + 1)
    plt.xlabel('Предсказанный класс', fontsize=f_size + 1)
    plt.savefig(filename + 'pict.png')



# Функция создания модели
def create_model(vocab_size, output_dim, max_sequence_length, num_classes, count_neurons=64, dropout=0.2,
                 recurrent_dropout=0.2):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=max_sequence_length),
        GRU(count_neurons, dropout=dropout, recurrent_dropout=recurrent_dropout),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Функция для обучения одной модели
def train_single_model(model, X_train, y_train, X_valid, y_valid, epochs=20, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_valid, y_valid),
                        verbose=0)
    return model, history


# Функция бустинга
def boosting_ensemble(X_train, y_train, X_valid, y_valid, X_test, y_test, vocab_size, output_dim, max_sequence_length,
                      num_classes, n_models=5):
    models = []
    alpha = []  # Веса для каждой модели
    y_train_boost = np.copy(y_train)

    for i in range(n_models):
        print(f"Тренируем модель {i + 1}/{n_models}")
        # Создаем и обучаем модель
        model = create_model(vocab_size, output_dim, max_sequence_length, num_classes)
        model, history = train_single_model(model, X_train, y_train_boost, X_valid, y_valid, epochs=50)

        # Получаем предсказания и вычисляем веса ошибки
        predictions = model.predict(X_train).argmax(axis=1)
        y_true = y_train.argmax(axis=1)
        errors = (predictions != y_true).astype(float)

        # Вычисляем взвешенную ошибку модели
        error_rate = np.sum(errors) / len(errors)
        if error_rate > 0.5:
            print(f"Пропускаем модель {i + 1} из-за высокой ошибки")
            continue

        # Вычисляем вес модели
        model_weight = np.log((1 - error_rate) / error_rate) + np.log(num_classes - 1)
        alpha.append(model_weight)
        models.append(model)

        # Обновляем веса обучающих данных
        sample_weights = np.exp(model_weight * errors)
        sample_weights /= np.sum(sample_weights)
        for j, w in enumerate(sample_weights):
            y_train_boost[j] = y_train[j] * w  # Масштабируем значения меток

    return models, alpha


# Функция ансамблирования предсказаний
def ensemble_predict(models, alpha, X):
    weighted_predictions = np.zeros((len(X), len(models[0].predict(X)[0])))
    for model, weight in zip(models, alpha):
        preds = model.predict(X)
        weighted_predictions += weight * preds
    return weighted_predictions.argmax(axis=1)


# Главный блок
if __name__ == "__main__":
    # Загрузка данных
    df = pd.read_excel("df_prep.xlsx")
    X_vt_1, X_test_1, y_vt_1, y_test_1 = train_test_split(df['preprocessed_text'], df['class'], random_state=42,
                                                          test_size=0.1, stratify=df['class'])
    X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_vt_1, y_vt_1, test_size=0.2, random_state=42)

    vocab_size = 10000
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(df['preprocessed_text'])

    max_sequence_length = max(len(desc.split()) for desc in df['preprocessed_text'])
    X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_1), maxlen=max_sequence_length)
    X_valid = pad_sequences(tokenizer.texts_to_sequences(X_valid_1), maxlen=max_sequence_length)
    X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_1), maxlen=max_sequence_length)

    num_classes = df['class'].nunique()
    y_train = to_categorical(y_train_1, num_classes)
    y_valid = to_categorical(y_valid_1, num_classes)
    y_test = to_categorical(y_test_1, num_classes)

    output_dim = 4 * max_sequence_length

    # Запуск бустинга
    n_models = 5
    models, alpha = boosting_ensemble(X_train, y_train, X_valid, y_valid, X_test, y_test, vocab_size, output_dim,
                                      max_sequence_length, num_classes, n_models)

    # Предсказания ансамбля
    y_pred = ensemble_predict(models, alpha, X_test)
    y_true = y_test.argmax(axis=1)

    # Метрики и отчеты
    report = classification_report(y_true, y_pred)
    print(report)

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f'Macro F1 Score: {macro_f1}')

    with open('bossting_log.txt', 'w', encoding='utf-8') as f:
        f.write("Model Parameters:\n")
        f.write(f"Input Dim: {vocab_size}\n")
        f.write(f"Output Dim: {output_dim}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write(f"\nMacro F1 Score: {macro_f1}")

    conf_matr = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(conf_matr, classes=df['class'].astype(str).unique(), f_size=16, filename='bossting_log')
