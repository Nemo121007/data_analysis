import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, GRU
from tensorflow.keras import utils as keras_utils
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
# Фиксируем сиды
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
tf.keras.utils.set_random_seed(42)

filename = 'эксперименты с эпохами/'
# count_epoch = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
count_epoch = [50, 100, 150, 200, 300, 500, 800, 1000]
f1_epoch = []

def plot_confusion_matrix(cm, classes, filename, f_size=16, normalize=False, title='Матрица ошибок', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'
    """
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
    else:
        pass

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=f_size)
    plt.tight_layout()
    plt.ylabel('Действительный класс', fontsize=f_size + 1)
    plt.xlabel('Предсказанный класс', fontsize=f_size + 1)

    # Сохранение матрицы ошибок как изображение
    plt.savefig(filename + 'pict.png')


def train_and_evaluate_model(epochs, X_train, y_train, X_valid, y_valid, X_test, y_test, df, filename):
    # Построение модели LSTM для классификации текста
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=maxSequenceLength, input_length=maxSequenceLength))
    model.add(GRU(32, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    print(model.summary())

    # Параметры обучения
    batch_size = 32

    # Обучение модели
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid))

    # Оценка модели на тестовых данных и отчет о качестве
    predictions = model.predict(X_test).argmax(axis=1)
    y_true = np.array(y_test_1)
    report = classification_report(y_true, predictions)
    print(f"Результаты для {epochs} эпох:")
    print(report)
    macro_f1 = f1_score(y_true, predictions, average='macro')
    print(f'Macro F1 Score: {macro_f1}')

    global f1_epoch
    f1_epoch.append([int(epochs), float(macro_f1)])

    # Сохранение в log.txt
    with open(filename + f'{epochs}_log.txt', 'a', encoding='utf-8') as f:
        f.write(f"Model Parameters for {epochs} epochs:\n")
        f.write(f"Input Dim: {vocab_size}\n")
        f.write(f"Output Dim: {maxSequenceLength}\n")
        f.write("LSTM Layer:\n")
        f.write("Layer: LSTM\n")
        f.write("Neurons: 32\n")
        f.write("Dropout: 0.3\n")
        f.write("Recurrent Dropout: 0.3\n")
        f.write("Activation Function: Sigmoid\n")
        f.write(f"Loss Function: categorical_crossentropy\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Metrics: Accuracy\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write(f"\nMacro F1 Score: {macro_f1}")

    # Матрица путаницы
    conf_matr = confusion_matrix(y_true, predictions)
    plot_confusion_matrix(conf_matr, classes=df['class'].astype(str).unique(), f_size=16, filename=filename + f"_{epochs}")


if __name__ == "__main__":
    # Чтение данных
    df = pd.read_excel("df_prep.xlsx")  # Замените "data.csv" на путь к вашему датасету

    # Разделение данных на тренировочную, валидационную и тестовую выборки
    X_vt_1, X_test_1, y_vt_1, y_test_1 = train_test_split(df['preprocessed_text'], df['class'], random_state=42,
                                                            test_size=0.9, stratify=df['class'])
    X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_vt_1, y_vt_1, test_size=0.5, random_state=42)

    # Подсчет количества классов
    print(f'y_train_1: \n{y_train_1.value_counts()}')
    print(f'y_valid_1: \n{y_valid_1.value_counts()}')
    print(f'y_test_1: \n{y_test_1.value_counts()}')

    # Подсчет максимальной длины текста в словах и общего количества уникальных слов
    max_words = max(len(desc.split()) for desc in df['preprocessed_text'].tolist())
    print(f'Максимальное количество слов в самом длинном письме: {max_words} слов')

    # Создаем токенизатор и составляем словарь для преобразования текста в числовые последовательности
    vocab_size = 10000  # Ограничиваем словарь до 10,000 наиболее частых слов
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(df['preprocessed_text'])

    # Преобразуем текст в числовые последовательности и заполняем до одинаковой длины
    maxSequenceLength = max_words
    X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train_1), maxlen=maxSequenceLength)
    X_valid = sequence.pad_sequences(tokenizer.texts_to_sequences(X_valid_1), maxlen=maxSequenceLength)
    X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test_1), maxlen=maxSequenceLength)

    # Преобразуем целевые классы в категориальные (one-hot) векторы для использования с categorical_crossentropy
    num_classes = df['class'].nunique()
    y_train = keras_utils.to_categorical(y_train_1, num_classes)
    y_valid = keras_utils.to_categorical(y_valid_1, num_classes)
    y_test = keras_utils.to_categorical(y_test_1, num_classes)

    # Запуск циклического обучения для разных значений эпох
    for epochs in count_epoch:
        train_and_evaluate_model(epochs, X_train, y_train, X_valid, y_valid, X_test, y_test, df, filename)

    print(f1_epoch)