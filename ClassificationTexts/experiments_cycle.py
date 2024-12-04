import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras import utils as keras_utils
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
import os

# Фиксируем сиды
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
tf.keras.utils.set_random_seed(random_seed)

# Параметры
filename = 'эксперименты_итератор/'
iterator = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
cycle = []

output_dim = 128
count_neurons = 64
dropout = 0.2
recurrent_dropout = 0.2

activation_function_result = 'sigmoid'
optimizer = RMSprop(learning_rate=0.01)
loss_function = 'categorical_crossentropy'
list_metrics = ['accuracy']

batch_size = 32
epochs = 140

# Создание папки для сохранения результатов
os.makedirs(filename, exist_ok=True)


def plot_confusion_matrix(cm, classes, filename, f_size=16, normalize=False, title='Матрица ошибок', cmap=plt.cm.Blues):
    """
    Построение и сохранение матрицы ошибок
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

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=f_size)

    plt.tight_layout()
    plt.ylabel('Действительный класс', fontsize=f_size + 1)
    plt.xlabel('Предсказанный класс', fontsize=f_size + 1)
    plt.savefig(filename + '_matrix.png')


def train_and_evaluate_model(iter, X_train, y_train, X_valid, y_valid, X_test, y_test, df, filename):
    global output_dim, count_neurons, dropout, recurrent_dropout, activation_function_result
    global optimizer, loss_function, list_metrics, batch_size, epochs, cycle

    # Создание модели
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=maxSequenceLength))
    model.add(GRU(count_neurons, dropout=iter, recurrent_dropout=recurrent_dropout))
    model.add(Dense(num_classes, activation=activation_function_result))

    model.compile(optimizer=optimizer, loss=loss_function, metrics=list_metrics)
    print(model.summary())

    # Обучение модели
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid))

    # Оценка модели
    predictions = model.predict(X_test).argmax(axis=1)
    y_true = np.argmax(y_test, axis=1)
    report = classification_report(y_true, predictions)
    print(report)

    macro_f1 = f1_score(y_true, predictions, average='macro')
    print(f'Macro F1 Score: {macro_f1}')
    cycle.append([iter, macro_f1])

    # Логирование
    with open(filename + '_log.txt', 'a', encoding='utf-8') as f:
        f.write("Model Parameters:\n")
        f.write(f"Input Dim: {vocab_size}\n")
        f.write(f"Output Dim: {output_dim}\n")
        f.write(f"Neurons: {count_neurons}\n")
        f.write(f"Dropout: {dropout}\n")
        f.write(f"Recurrent Dropout: {recurrent_dropout}\n")
        f.write(f"Activation Function: {activation_function_result}\n")
        f.write(f"Loss Function: {loss_function}\n")
        f.write(f"Optimizer: {str(optimizer)}\n")
        f.write(f"Metrics: {list_metrics}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write(f"\nMacro F1 Score: {macro_f1}\n")

    # Построение матрицы ошибок
    conf_matr = confusion_matrix(y_true, predictions)
    plot_confusion_matrix(conf_matr, classes=df['class'].astype(str).unique(), f_size=16, filename=filename + f"_{iter}")


if __name__ == "__main__":
    # Чтение данных
    df = pd.read_excel("df_prep.xlsx")  # Укажите путь к вашему датасету

    # Разделение данных на тренировочную, валидационную и тестовую выборки
    X_vt_1, X_test_1, y_vt_1, y_test_1 = train_test_split(
        df['preprocessed_text'], df['class'], random_state=42, test_size=0.1, stratify=df['class']
    )
    X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(
        X_vt_1, y_vt_1, test_size=0.2, random_state=42, stratify=y_vt_1
    )

    # Подсчет максимальной длины текста и создание токенизатора
    max_words = max(len(desc.split()) for desc in df['preprocessed_text'].tolist())
    print(f'Максимальное количество слов в самом длинном письме: {max_words} слов')

    vocab_size = 10000
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(df['preprocessed_text'])

    maxSequenceLength = max_words
    X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train_1), maxlen=maxSequenceLength)
    X_valid = sequence.pad_sequences(tokenizer.texts_to_sequences(X_valid_1), maxlen=maxSequenceLength)
    X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test_1), maxlen=maxSequenceLength)

    # Преобразование меток в one-hot векторы
    num_classes = df['class'].nunique()
    y_train = keras_utils.to_categorical(y_train_1, num_classes)
    y_valid = keras_utils.to_categorical(y_valid_1, num_classes)
    y_test = keras_utils.to_categorical(y_test_1, num_classes)

    # Запуск обучения с разными параметрами dropout
    for iter in iterator:
        train_and_evaluate_model(iter, X_train, y_train, X_valid, y_valid, X_test, y_test, df, filename)

    print("Результаты экспериментов:", cycle)
