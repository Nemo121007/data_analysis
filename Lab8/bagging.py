import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras import utils as keras_utils
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.optimizers import RMSprop

filename = 'bagging_model_'
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
tf.keras.utils.set_random_seed(42)


# Функция для построения матрицы ошибок
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


# Беггинг: обучение нескольких моделей
def train_bagging_model(X_train, y_train, X_valid, y_valid, X_test, y_test, n_models=5):
    models = []  # Список для моделей
    histories = []  # Истории обучения моделей

    # Обучаем n моделей на случайных подмножествах данных
    for i in range(n_models):
        print(f'Обучение модели {i + 1}/{n_models}...')

        # Бутстрэпинг: случайный отбор с повторениями
        X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.2,
                                                                random_state=random_seed)

        # Клонируем модель
        model = create_model(X_train.shape[1], num_classes=y_train.shape[1])
        model.set_weights(clone_model(model).get_weights())
        model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        # Обучаем модель
        history = model.fit(X_train_subset, y_train_subset, batch_size=32, epochs=70,
                            validation_data=(X_valid, y_valid))

        models.append(model)
        histories.append(history)

    # Возвращаем список обученных моделей и истории обучения
    return models, histories


# Функция для предсказания с использованием ансамбля
def predict_with_bagging(models, X_test):
    # Собираем предсказания от всех моделей
    predictions = np.zeros((len(models), X_test.shape[0], models[0].output_shape[1]))

    for i, model in enumerate(models):
        predictions[i] = model.predict(X_test)

    # Усредняем предсказания
    final_predictions = np.mean(predictions, axis=0).argmax(axis=1)
    return final_predictions


# Функция для создания модели
def create_model(input_length, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=4 * input_length, input_length=input_length))
    model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Главная часть программы
if __name__ == "__main__":
    df = pd.read_excel("df_prep.xlsx")

    X_vt_1, X_test_1, y_vt_1, y_test_1 = train_test_split(df['preprocessed_text'], df['class'], random_state=42,
                                                          test_size=0.1, stratify=df['class'])
    X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_vt_1, y_vt_1, test_size=0.2, random_state=42)

    max_words = max(len(desc.split()) for desc in df['preprocessed_text'].tolist())
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(df['preprocessed_text'])

    # Преобразование последовательностей в числовые вектора
    maxSequenceLength = max_words
    X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train_1), maxlen=maxSequenceLength)
    X_valid = sequence.pad_sequences(tokenizer.texts_to_sequences(X_valid_1), maxlen=maxSequenceLength)
    X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test_1), maxlen=maxSequenceLength)

    num_classes = df['class'].nunique()
    y_train = keras_utils.to_categorical(y_train_1, num_classes)
    y_valid = keras_utils.to_categorical(y_valid_1, num_classes)
    y_test = keras_utils.to_categorical(y_test_1, num_classes)

    # Обучаем ансамбль моделей с беггингом
    models, histories = train_bagging_model(X_train, y_train, X_valid, y_valid, X_test, y_test, n_models=5)

    # Предсказания с помощью ансамбля
    predictions = predict_with_bagging(models, X_test)

    y_true = np.array(y_test_1)
    report = classification_report(y_true, predictions)
    print(report)

    macro_f1 = f1_score(y_true, predictions, average='macro')
    print(f'Macro F1 Score: {macro_f1}')

    conf_matr = confusion_matrix(y_true, predictions)
    plot_confusion_matrix(conf_matr, classes=df['class'].astype(str).unique(), f_size=16, filename=filename)
