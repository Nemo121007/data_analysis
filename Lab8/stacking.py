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
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

# Фиксируем сиды
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
tf.keras.utils.set_random_seed(42)

filename = 'stacking_model_'


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


def create_binary_model(vocab_size, output_dim, max_sequence_length, dropout, recurrent_dropout):
    """Создает бинарную модель для стекинга"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=max_sequence_length),
        GRU(64, dropout=dropout, recurrent_dropout=recurrent_dropout),
        Dense(1, activation='sigmoid')  # Для бинарной классификации
    ])
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def create_multiclass_model(input_dim, num_classes):
    """Создает вторую модель для объединения результатов первого уровня"""
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def log_to_file(filename, content):
    """Функция логирования"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)


def bootstrap_sampling(X, y, seed):
    """Функция для создания бутсрэп-выборки"""
    X_1, X_2, y_1, y_2 = train_test_split(X, y, random_state=seed, test_size=0.3)
    return X_1, y_1



if __name__ == "__main__":
    # Загрузка данных
    df = pd.read_excel("df_prep.xlsx")
    X_vt_1, X_test_1, y_vt_1, y_test_1 = train_test_split(df['preprocessed_text'], df['class'], random_state=42,
                                                          test_size=0.1, stratify=df['class'])
    X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_vt_1, y_vt_1, test_size=0.2, random_state=42)

    vocab_size = 10000
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(df['preprocessed_text'])

    maxSequenceLength = max(len(desc.split()) for desc in df['preprocessed_text'].tolist())
    X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train_1), maxlen=maxSequenceLength)
    X_valid = sequence.pad_sequences(tokenizer.texts_to_sequences(X_valid_1), maxlen=maxSequenceLength)
    X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test_1), maxlen=maxSequenceLength)

    num_classes = df['class'].nunique()
    output_dim = 4 * maxSequenceLength

    # Первый уровень стемпинга: бинарные модели с бутсрэпом
    binary_models = []
    binary_predictions_valid = []
    binary_predictions_test = []
    class_labels = sorted(df['class'].unique())

    count = 0
    for class_label in class_labels:
        count += 1
        print(f"Тренировка бинарной модели для класса {class_label}...")

        # Бинаризация меток для текущего класса
        y_train_binary = (y_train_1 == class_label).astype(int)
        y_valid_binary = (y_valid_1 == class_label).astype(int)

        # Создание модели
        binary_model = create_binary_model(vocab_size, output_dim, maxSequenceLength, dropout=0.2, recurrent_dropout=0.2)

        # Создание бутсрэп-выборки
        X_train_bootstrap, y_train_binary_bootstrap = bootstrap_sampling(X_train, y_train_binary, seed=count)

        # Обучение модели на бутсрэп-выборке
        binary_model.fit(X_train_bootstrap, y_train_binary_bootstrap, validation_data=(X_valid, y_valid_binary), batch_size=32, epochs=10, verbose=1)

        # Предсказания для валидации и теста
        binary_predictions_valid.append(binary_model.predict(X_valid).flatten())
        binary_predictions_test.append(binary_model.predict(X_test).flatten())

        binary_models.append(binary_model)

    # Формируем входные данные для второго уровня (объединяем предсказания)
    binary_predictions_valid = np.array(binary_predictions_valid).T
    binary_predictions_test = np.array(binary_predictions_test).T

    # Второй уровень стемпинга: многоклассовая модель
    print("Тренировка модели второго уровня...")
    multiclass_model = create_multiclass_model(input_dim=len(class_labels), num_classes=num_classes)

    # Преобразование меток для многоклассовой задачи
    y_train_multiclass = keras_utils.to_categorical(y_valid_1.map(lambda x: class_labels.index(x)), num_classes)
    y_test_multiclass = keras_utils.to_categorical(y_test_1.map(lambda x: class_labels.index(x)), num_classes)

    # Обучение модели второго уровня
    history = multiclass_model.fit(binary_predictions_valid, y_train_multiclass, epochs=10, batch_size=32, verbose=1)

    # Предсказания и оценка
    final_predictions = multiclass_model.predict(binary_predictions_test).argmax(axis=1)
    y_true = y_test_1.map(lambda x: class_labels.index(x)).values

    # Проверяем уникальные метки
    unique_classes_in_y_true = np.unique(y_true)
    print(f"Уникальные метки в y_true: {unique_classes_in_y_true}")
    print(f"Классы из class_labels: {class_labels}")

    # Если классы из y_true не совпадают с class_labels, корректируем target_names
    actual_class_labels = [str(class_labels[i]) for i in unique_classes_in_y_true]

    # Генерация отчета
    report = classification_report(y_true, final_predictions, target_names=actual_class_labels)
    print(report)

    # Логирование
    log_to_file(filename + "_log.txt", report)

    # Матрица ошибок
    conf_matr = confusion_matrix(y_true, final_predictions)
    plot_confusion_matrix(conf_matr, classes=class_labels, f_size=16, filename=filename)

    # Построение графиков
    plt.figure(figsize=(12, 5))

    # График функции ошибки
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.title('График функции ошибки')
    plt.xlabel('Эпохи')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.title('График точности')
    plt.xlabel('Эпохи')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{filename}_training_plot.png")
    plt.show()

