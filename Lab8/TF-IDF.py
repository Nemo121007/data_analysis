from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras import utils as keras_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

filename = '4_tfidf_model'

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

if __name__ == "__main__":
    df = pd.read_excel("df_prep.xlsx")

    # Разделение на обучающую, валидационную и тестовую выборки
    X_vt_1, X_test_1, y_vt_1, y_test_1 = train_test_split(
        df['preprocessed_text'], df['class'], random_state=42, test_size=0.9, stratify=df['class']
    )
    X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(
        X_vt_1, y_vt_1, test_size=0.5, random_state=42
    )

    # Преобразование текстов с использованием TF-IDF
    maxSequenceLength = max(len(text.split()) for text in df['preprocessed_text'])
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train = vectorizer.fit_transform(X_train_1).toarray()
    X_valid = vectorizer.transform(X_valid_1).toarray()
    X_test = vectorizer.transform(X_test_1).toarray()

    # Количество классов
    num_classes = df['class'].nunique()

    # Преобразование меток в one-hot формат
    y_train = keras_utils.to_categorical(y_train_1, num_classes)
    y_valid = keras_utils.to_categorical(y_valid_1, num_classes)
    y_test = keras_utils.to_categorical(y_test_1, num_classes)

    # Определение модели
    model = Sequential()
    model.add(Dense(maxSequenceLength, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    print(model.summary())

    # Параметры обучения
    batch_size = 32
    epochs = 500

    # Обучение модели
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid))

    # Оценка модели
    predictions = model.predict(X_test).argmax(axis=1)
    y_true = np.array(y_test_1)
    report = classification_report(y_true, predictions)
    print(report)

    macro_f1 = f1_score(y_true, predictions, average='macro')
    print(f'Macro F1 Score: {macro_f1}')

    # Сохранение результатов
    with open(filename + '_log.txt', 'w', encoding='utf-8') as f:
        f.write("Model Parameters:\n")
        f.write("Layer 1: Dense, 64 neurons, ReLU\n")
        f.write("Layer 2: Dense, 32 neurons, ReLU\n")
        f.write("Output Layer: Dense, Softmax\n")
        f.write(f"Loss Function: categorical_crossentropy\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Metrics: Accuracy\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write(f"\nMacro F1 Score: {macro_f1}")

    # Построение и сохранение матрицы ошибок
    conf_matr = confusion_matrix(y_true, predictions)
    plot_confusion_matrix(conf_matr, classes=df['class'].astype(str).unique(), f_size=16, filename=filename)
