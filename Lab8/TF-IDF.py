import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils as keras_utils
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adagrad, Adadelta

filename = '8_tfidf_'
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


if __name__ == "__main__":
    df = pd.read_excel("df_prep.xlsx")

    X_vt_1, X_test_1, y_vt_1, y_test_1 = train_test_split(df['preprocessed_text'], df['class'], random_state=42,
                                                          test_size=0.1, stratify=df['class'])
    X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_vt_1, y_vt_1, test_size=0.2, random_state=42)

    print(f'y_train_1: \n{y_train_1.value_counts()}')
    print(f'y_valid_1: \n{y_valid_1.value_counts()}')
    print(f'y_test_1: \n{y_test_1.value_counts()}')

    # Преобразование текстов в матрицу TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # Ограничиваем до 10k признаков
    X_train = tfidf_vectorizer.fit_transform(X_train_1).toarray()
    X_valid = tfidf_vectorizer.transform(X_valid_1).toarray()
    X_test = tfidf_vectorizer.transform(X_test_1).toarray()

    num_classes = df['class'].nunique()
    y_train = keras_utils.to_categorical(y_train_1, num_classes)
    y_valid = keras_utils.to_categorical(y_valid_1, num_classes)
    y_test = keras_utils.to_categorical(y_test_1, num_classes)

    # Модель на основе TF-IDF
    model = Sequential()
    model.add(Dense(1024,  activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    batch_size = 32
    epochs = 100

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid))

    predictions = model.predict(X_test).argmax(axis=1)
    y_true = np.array(y_test_1)
    report = classification_report(y_true, predictions)
    print(report)

    macro_f1 = f1_score(y_true, predictions, average='macro')
    print(f'Macro F1 Score: {macro_f1}')

    with open(filename + '_log.txt', 'w', encoding='utf-8') as f:
        f.write("Model Parameters:\n")
        f.write("Dense Layer:\n")
        f.write("Layer 1: 512 neurons, ReLU activation, Dropout 0.5\n")
        f.write("Layer 2: 256 neurons, ReLU activation, Dropout 0.5\n")
        f.write(f"Loss Function: categorical_crossentropy\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Metrics: Accuracy\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write(f"\nMacro F1 Score: {macro_f1}")

    conf_matr = confusion_matrix(y_true, predictions)
    plot_confusion_matrix(conf_matr, classes=df['class'].astype(str).unique(), f_size=16, filename=filename)

    # Построение графиков обучения
    plt.figure(figsize=(12, 5))

    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('График функции ошибки')
    plt.xlabel('Эпохи')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('График точности')
    plt.xlabel('Эпохи')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{filename}_training_plot.png")
    plt.show()
