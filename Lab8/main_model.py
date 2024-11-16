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

filename = '8_'
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


if __name__ == "__main__":
    df = pd.read_excel("df_prep.xlsx")

    X_vt_1, X_test_1, y_vt_1, y_test_1 = train_test_split(df['preprocessed_text'], df['class'], random_state=42,
                                                          test_size=0.9, stratify=df['class'])
    X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_vt_1, y_vt_1, test_size=0.5, random_state=42)

    print(f'y_train_1: \n{y_train_1.value_counts()}')
    print(f'y_valid_1: \n{y_valid_1.value_counts()}')
    print(f'y_test_1: \n{y_test_1.value_counts()}')

    max_words = max(len(desc.split()) for desc in df['preprocessed_text'].tolist())
    print(f'Максимальное количество слов в самом длинном письме: {max_words} слов')

    vocab_size = 10000
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(df['preprocessed_text'])

    maxSequenceLength = max_words
    X_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train_1), maxlen=maxSequenceLength)
    X_valid = sequence.pad_sequences(tokenizer.texts_to_sequences(X_valid_1), maxlen=maxSequenceLength)
    X_test = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test_1), maxlen=maxSequenceLength)

    num_classes = df['class'].nunique()
    y_train = keras_utils.to_categorical(y_train_1, num_classes)
    y_valid = keras_utils.to_categorical(y_valid_1, num_classes)
    y_test = keras_utils.to_categorical(y_test_1, num_classes)

    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=8*maxSequenceLength, input_length=maxSequenceLength))
    # GRU имеет 2 типа ворот. Быстрее и проще. На коротких и средних промежутках не уступает LSTM
    # LSTM имеет 3 типа ворот. Дольше учится и вычисляется. Лучше запоминает контекст на больших промежутках
    model.add(GRU(64, dropout=0.35, recurrent_dropout=0.25, return_sequences=True))
    model.add(GRU(64, dropout=0.35, recurrent_dropout=0.25))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    print(model.summary())

    batch_size = 32
    epochs = 500

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid))

    predictions = model.predict(X_test).argmax(axis=1)
    y_true = np.array(y_test_1)
    report = classification_report(y_true, predictions)
    print(report)

    macro_f1 = f1_score(y_true, predictions, average='macro')
    print(f'Macro F1 Score: {macro_f1}')

    with open(filename + '_log.txt', 'w', encoding='utf-8') as f:
        f.write("Model Parameters:\n")
        f.write(f"Input Dim: {vocab_size}\n")
        f.write(f"Output Dim: {maxSequenceLength}\n")
        f.write("GRU Layer:\n")  # Обновляем название слоя
        f.write("Layer: GRU\n")
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

    conf_matr = confusion_matrix(y_true, predictions)
    plot_confusion_matrix(conf_matr, classes=df['class'].astype(str).unique(), f_size=16, filename=filename)
