import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, LSTM
from tensorflow.keras import utils as keras_utils
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.optimizers import RMSprop, Adagrad, Adadelta

filename = 'preprocessed_text_'
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
    cls_dic = {0: ['Условия подачи', 'Условия подачи документов, сроки, документы, места, льготы'],
               1: ['Проходной и допустимый балл', 'Минимальный проходной балл и Минимальный балл для подачи заявления'],
               2: ['Достижения', 'Индивидуальные достижения.'],
               3: ['Общежития', 'Общежития'],
               4: ['Вступительные испытания',
                   'Вступительные испытания, экзамены, кто может поступать и сдавать экзамены'],
               5: ['Перевод', 'Перевод с направления на направление'],
               6: ['Аспирантура', 'Вопросы по аспирантуре'],
               7: ['Регистрация', 'Регистрация в электронных системах'],
               }

    cls_list = []
    for i in range(0, len(cls_dic)):
        cls_list.append(cls_dic[i][0])


    df = pd.read_excel("df_prep.xlsx")

    X_train, X_test, y_train, y_test = train_test_split(df['preprocessed_text'], df['class'], random_state=42, test_size=0.3)

    # создаем единый словарь (слово -> число) для преобразования
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_vt_1, X_test_1, y_vt_1, y_test_1 = train_test_split(df['preprocessed_text'], df['class'], random_state=42,
                                                          test_size=0.1, stratify=df['class'])
    X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_vt_1, y_vt_1, test_size=0.2, random_state=42)

    # Максимальное количество слов в самом длинном письме
    max_words = 0
    for desc in df['preprocessed_text'].tolist():
        words = len(desc.split())
        if words > max_words:
            max_words = words
    print('Максимальное количество слов в самом длинном письме: {} слов'.format(max_words))

    total_unique_words = len(tokenizer.word_counts)
    print('Всего уникальных слов в словаре: {}'.format(total_unique_words))

    maxSequenceLength = max_words

    # Преобразуем описания заявок в векторы чисел
    vocab_size = 10000
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(df['preprocessed_text'])

    X_train = tokenizer.texts_to_sequences(X_train_1)
    X_valid = tokenizer.texts_to_sequences(X_valid_1)
    X_test = tokenizer.texts_to_sequences(X_test_1)

    X_train = sequence.pad_sequences(X_train, maxlen=maxSequenceLength)
    X_valid = sequence.pad_sequences(X_valid, maxlen=maxSequenceLength)
    X_test = sequence.pad_sequences(X_test, maxlen=maxSequenceLength)

    print('Размерность X_train:', X_train.shape)
    print('Размерность X_valid:', X_valid.shape)
    print('Размерность X_test:', X_test.shape)

    # Преобразуем категории в матрицу двоичных чисел (для использования categorical_crossentropy)

    num_classes = df['class'].unique().shape[0] + 1

    y_train = keras_utils.to_categorical(y_train_1, num_classes)
    y_valid = keras_utils.to_categorical(y_valid_1, num_classes)
    y_test = keras_utils.to_categorical(y_test_1, num_classes)
    print('y_train shape:', y_train.shape)
    print('y_valid shape:', y_valid.shape)
    print('y_test shape:', y_test.shape)

    # максимальное количество слов для анализа
    max_features = vocab_size

    print(u'Собираем модель...')
    model = Sequential()
    model.add(Embedding(max_features, maxSequenceLength))
    model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())

    batch_size = 32
    epochs = 25

    print(u'Тренируем модель...')
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_valid, y_valid))

    predictions = model.predict(X_test).argmax(axis=1)
    y2 = np.array(y_test_1.to_list())
    pred2 = np.array(predictions)

    print(classification_report(y2, pred2))

    report = classification_report(y_test_1, pred2)

    macro_f1 = f1_score(y_test_1, pred2, average='macro')
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

    conf_matr = confusion_matrix(y_test_1, predictions)
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
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')  # Изменено на 'accuracy'
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')  # Изменено на 'val_accuracy'
    plt.title('График точности')
    plt.xlabel('Эпохи')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{filename}_training_plot.png")
    plt.show()


