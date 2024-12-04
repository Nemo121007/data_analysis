import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from keras.src.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from gensim.models import Word2Vec

filename = '5_Word2Vec_model'

# Функция для создания матрицы эмбеддингов Word2Vec
def create_embedding_matrix(word_index, word2vec_model, embedding_dim):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]
    return embedding_matrix

# Функция построения матрицы ошибок
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
    # Загрузка данных
    df = pd.read_excel("df_prep.xlsx")

    # Разделение на выборки
    X_vt_1, X_test_1, y_vt_1, y_test_1 = train_test_split(df['preprocessed_text'], df['class'], random_state=42,
                                                          test_size=0.1, stratify=df['class'])
    X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_vt_1, y_vt_1, test_size=0.2, random_state=42)

    # Токенизация текстов
    X_train_split = [text.split() for text in X_train_1]
    X_valid_split = [text.split() for text in X_valid_1]
    X_test_split = [text.split() for text in X_test_1]

    # Обучение Word2Vec
    # Размерность вектора для каждого слова
    embedding_dim = 100
    '''
    vector_size - размерность выходного вектора
    window - окно для анализа контекста
    min_count - минимальная частота для анализа
    workers - ядра
    '''
    word2vec_model = Word2Vec(sentences=X_train_split, vector_size=embedding_dim, window=5, min_count=1, workers=-1)

    # Создание индекса слов
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['preprocessed_text'])
    word_index = tokenizer.word_index

    # Преобразование текстов в последовательности
    maxSequenceLength = max(len(text.split()) for text in df['preprocessed_text'])
    X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_1), maxlen=maxSequenceLength)
    X_valid = pad_sequences(tokenizer.texts_to_sequences(X_valid_1), maxlen=maxSequenceLength)
    X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_1), maxlen=maxSequenceLength)

    # Преобразование меток классов
    num_classes = df['class'].nunique()
    y_train = to_categorical(y_train_1, num_classes)
    y_valid = to_categorical(y_valid_1, num_classes)
    y_test = to_categorical(y_test_1, num_classes)

    # Создание матрицы эмбеддингов
    embedding_matrix = create_embedding_matrix(word_index, word2vec_model, embedding_dim)

    # Построение модели
    model = Sequential()
    model.add(Embedding(
        input_dim=len(word_index) + 1,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=maxSequenceLength,
        trainable=False  # Если Word2Vec предобучен, заморозим веса
    ))
    model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # Обучение модели
    batch_size = 32
    epochs = 500

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
        f.write(report)
        f.write(f"\nMacro F1 Score: {macro_f1}")

    # Построение матрицы ошибок
    conf_matr = confusion_matrix(y_true, predictions)
    plot_confusion_matrix(conf_matr, classes=df['class'].astype(str).unique(), f_size=16, filename=filename)
