import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, LSTM, Flatten
from tensorflow.keras import utils as keras_utils
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.optimizers import RMSprop
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import concatenate

filename = 'stacking_model_'
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


# Функция для создания модели с классическим эмбеддингом
def create_embedding_model(input_length, num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=4 * input_length, input_length=input_length))
    model.add(GRU(64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Функция для создания модели с TF-IDF
def create_tfidf_model(X_train_tfidf, num_classes):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_tfidf.shape[1], activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Функция для создания модели с Word2Vec
def create_word2vec_model(X_train_word2vec, num_classes):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_word2vec.shape[1], activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Функция для обучения моделей на первом уровне
def train_base_models(X_train, y_train, X_valid, y_valid, X_train_tfidf, X_train_word2vec, num_classes):
    # Модели для первого уровня
    embedding_model = create_embedding_model(X_train.shape[1], num_classes)
    tfidf_model = create_tfidf_model(X_train_tfidf, num_classes)
    word2vec_model = create_word2vec_model(X_train_word2vec, num_classes)

    # Компиляция моделей
    embedding_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy',
                            metrics=['accuracy'])
    tfidf_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    word2vec_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy',
                           metrics=['accuracy'])

    # Обучение моделей
    embedding_model.fit(X_train, y_train, batch_size=32, epochs=70, validation_data=(X_valid, y_valid))
    tfidf_model.fit(X_train_tfidf, y_train, batch_size=32, epochs=70, validation_data=(X_valid, y_valid))
    word2vec_model.fit(X_train_word2vec, y_train, batch_size=32, epochs=70, validation_data=(X_valid, y_valid))

    return embedding_model, tfidf_model, word2vec_model


# Функция для предсказания с использованием моделей первого уровня
def predict_base_models(embedding_model, tfidf_model, word2vec_model, X_test, X_test_tfidf, X_test_word2vec):
    embedding_preds = embedding_model.predict(X_test)
    tfidf_preds = tfidf_model.predict(X_test_tfidf)
    word2vec_preds = word2vec_model.predict(X_test_word2vec)

    return embedding_preds, tfidf_preds, word2vec_preds


# Стеккинг: обучение второго уровня
def train_stacking_model(X_train_meta, y_train, X_valid_meta, y_valid, num_classes):
    input_layer = Input(shape=(X_train_meta.shape[1],))
    dense1 = Dense(64, activation='relu')(input_layer)
    output_layer = Dense(num_classes, activation='softmax')(dense1)

    stacking_model = Model(inputs=input_layer, outputs=output_layer)
    stacking_model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy',
                           metrics=['accuracy'])

    stacking_model.fit(X_train_meta, y_train, batch_size=32, epochs=70, validation_data=(X_valid_meta, y_valid))

    return stacking_model


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

    # Генерация TF-IDF признаков
    tfidf_vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_1).toarray()
    X_valid_tfidf = tfidf_vectorizer.transform(X_valid_1).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(X_test_1).toarray()

    # Убедитесь, что X_train_tfidf и другие массивы имеют форму (samples, features)
    print(f"X_train_tfidf shape: {X_train_tfidf.shape}")
    print(f"X_valid_tfidf shape: {X_valid_tfidf.shape}")
    print(f"X_test_tfidf shape: {X_test_tfidf.shape}")

    # Обучение Word2Vec
    sentences = [text.split() for text in df['preprocessed_text']]
    word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_vectors = {word: word2vec_model.wv[word] for word in word2vec_model.wv.index_to_key}


    def get_word2vec_embeddings(texts):
        embeddings = np.zeros((len(texts), 100))
        for i, text in enumerate(texts):
            words = text.split()
            word_embeddings = np.array([word2vec_vectors.get(word, np.zeros(100)) for word in words])
            if word_embeddings.any():
                embeddings[i] = word_embeddings.mean(axis=0)
        return embeddings


    X_train_word2vec = get_word2vec_embeddings(X_train_1)
    X_valid_word2vec = get_word2vec_embeddings(X_valid_1)
    X_test_word2vec = get_word2vec_embeddings(X_test_1)

    # Обучение моделей первого уровня
    embedding_model, tfidf_model, word2vec_model = train_base_models(X_train, y_train, X_valid, y_valid, X_train_tfidf,
                                                                     X_train_word2vec, num_classes)

    # Предсказания на тестовых данных
    embedding_preds, tfidf_preds, word2vec_preds = predict_base_models(embedding_model, tfidf_model, word2vec_model,
                                                                       X_test, X_test_tfidf, X_test_word2vec)

    # Создаем мета-признаки для второго уровня
    meta_X_train = np.concatenate([embedding_preds, tfidf_preds, word2vec_preds], axis=1)
    meta_X_test = np.concatenate([embedding_preds, tfidf_preds, word2vec_preds], axis=1)

    # Обучение модели второго уровня
    stacking_model = train_stacking_model(meta_X_train, y_train, meta_X_test, y_test, num_classes)

    # Предсказания с использованием модели второго уровня
    stacking_predictions = stacking_model.predict(meta_X_test)
    stacking_predictions = np.argmax(stacking_predictions, axis=1)

    # Оценка модели
    report = classification_report(np.argmax(y_test, axis=1), stacking_predictions)
    print(report)

    macro_f1 = f1_score(np.argmax(y_test, axis=1), stacking_predictions, average='macro')
    print(f'Macro F1 Score: {macro_f1}')

    conf_matr = confusion_matrix(np.argmax(y_test, axis=1), stacking_predictions)
    plot_confusion_matrix(conf_matr, classes=df['class'].astype(str).unique(), f_size=16, filename=filename)
