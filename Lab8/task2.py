import re
import random
import pymorphy2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.text as text
import matplotlib.cm as cm

# Нужно раскоментировать чтобы работало построение разного рода графиков,
# но тогда на матрице ошибок возникают лишние линии
import seaborn as sns; sns.set()

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
import itertools


# Т.к. вопрорсов по аспирантуре мало, и в них в основном встречается подстрока 'аспирант', то можно записи проклассифицировать по наличию этой подстроки
def find_aspirant(txt):
    '''
    Возвращает True если в тексте есть слово, начинающееся на 'аспирант'
    '''
    return 'аспирант' in [x[:8] for x in txt.split()]


def classifier(X_train, y_train, C=10.):
    '''
    Возвращает обученный классификатор и векторизатор.
    '''

    tfv = TfidfVectorizer()
    X_train = tfv.fit_transform(X_train)

    clf = LogisticRegression(C=C)
    clf = clf.fit(X_train, y_train)

    return tfv, clf


def predictor(text, clf, tfv):
    '''
    text - классифицируемый текс
    clf - обученный классификатор
    tfv - обученный векторизатор
    '''
    X_test = tfv.transform([text])

    pred = clf.predict(X_test)

    return pred[0]


def plot_confusion_matrix(cm, classes, f_size=16, normalize=False, title='Матрица ошибок', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'
    """
    plt.figure(figsize=(14,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=f_size + 2)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=f_size - 6)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=f_size)
    plt.yticks(tick_marks, classes, fontsize=f_size)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print("Confusion matrix, without normalization")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=f_size)
    plt.tight_layout()
    plt.ylabel('Действительный класс', fontsize=f_size + 1)
    plt.xlabel('Предсказанный класс', fontsize=f_size + 1)
    plt.show()



if __name__ == "__main__":

    cls_dic = {1: ['Условия подачи', 'Условия подачи документов, сроки, документы, места, льготы'],
               2: ['Проходной и допустимый балл', 'Минимальный проходной балл и Минимальный балл для подачи заявления'],
               3: ['Достижения', 'Индивидуальные достижения.'],
               4: ['Общежития', 'Общежития'],
               5: ['Вступительные испытания',
                   'Вступительные испытания, экзамены, кто может поступать и сдавать экзамены'],
               6: ['Перевод', 'Перевод с направления на направление'],
               7: ['Аспирантура', 'Вопросы по аспирантуре'],
               8: ['Регистрация', 'Регистрация в электронных системах'],
               }


    df = pd.read_excel('df_prep.xlsx')
    print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(df.text, df['class'], random_state=42, test_size=0.3)

    tfv = TfidfVectorizer()  # Функция получения векторного представления
    X_train = tfv.fit_transform(X_train)
    X_test = tfv.transform(X_test)

    param_grid = {'C': [1., 10.0]}

    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=3)

    grid_search.fit(X_train, y_train)

    grid_search.score(X_test, y_test)

    grid_search.best_params_

    # Обучение
    X_train, X_test, y_train, y_test = train_test_split(df.text, df['class'], random_state=42, test_size=0.3)
    tfv, clf = classifier(X_train, y_train, C=10.0)

    # Предсказание
    pred_save = []
    class_save = []
    pred = []
    print(X_test.shape)
    for nom, txt in enumerate(X_test.values):
        if find_aspirant(txt):
            # УДАЛЕНИЕ "аспирант" по подстроке
            pred_save.append(7)
            del_index = X_test.index.to_numpy()[nom]
            X_test = X_test.drop(index=[del_index])
            class_save.append(y_test[y_test.index == del_index].values[0])
            y_test = y_test.drop(index=[del_index])
        else:
            pred.append(predictor(txt, clf, tfv))
    print(X_test.shape)

    y_test_list = y_test.tolist()
    y_test_list.extend(class_save)
    pred_list = pred[:]
    pred_list.extend(pred_save)

    mtrs = metrics.classification_report([cls_dic[x][0] for x in y_test_list], [cls_dic[x][0] for x in pred_list])
    print(mtrs)

    # conf_matr = confusion_matrix(y_test_list, pred_list, normalize='true')
    conf_matr = confusion_matrix(y_test_list, pred_list)
    plot_confusion_matrix(conf_matr, cls_list, f_size=16)

    y_te = [cls_dic[i][0] for i in y_test_list]
    y_pr = [cls_dic[i][0] for i in pred_list]
    mat = confusion_matrix(y_te, y_pr, normalize='true')
    mat = pd.DataFrame(mat, index=np.unique(y_te), columns=np.unique(y_pr))

    f, ax = plt.subplots(figsize=(16, 11))
    sns_plot = sns.heatmap(mat, annot=True, cbar=False, cmap="Greens")