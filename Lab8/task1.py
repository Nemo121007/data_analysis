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

import re
import pymorphy2
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import ngrams
from collections import Counter
from nltk.corpus import stopwords
import pandas as pd


def del_punct(text):
    """
    Удаляет знаки пунктуации из текста.
    """
    return re.sub(r'[^\w\s]', '', text)


def morphan(word, morph):
    '''
    Приведение слов в нормальную форму, удаление числительных и ФИО.
    '''
    word = del_punct(word).strip()
    p = morph.parse(word)[0]

    word_new = word
    if (not 'Surn' in p.tag) and (not 'Name' in p.tag) and (not 'Patr' in p.tag) and ('NOUN' in p.tag):
        #существительное не ФИО
        word_new = p.normal_form
    elif 'Surn' in p.tag:
        word_new = 'ФАМИЛИЯ'
    elif 'Name' in p.tag:
        word_new = 'ИМЯ'
    elif 'Patr' in p.tag:
        word_new = 'ОТЧЕСТВО'


    elif ('INFN' in p.tag) or ('VERB' in p.tag): #глагол
        word_new = p.normal_form

    elif ('ADJF' in p.tag) or ('ADJS' in p.tag) or ('COMP' in p.tag): #прилагательное
        word_new = p.normal_form


    elif ('PRTF' in p.tag) or ('PRTS' in p.tag) or ('GRND' in p.tag): #причастие, похоже на глагол
        word_new = p.normal_form

    elif ('ADVB' in p.tag) or ('NPRO' in p.tag) or ('PRED' in p.tag) or ('PREP' in p.tag) or ('CONJ' in p.tag) or ('PRCL' in p.tag) or ('INTJ' in p.tag):
        # предлоги, местоимения и пр.
        word_new = p.normal_form

    elif ('NUMR' in p.tag) or ('NUMB' in p.tag) or ('intg' in p.tag): # числительные NUMB,intg
        word_new = ''

    else:
        word_new = word
    return word_new


def normtext(txt, morph):
    '''
    Возвращает текст из слов в нормальной форме
    '''
    return str(' '.join([morphan(x, morph) for x in txt.split()]))



if __name__ == "__main__":

    # Загружаем список русских стоп-слов
    stop_words = set(stopwords.words("russian"))

    # Инициализируем лемматизатор
    morph = pymorphy2.MorphAnalyzer()

    # Пример словаря синонимов (можно расширить по необходимости)
    synonyms_dict = {
        'преподаватель': 'учитель',
        'поступление': 'зачисление',
        'экзамен': 'тест'
    }

    pd.options.display.width = 0
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 11)

    df = pd.read_excel('mails.xlsx')
    df['TYPE_HOTLINE'].unique()

    # Меняем наименование столбца только для удобства
    df.rename(columns={'CLASS_': 'lbl'}, inplace=True)
    # Для удобства добавляем два столбца с наименованиями классов и каналов поствупления обращений.
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
    hlt_dic = {1: 'ONLINE приёмная',
               2: 'Очная приемная',
               3: 'Приёмная аспирантуры'
               }

    df['cls_name'] = df['class'].map(lambda x: cls_dic[x][0])
    df['hlt_name'] = df.TYPE_HOTLINE.map(lambda x: hlt_dic[x])
    print(df.head())

    cls_list = []
    for i in range(1, len(cls_dic) + 1):
        cls_list.append(cls_dic[i][0])

    g = df.groupby('cls_name')['cls_name'].count().sort_values()
    print(df.groupby('cls_name')['cls_name'].count())

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # График 1: Распределение по категориям обращений
    sns.barplot(ax=ax1, x=g.values, y=g.index, orient='h')
    ax1.set_xlabel('Количество обращений')
    ax1.set_ylabel('Категория обращения')
    ax1.set_title('Распределение обращений по категориям')

    # График 2: Распределение по типам горячей линии
    g = df.groupby('TYPE_HOTLINE')['TYPE_HOTLINE'].count().sort_values()
    sns.barplot(ax=ax2, x=g.values, y=g.index, orient='h')
    ax2.set_xlabel('Количество обращений')
    ax2.set_ylabel('Тип горячей линии')
    ax2.set_title('Распределение обращений по типам горячей линии')

    plt.show()

    morph = pymorphy2.MorphAnalyzer()

    df['text'] = df.CONTENT.map(lambda x: normtext(x, morph))
    print(df.head())

    # Сохраняем обработанные данные
    df.to_excel('df_prep.xlsx', index=False)
