import matplotlib.pyplot as plt

# Нужно раскоментировать чтобы работало построение разного рода графиков,
# но тогда на матрице ошибок возникают лишние линии
import seaborn as sns
import re
from pymorphy3 import MorphAnalyzer
from collections import Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords

sns.set()
Count = 0
Morph = MorphAnalyzer()
dict_error = {
    "уважние": "уважение",
    "психолый-педагогический": "психолого-педагогический",
    "дистационный": "дистанционный",
    "ягражданин": "я гражданин",
    "соотвествуть": "соответствовать",
    "аспиринтура": "аспирантура",
    "отзвать": "отозвать",
    "перездать": "пересдать",
    "петрг": "ПетрГУ",
    "напрвление": "направление",
    "обслеживание": "обслуживание",
    "непригодиться": "не пригодиться",
    "тотожий": "тот",
    "учавствовать": "участвовать",
    "всвязь": "в связи",
    "лечфак": "лечебный факультет",
    "обьявленный": "объявленный",
    "намисанный": "написанный",
    "подгтовка": "подготовка",
    "птрг": "ПетрГУ",
    "твовать": "присутствовать",
    "переподать": "преподать",
    "зарегестрироваться": "зарегистрироваться",
    "волонтерство": "волонтерство",
    "дизайн-пб": "дизайн",
    "допнабор": "дополнительный набор",
    "тд": "так далее",
    "переть": "идти",
    "побиология": "по биологии",
    "траспортно-технологический": "транспортно-технологический",
    "скалиоз": "сколиоз",
    "чие": "чье",
    "балломить": "баллов",
    "симоново": "Семён",
    "концеренция": "конференция",
    "ифксит": "институт",
    "хоз": "хозяйственный",
    "бюджетныe": "бюджетные",
    "вступа": "вступать",
    "вст": "встать",
    "пришëть": "пришел",
    "ещë": "еще",
    "здраствовать": "здравствовать",
    "нибыть": "нибудь",
    "бюд": "бюджет",
    "iv": "ЧИСЛО",
    "педаго-психолог": "педагог-психолог",
    "дркумёт": "документ",
    "пофамильный": "по ФАМИЛИЯ",
    "напрвления-дизайн": "направления дизайн",
    "бюдж": "бюджет",
    "фп": "физическая подготовка",
    "билогия": "биология",
    "попоступать": "поступать",
    "соц": "социальный",
    "фактчекинг": "проверить факт",
    "призерство": "призер",
    "-ех": " ",
    "сдавть": "сдавать",
    "сми": "средства массовой информации",
    'инострый': 'иностранный',
    'англ': 'английский',
    '-немой': 'немой',
    '-англ': 'английский',
    'пу-': 'профессиональное училище',
    'постепление': 'поступление',
    'раброт': 'работ',
    'аттест': 'аттестат',
    'проф': 'профессиональный',
    'перепоступление': 'поступление',
    'родочинский': 'ФАМИЛИЯ',
    'здравствуйтеть': 'здравствуйте',
    'е-мейл': 'почта',
    'обжа': 'ОБЖ',
    'рег': 'регистрация',
    'бал': 'балл'
}
nltk.download('stopwords')
russian_stopwords = stopwords.words('russian')
list_stop_words = ['здравствуйте', 'это', 'спасибо', 'уважение', 'нужно', 'ещё', 'который', 'всё', '-', 'возможно',
                   'необходимо', 'также', 'сколько', 'ви', 'каждый', 'просто', 'следующий', 'г', 'почему', 'например',
                   'правильно', 'ранний', 'несколько', 'какой-то', 'новый',
                   'благодарить', 'хотя', 'очень', 'сразу', 'из-за', 'поздно', 'вообще', 'назад', 'оно', 'лично',
                   'туда', 'необходимый', 'никакой', 'как-то', 'её', 'некоторый', 'лишь', 'каковой', 'точный', 'однако',
                   'какой-либо', 'нигде', '_____', 'заново', 'пока',
                   'ранее', 'что-либо', 'снова', 'ин', 'около', 'затем', 'любой', 'по', 'вместо', 'особый', 'никак',
                   'всё-таки', 'целое', 'никто', 'п', 'особо', '-то', 'сей', 'пора', 'внутри', 'последний', 'что-то',
                   '-ий', 'войти', '-ом', 'крайний', 'иу', 'кл', '-й', 'n', 'никуда', 'кто-то', '-х', 'чей', 'ч', 'не',
                   'так', 'в', 'уважаемый', 'kак', 'э', 'значит', 'я', 'вроде', 'ой', 'наотрез', 'примерный', '-м',
                   'так-как', 'чье', 'подпункт', 'где-то', 'нету', 'еще', 'вновь', 'вновь', 'ниже', 'немного',
                   'почему-то', '-ми', '-ми', 'тот', 'взад', 'неясно', 'либо', 'везде', 'часто', 'больший',
                   'какой-нибудь', 'х', '-у']
dict_synonyms = {
    'хотеть': 'сделать',
    'поступить': 'поступление',
    'поступать': 'поступление',
    'профиль': 'направление',
    'возможность': 'мочь',
    'зачисление': 'поступление',
    'специальность': 'направление',
    'результат': 'балл',
    'факультет': 'ИНСТИТУТ',
    'конкурс': 'испытание',
    'мочь': 'сделать',
    'помочь': 'сделать',
    'бюджетный': 'бюджет',
    'первый': 'ЦИФРА',
    'университет': 'ИНСТИТУТ',
    'число': 'ЦИФРА',
    'ваш': 'ИМЯ',
    'дать': 'сделать',
    'август': 'ДАТА',
    'июль': 'ДАТА',
    'язык': 'ЯЗЫК',
    'русский': 'ЯЗЫК',
    'сдать': 'сделать',
    'второй': 'ЦИФРА',
    'целевой': 'бюджет',
    'офп': 'испытание',
    'приём': 'поступление',
    'указать': 'сделать',
    'заселение': 'общежитие',
    'достижение': 'испытание',
    'день': 'ДАТА',
    'физический': 'ИНСТИТУТ',
    'медицинский': 'ИНСТИТУТ',
    'отозвать': 'согласие',
    'волна': 'поступление',
    'сдача': 'сделать',
    'лечебный': 'ИНСТИТУТ',
    'зачислить': 'поступление',
    'сегодня': 'ДАТА',
    'смочь': 'сделать',
    'просить': 'сделать',
    'дата': 'ДАТА',
    'набор': 'поток',
    'получаться': 'сделать',
    'хотеться': 'хотеть',
    'пройда': 'поступить',
    'платно': 'бюджет',
    'указывать': 'сделать',
    'спо': 'ИНСТИТУТ',
    'гто': 'испытание',
    'сентябрь': 'ДАТА',
    'иметься': 'есть',
    'английский': 'ЯЗЫК',
    'медаль': 'достижение',
    'zoom': 'дистанционный',
    'удостоверение': 'документ',
    'регистрационный': 'регистрация',
    'колледж': 'ИНСТИТУТ',
    'заведение': 'ИНСТИТУТ',
    'доп': 'дополнительный',
    'отчислиться': 'отчисление',
    'квота': 'целевой',
    'олимпиада': 'испытание',
    'мёд': 'медицинский',
    'физвоз': 'ИНСТИТУТ',
    'военный': 'армия',
    'бакалавр': 'бакалавриат',
    'вчера': 'ДАТА',
    'накануне': 'ДАТА',
    'фк': 'физический',
    'студенческий': 'студент',
    'физ': 'физический',
    'пед': 'педагогический',
    'завтра': 'ДАТА',
    'шведский': 'ЯЗЫК',
    'кардиология': 'медицинский',
    'серебряный': 'достижение',
    'инф': 'информация',
    'физкультурный': 'физический',
    'имя': 'ИМЯ',
    'мат': 'материальный',
    'универ': 'ИНСТИТУТ',
    'бжд': 'ОБЖ',
    'платник': 'бюджет',
    'боевой': 'армия',
    'зум': 'дистанционный',
    'ПетрГУ': 'ИНСТИТУТ',
    'упр': 'управление',
    'гос': 'государство',
    'эл': 'дистанционный',
    'жизнедеятельность': 'ОБЖ',
    'безопасность': 'ОБЖ',
    'училище': 'ИНСТИТУТ',
    'кмс': 'физический достижение',
    'российский': 'рф',
    'яз': 'ЯЗЫК',
    '-педагогический': 'педагогический',
    'бух': 'бухгалтерский',
    'ii': 'ЦИФРА',
    'служба': 'армия',
    'физическая': 'физический',
    'дозвониться': 'сообщить',
    'июнь': 'ДАТА',
    'док': 'документ',
    'апрель': 'ДАТА',
    'Семён': 'ИМЯ',
    'первенство': 'достижение',
    'россия': 'рф',
    'баллов': 'балл',
    'удалённый': 'дистанционный',
    'корона-вирус': 'пандемия',
    'первокурсник': 'студент',
    'зачётный': 'зачетка',
    'книжка': 'зачетка',
    'интернет-олимпиада': 'испытание',
    'спб': 'санкт-петербург',
    'деканат': 'дирекция',
    'бег': 'физический',
    'прыжок': 'физический',
    'зачётка': 'зачетка',
    '-медицинский': 'медицинский',
    'вакантный': 'вакансия',
    'док-т': 'документ',
    'биология-': 'биология',
    'призовой': 'достижение',
    'отбор': 'конкурс',
    'бухучёт': 'бухгалтерский',
    'французский-английский': 'ЯЗЫК',
    'обучения-бюджет': 'бюджет',
    'педагог-': 'педагог',
    'экзамена-': 'экзамена'
}


def write_list_unique_words(df):
    unique_words_df = pd.DataFrame(columns=['word', 'count'])

    # Объединяем все тексты в один большой список
    all_words = ' '.join(df['preprocessed_text']).split()

    # Подсчитываем частоту слов с помощью Counter
    word_counts = Counter(all_words)

    # Заполняем новый DataFrame
    unique_words_df['word'] = list(word_counts.keys())
    unique_words_df['count'] = list(word_counts.values())

    unique_words_df = unique_words_df.sort_values(by='count', ascending=False)

    # Сохраняем в файл
    unique_words_df.to_csv('unique_words.csv', index=False, sep='\t')


def append_column_comment(df):
    """
    Добавление к датафрейму солбцов с пояснениями
    """
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
    df['class'] = df['class'].map(lambda x: int(x) - 1)
    df['hlt_name'] = df['TYPE_HOTLINE'].map(lambda x: hlt_dic[x])
    df['preprocessed_text'] = df['CONTENT'].map(lambda x: del_punctuation(x.lower()))


def del_punctuation(raw_text):
    """
    Удаляет знаки пунктуации из текста.
    """
    '''
    [] - множество, к которому применяется ^
    ^ - не
    \w - любые буквы и цифры
    \s - пробел
    '''
    # Замена всех найденных последовательностей на "ЦИФРА"
    result = re.sub(r'\d+', ' ЦИФРА ', raw_text)
    result = re.sub(r'[^\w\s-]', ' ', result)
    return result



# def error_correction(error_text: str):
#     current_word = ""
#
#     error_text = del_punctuation(error_text).strip()
#
#     # list_missing_space = ['здравствуйте', 'например', 'пожалуйста', 'подскажите', 'офп', 'спасибо',
#     #                       'несколько', 'сегодня', 'право', 'гражданин', 'электротехника', 'почему', 'методический',
#     #                       'интернет', 'предметы', 'нужный', 'профилю', 'выпускник', 'согласие',
#     #                       'отделение', 'егэ', 'бюджет', 'абитуриент', 'тестирование', 'уточнить', 'управленческий',
#     #                       'учиться', 'конкурс', 'образование', 'индивидуальный', 'английский', 'специальности',
#     #                       'день', 'приоритет', 'технологический', 'целевой', 'август', 'культура'
#     #                       'лучше', 'порядка', 'языку', 'обращаться', 'приоритеты', 'узнать', 'бакалавриат', 'как',
#     #                       'заявления', 'финский', 'платно', 'лет', 'биология', 'комиссию', 'английский', 'химия',
#     #                       'пригодиться', 'исправить', 'связь', 'скажите', 'мочь', 'собеседованию', 'собеседование',
#     #                       'возможности', 'собеседовании', 'списке', 'учебно', 'места', 'педагогический', 'чего', 'что',
#     #                       'дизайн', 'лучше', 'порядок', 'можно', 'вопрос', 'институт', 'если', 'нельзя']
#     #
#     # for word in list_missing_space:
#     #     error_text = error_text.replace(word, f' {word} ')
#     #
#     # dict_error_word = {
#     #     'пожалуйст': 'пожалуйста',
#     #     'уважние': 'уважение',
#     #     'опедагогический': 'педагогический',
#     #     'указаночтый': 'указанный',
#     #     'попоступать': 'поступать',
#     #     'собеседованиить': 'собеседование',
#     #     'иеть': 'и',
#     #     'чтоть': 'что',
#     #     'местан': 'места',
#     #     'оеть': 'и',
#     #     'иелибо': 'либо',
#     #     'увиделачтый': 'увиденный',
#     #     'вступа': 'вступал',
#     #     'упражнениймн': 'упражнений',
#     #     'такести': 'так есть',
#     #     'психолого': 'психолог',
#     #     'концеренция': 'конфуренция',
#     #     'уважнием': 'уважением',
#     #     'изза': 'из-за',
#     #     'психологопедагогическое': 'психолого-педагогическое',
#     #     'ието': 'и это',
#     #     'испытанияа': 'испытания',
#     #     'чтото': 'что-то',
#     #     'оето': 'это',
#     #     'докты': 'документы',
#     #     'психолый-педагогический': 'психолого-педагогическое',
#     #     'психолый': 'психологический',
#     #     'твовать': 'присутсвовать',
#     #     'физикетк': 'физике',
#     #     'физтехнический': 'физико-технический',
#     #     'пришëть': 'пришёл',
#     #     'делоиести': 'дело есть',
#     #     'проходногоесть': 'проходного есть',
#     #     'дркумёт': 'документ',
#     #     'ехотела': 'хотела',
#     #     'напрвления': 'направления',
#     #     'птрг': 'УНИВЕРСИТЕТ',
#     #     'матпроф': 'математический профиль',
#     #     'анажимаю': 'нажимаю',
#     #     'ихлибо': 'их либо',
#     #     'нужнаа': 'нужна',
#     #     'ктото': 'кто-то',
#     #     'учавствует': 'участвует',
#     #     'университетно': 'УНИВЕРСИТЕТ',
#     #     'перездать': 'пересдать',
#     #     'намисанно': 'написано',
#     #     'обьявленно': 'объявлено',
#     #     'твуйте': 'здравствуйте',
#     #     'ыдокументы': 'документы',
#     #     'элпочту': 'электронную почту',
#     #     'аспиринтуру': 'аспирантуру',
#     #     'соотвествующего': 'соответствующего',
#     #     'сканкопии': 'копии',
#     #     'темасамо': 'тема само',
#     #     'амисколько': 'сколько',
#     #     'птргу': 'УНИВЕРСИТЕТ',
#     #     'траспортнотехнологических': 'траспортно-технологических',
#     #     'регn': 'регистрация',
#     #     'здраствуйте': 'здравствуйте',
#     #     'квотецелевому': 'квоте целевому',
#     #     'всётаки': 'всё таки',
#     #     'призерство': 'призер',
#     #     'сделалс': 'сделал',
#     #     'напрвалений': 'направлений'
#     #     # '': '',
#     #     # '': '',
#     #     # '': '',
#     # }
#     #
#     # # Исправляем орфографические ошибки
#     # list_word = error_text.split(' ')
#     # for word in list_word:
#     #     if word in dict_error_word.keys():
#     #         error_text = error_text.replace(f' {word} ', f' {dict_error_word[word]} ')
#
#     return error_text


def normalization_form_world(word, morph=Morph):
    """
    Приведение слов в нормальную форму, удаление числительных и ФИО.
    """
    p = morph.parse(word)[0]

    global dict_error, russian_stopwords, list_stop_words
    word_new = p.normal_form

    if dict_error.__contains__(word_new):
        word_new = word_new.replace(word_new, dict_error[word_new])

    if russian_stopwords.__contains__(word_new):
        word_new = ''
    elif list_stop_words.__contains__(word_new):
        word_new = ''
    elif (not 'Surn' in p.tag) and (not 'Name' in p.tag) and (not 'Patr' in p.tag) and ('NOUN' in p.tag):  # существительное не ФИО
        word_new = p.normal_form
    elif 'Surn' in p.tag:
        word_new = 'ФАМИЛИЯ'
    elif 'Name' in p.tag:
        word_new = 'ИМЯ'
    elif 'Patr' in p.tag:
        word_new = 'ОТЧЕСТВО'
    elif ('INFN' in p.tag) or ('VERB' in p.tag):  # глагол
        word_new = p.normal_form
    elif ('ADJF' in p.tag) or ('ADJS' in p.tag) or ('COMP' in p.tag):  # прилагательное
        word_new = p.normal_form
    elif ('PRTF' in p.tag) or ('PRTS' in p.tag) or ('GRND' in p.tag):  # причастие, похоже на глагол
        word_new = p.normal_form
    elif ('ADVB' in p.tag) or ('NPRO' in p.tag) or ('PRED' in p.tag) or ('PREP' in p.tag) or ('CONJ' in p.tag) or (
            'PRCL' in p.tag) or ('INTJ' in p.tag): # предлоги, местоимения и пр.
        word_new = p.normal_form
    elif ('NUMR' in p.tag) or ('NUMB' in p.tag) or ('intg' in p.tag):  # числительные NUMB,intg
        word_new = ''
    else:
        word_new = word

    global dict_synonyms
    if dict_synonyms.__contains__(word_new):
        word_new = dict_synonyms[word_new]

    return word_new


def normalization_form_text(txt):
    """
    Возвращает текст из слов в нормальной форме
    """
    global Count
    Count = Count + 1
    print(Count)

    result = str(' '.join([normalization_form_world(x) for x in txt.split()]))

    return result


if __name__ == "__main__":
    df = pd.read_excel('mails.xlsx')
    append_column_comment(df)

    df['preprocessed_text'] = df['preprocessed_text'].map(lambda x: normalization_form_text(x))

    write_list_unique_words(df)

    # Загружаем список русских стоп-слов
    stop_words = set(stopwords.words("russian"))

    # Пример словаря синонимов (можно расширить по необходимости)
    synonyms_dict = {
        'преподаватель': 'учитель',
        'поступление': 'зачисление',
        'экзамен': 'тест'
    }


    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # График 1: Распределение по категориям обращений
    g = df.groupby('cls_name')['cls_name'].count().sort_values()
    print(df.groupby('cls_name')['cls_name'].count())
    sns.barplot(ax=ax1, x=g.values, y=g.index, orient='h')
    ax1.set_xlabel('Количество обращений')
    ax1.set_ylabel('Категория обращения')
    ax1.set_title('Распределение обращений по категориям')

    # График 2: Распределение по типам горячей линии
    g = df.groupby('hlt_name')['hlt_name'].count().sort_values()
    print(df.groupby('hlt_name')['hlt_name'].count())
    sns.barplot(ax=ax2, x=g.values, y=g.index, orient='h')
    ax2.set_xlabel('Количество обращений')
    ax2.set_ylabel('Тип горячей линии')
    ax2.set_title('Распределение обращений по типам горячей линии')

    plt.show()

    # Сохраняем обработанные данные
    df.to_excel('df_prep.xlsx', index=False)
    print('preprocessed text write')
