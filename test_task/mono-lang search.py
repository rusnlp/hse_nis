import os
import glob
import re

from nltk.corpus import stopwords
stop = stopwords.words('russian')
from pymystem3.mystem import Mystem
myst = Mystem()

from gensim.models import KeyedVectors
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm # визуализация в виде прогресс-бара


    # ПРЕДОБРАБОТКА

# словарь конвертации майстемовских тегов в теги universal
conv = {'A':      'ADJ',
        'ADV':    'ADV',
        'ADVPRO': 'ADV',
        'ANUM':   'ADJ',
        'APRO':   'DET',
        'COM':    'ADJ',
        'CONJ':   'SCONJ',
        'INTJ':   'INTJ',
        'NONLEX': 'X',
        'NUM':    'NUM',
        'PART':   'PART',
        'PR':     'ADP',
        'S':      'NOUN',
        'SPRO':   'PRON',
        'UNKN':   'X',
        'V':      'VERB'}

# Превращаем список токенов в список лемм с pos-тегами (этот формат нужен для предобученной модели с rusvectores)
def lemmas(tokens):
    lems = [] # список лемма_pos
    for token in tokens: # для каждого токена в списке токенов
        if token.isalpha() and token not in stop: # слово, но не стоп-слово
            #lem = myst.lemmatize(token)[0] # альтернативный способ получать леммы, но только их
            #print(lem)
            analiz = myst.analyze(token)[0]  # анализируем слово
            #print(analiz)
            if analiz['analysis']:  # если у майстема есть варианты разбора для токена
                lem = analiz['analysis'][0]['lex'] # берём из анализа лемму (там в по ключу берётся список, в котором только один элемент -- словарь)
                #print(lem)
                gram = re.findall('\w+', analiz['analysis'][0]['gr'])  # получаем все граммемы из анализа
                #print(gram)
                pos = gram[0] # первая граммема -- часть речи
                ud_pos = conv.get(pos) # конвертируем майстемовский тег в формат universal
                #print(pos, ud_pos)
                lems.append('{}_{}'.format(lem, ud_pos)) # сохраняем в список лемма_pos

    return lems

# Предобработка каждого текста в списке файлов и получение списка лемм для каждого
def process_corpus(files):
    corpus = [] # список [текст1[лемма1, лемма2], текст2[лемма1, лемма2] ...]
    for file in tqdm(files): # для каждого файла в списке файлов (с прогресс-баром)
        text = open(file, encoding='utf-8').read().lower() # читаем файл и приводим его к нижнему регистру
        #print(text)
        tokens = [word.replace('́', '') for word in re.findall('\w+́?\w+', text)] # ловим слова (в т.ч. с ударениями) и удаляем ударения, если есть
        #print(tokens)
        # превращаем список токенов в список лемм с pos-тегами
        lems = lemmas(tokens[:5]) #\\ возьмём первые n токенов для скорости
        #print(lems)
        corpus.append(lems) # добавляем список лемм в корпус

    return corpus


    # ВЕКТОРИЗАЦИЯ

# Переводим текст в вектор фиксированной размерности
def vectorize_text(tokens):
    dim = w2v.vector_size # получаем размерность предобученных векторв
    t_vec = np.zeros(dim) # заводим вектор нужной размерности, состоящий из нулей
    for token in tokens: # для каждого токена в списке
        if token in w2v.vocab: # если токен есть в словаре модели
            vec = w2v[token] # получаем вектор для токена из модели
            t_vec = np.vstack((t_vec, vec)) # к старым векторам снизу клеим новый
    t_vec = t_vec[1:]  # отбрасываем нулевую строку (все "настоящие" вектора у нас ниже)
    t_vec = t_vec.mean(axis=0)  # собираем массив векторов в один вектор, выбирая среднее значение в каждом столбце
    # print(t_vec.shape)

    return t_vec # возвращаем np.array размерноти (dim,)

# Составляем список с векторами текстов для всего корпуса
def vectorize_corpus(corpus):
    corpus_vecs = [] # список [[вектор текста 1], [вектор текста 2], [...]]
    for text in corpus: # для каждого текста в корпусе
        t_vec = vectorize_text(text) # получаем вектор для текста
        corpus_vecs.append(t_vec) # добавляем вектор текста в список векторов

    return corpus_vecs


    # ПОИСК

# Для каждого текста в корпусе считаем косинусную близость к данному
def search_similar(text, corpus_vecs): # подаём текст как список токенов и векторизованный корпус
    similars = {} # словарь {индекс текста в корпусе: близость к данному}
    t_vec = vectorize_text(text) # получаем вектор для данного текста
    for i, v in enumerate(corpus_vecs): # для индекса, вектора для каждого текста в корпусе
        # для cosine_similarity придётся измениро размерност векторов (dim,) -> (1, dim) (вместо [0 0 0 0] получаем [[0 0 0 0]])
        sim = cosine_similarity(t_vec.reshape(1, t_vec.shape[0]), v.reshape(1, v.shape[0])) # вычисляем косинусную близость для данного вектора и вектора текста
        similars[i] = sim[0][0] # для индекса текста добавили его близость к данному в словарь
    #print(similars)

    return similars

# Ранжируем тексты по близости и принтим красивый списочек
def rating(similars): # принимаем словарь близостей
    similars = sorted(similars.items(), key=lambda tup: tup[1], reverse=True) # сортируем словарь по значениям в порядке убывания: сортируем список кортежей (key, value) по value
    # similars: [(i, sim), (j, sim), (...)]
    for i, item in enumerate(similars): # для индекса, кортежа в similars
        print('{}. {}.txt ({})'.format(i+1, item[0], item[1])) # принтим текст и его близость #i+1, т.к. нумерация с 0



# получаем список файлов
os.chdir('texts/ruwiki') # заходим в папку с текстами
files = glob.glob('*.txt')[:5] # получаем список файлов txt #\\ для скорости взяли n текстов
#print(files)

corpus = process_corpus(files) # предобрабатываем текст
#print(corpus)
print()

os.chdir('../../models') # заходим в папку с моделью
w2v = KeyedVectors.load_word2vec_format('ru.bin', binary=True, encoding='utf-8') # подгружаем предобученную модель
#print(w2v.vector_size) # посмотреть размерность векторов

corpus_vecs = vectorize_corpus(corpus) # векторизуем корпус
#print(corpus_vecs.shape) # (corpus_size, dim)

similars = search_similar(corpus[0], corpus_vecs) # рассчитываем близости текстов корпуса к данному тексту (взяли первый текст в корпусе)
#print(similars)

rating(similars) # отражировали тексты по близости