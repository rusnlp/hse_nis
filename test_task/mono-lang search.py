import os
import glob
import re
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import sys
from preprocess import unify_sym, process #https://github.com/akutuzov/webvectors/blob/master/preprocessing/rus_preprocessing_udpipe.py
from ufal.udpipe import Model, Pipeline
from nltk.corpus import stopwords
stop = stopwords.words('russian')
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm # визуализация в виде прогресс-бара
from ufal.udpipe import Model, Pipeline


ru_model_path = 'models\\ru_syntagrus.udpipe'
ru_model = Model.load(ru_model_path)
ru_process_pipeline = Pipeline(ru_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')



    # ПРЕДОБРАБОТКА

def udpipe_process(string, lang, keep_pos=True, keep_punct=False):
    # принимает строку, возвращает список токенов
    res = unify_sym(string.strip())
    if lang == 'ru':
        output = process(ru_process_pipeline, res, keep_pos, keep_punct)
    return output


# Превращаем список токенов в список лемм с pos-тегами (этот формат нужен для предобученной модели с rusvectores)
def lemmas(tokens):
    lems = [] # список лемма_pos
    for token in tokens: # для каждого токена в списке токенов
        if token.isalpha() and token not in stop: # слово, но не стоп-слово
            analiz = udpipe_process(token, lang='ru', keep_pos=True)  # анализируем слово
            lems += analiz
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
        lems = lemmas(tokens)
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
        t_vec = vectorize_text(text) # получаем вектор для текста'
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
    similars_list = []
    for i, item in enumerate(similars):
        similars_list += [('{}. {}.txt ({})'.format(i, item[0], item[1]))]
    print(*(similars_list[1:11]))
    return 0


# получаем список файлов
os.chdir('texts/ruwiki_named') # заходим в папку с текстами
files = glob.glob('*.txt') # получаем список файлов txt #\\ для скорости взяли n текстов
#print(files)

corpus = process_corpus(files) # предобрабатываем текст
#print(corpus)

os.chdir('../../models') # заходим в папку с моделью
w2v = KeyedVectors.load_word2vec_format('rusvect.bin', binary=True, encoding='utf-8') # подгружаем предобученную модель
#print(w2v.vector_size) # посмотреть размерность векторов
corpus_vecs = vectorize_corpus(corpus) # векторизуем корпус
#print(corpus_vecs.shape) # (corpus_size, dim)


similars = search_similar(corpus[0], corpus_vecs) # рассчитываем близости текстов корпуса к данному тексту (взяли первый текст в корпусе)

if __name__ == "__main__":
    rating(similars) # отражировали тексты по близости