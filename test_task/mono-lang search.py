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
from json import load

ru_model_path = 'models//ru_syntagrus.udpipe'
ru_model = Model.load(ru_model_path)
ru_process_pipeline = Pipeline(ru_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')


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
    corpus = []  # список [текст1[лемма1, лемма2], текст2[лемма1, лемма2] ...]
    for file in tqdm(files):  # для каждого файла в списке файлов (с прогресс-баром)
        text = open(file,
                    encoding='utf-8').read().lower().strip().splitlines()  # читаем файл и приводим его к нижнему регистру
        # превращаем список токенов в список лемм с pos-тегами # как обрабатывает слова с ударениями?
        lems = []  # придётся экстендить, поэтому без генератора \\есть способ?
        for line in text:
            line_lems = process(ru_process_pipeline, line, keep_pos=True, keep_punct=False)
            if line_lems:  # если не пустая строка
                lems.extend(line_lems)
        corpus.append(lems)  # добавляем список лемм в корпус
    return corpus
    # ВЕКТОРИЗАЦИЯ


def vectorize_text(tokens, w2v): #\\векторизуем по всем словам, даже если повторяются, или по множетсву?
    words = [token for token in tokens if token in w2v] # составляем список токенов, которые есть в модели
    if not words: # если ни одного слова нет в модели
        print('Empty lexicon in', tokens, file=sys.stderr)
        return np.zeros(w2v.vector_size) # возвращаем нули
    t_vecs = np.zeros((len(words), w2v.vector_size)) # заводим матрицу нужной размерности (кол-во слов, размерность предобученных векторв), состоящую из нулей
    for i, token in enumerate(words): # для каждого слова
        t_vecs[i, :] = vec = w2v.wv.get_vector(token) # получаем вектор из модели
    t_vec = np.sum(t_vecs, axis=0)  # суммируем вектора по столбцам
    t_vec = np.divide(t_vec, len(words))  # Computing average vector
    return t_vec # возвращаем np.array размерноти (dim,)


# Составляем список с векторами текстов для всего корпуса
def vectorize_corpus(corpus):
    corpus_vecs = [] # список [[вектор текста 1], [вектор текста 2], [...]]
    for text in corpus: # для каждого текста в корпусе
        t_vec = vectorize_text(text, w2v) # получаем вектор для текста
        corpus_vecs.append(t_vec) # добавляем вектор текста в список векторов
    return corpus_vecs


    # ПОИСК
# Для каждого текста в корпусе считаем косинусную близость к данному
def search_similar(text, corpus_vecs): # подаём текст как список токенов и векторизованный корпус
    similars = {} # словарь {индекс текста в корпусе: близость к данному}
    t_vec = vectorize_text(text, w2v) # получаем вектор для данного текста
    for i, v in enumerate(corpus_vecs):
       # для индекса, вектора для каждого текста в корпусе
        # для cosine_similarity придётся измениро размерност векторов (dim,) -> (1, dim) (вместо [0 0 0 0] получаем [[0 0 0 0]])
        sim = cosine_similarity(t_vec.reshape(1, t_vec.shape[0]), v.reshape(1, v.shape[0])) # вычисляем косинусную близость для данного вектора и вектора текста
        similars[i] = sim[0][0] # для индекса текста добавили его близость к данному в словарь
    #print(similars)
    return similars


# Ранжируем тексты по близости и принтим красивый списочек
def rating(similars): # принимаем словарь близостей
    sims = sorted(similars, key=similars.get, reverse=True) # сортируем словарь по значениям в порядке убывания: сортируем список кортежей (key, value) по value
    # similars: [(i, sim), (j, sim), (...)]
    similars_list = []
    for i, item in enumerate(sims):
        similars_list += [('{}. {} ({})'.format(i, file_names[item], similars[item]))]
    print(*(similars_list[1:11]))
    return 0


if __name__ == "__main__":
    target_article_title = input('Enter an article title ')
    texts_mapping = load(open('texts/titles_mapping.json'))
    files = ['texts/ruwiki_named/{}.txt'.format(texts_mapping['i2ru'].get(str(i))) for i in
             range(53)]  # получаем список файлов
    file_names = [texts_mapping['i2ru'].get(str(i)) for i in range(53)]
    # print(files)

    corpus = process_corpus(files)  # предобрабатываем текст

    w2v = KeyedVectors.load_word2vec_format('models/rusvect.bin', binary=True,
                                            encoding='utf-8')  # подгружаем предобученную модель
    # print(w2v.vector_size) # посмотреть размерность векторов
    corpus_vecs = vectorize_corpus(corpus)  # векторизуем корпус
    # print(corpus_vecs.shape) # (corpus_size, dim)
    target_article = corpus[file_names.index(target_article_title)]
    similars = search_similar(target_article, corpus_vecs)
    rating(similars) # отранжировали тексты по близости
