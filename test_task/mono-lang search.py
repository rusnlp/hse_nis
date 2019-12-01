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


def get_udpipe_lemmas(string, lang, keep_pos=True, keep_punct=False):
    # принимает строку, возвращает список токенов
    res = unify_sym(string.strip())
    if lang == 'ru':
        output = process(ru_process_pipeline, res, keep_pos, keep_punct)
    return output
# Превращаем список токенов в список лемм с pos-тегами (этот формат нужен для предобученной модели с rusvectores)


# def lemmas(tokens): #! вообще больше не нужна, т.к. у нас это udpipe делает

def process_corpus(files):
    corpus = []  # список [текст1[лемма1, лемма2], текст2[лемма1, лемма2] ...]
    for file in tqdm(files):  # для каждого файла в списке файлов (с прогресс-баром)
        text = open(file,
                    encoding='utf-8').read().lower().strip().splitlines()  # читаем файл и приводим его к нижнему регистру
        # превращаем список токенов в список лемм с pos-тегами # как обрабатывает слова с ударениями?
        lems = []  # придётся экстендить, поэтому без генератора \\есть способ?
        for line in text:
            line_lems = get_udpipe_lemmas(line, lang='ru') #! идём через get_udpipe_lemmas, т.к. он и унификацию символов сделает, и вызов у него покороче
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
    corpus_vecs = [vectorize_text(text, w2v) for text in corpus] #! Нам и Ира, и Андрей сказали, что лучше генераторами
    # список [[вектор текста 1], [вектор текста 2], [...]]
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
def print_rating(similars): # принимаем словарь близостей #! Ира просила переименовать
    sims = sorted(similars, key=similars.get, reverse=True) # сортируем словарь по значениям в порядке убывания: сортируем список кортежей (key, value) по value
    # similars: [(i, sim), (j, sim), (...)]
    similars_list = []
    for i, item in enumerate(sims):
        similars_list += [('{}. {} ({})'.format(i, texts_mapping['i2ru'].get(str(item)), similars[item]))]
    print(*(similars_list[1:11]))
    #! return 0 не надо, это нормально, когда функция ничего не возвращает. А так мы теряем возможность отследить ошибку, потому что 0 может получаться в результате работы функции, а так бы возвращался None


if __name__ == "__main__":
    target_article_title = input('Введите название русской статьи: ').lower() #! пусть сразу будет ясно, для какого языка поиск #! и на всякий приведём к нижнему регистру
    texts_mapping = load(open('texts/titles_mapping.json'))

    files = ['texts/ruwiki_named/{}.txt'.format(texts_mapping['i2ru'].get(str(i))) for i in
             range(len(texts_mapping['i2ru']))]  # получаем список файлов #! длина может быть не обязательно 54

    # file_names = [texts_mapping['i2ru'].get(str(i)) for i in range(53)] #! не нужен, у нас есть маппинг, чтобы по названию получать индекс
    # print(files)

    corpus = process_corpus(files)  # предобрабатываем текст

    w2v = KeyedVectors.load_word2vec_format('models/rusvect.bin', binary=True,
                                            encoding='utf-8')  # подгружаем предобученную модель
    corpus_vecs = vectorize_corpus(corpus)  # векторизуем корпус
    # print(corpus_vecs.shape) # (corpus_size, dim)
    target_article = corpus[texts_mapping['ru2i'].get(target_article_title)] #! убрала file_names, он избыточен #! добавила txt, чтобы не приходилось каждый раз в названии его прописывать
    similars = search_similar(target_article, corpus_vecs)
    print_rating(similars) # отранжировали тексты по близости


    def the_closest_article(similars):  # функция для оценки качества
        sims = sorted(similars, key=similars.get, reverse=True)
        pred_id = str(sims[1])  # на 0 месте будет сама статья
        the_closest = texts_mapping['i2ru'].get(pred_id)
        return the_closest


    true_similar = {'аксон': 'кома', 'аллергия': 'холера', 'алюминий': 'бериллий', 'анаграмма': 'анахронизм',
                    'анахронизм': 'анаграмма', 'анемия': 'кровь', 'арабское письмо': 'анаграмма',
                    'артерия': 'кровь', 'беневенто': 'валентиниан i', 'бензин': 'алюминий', 'берилл': 'геология',
                    'бериллий': 'аллюминий', 'биом': 'экосистема', 'биосфера': 'экосистема',
                    'биотехнология': 'химия', 'блог': 'арабское письмо', 'булева алгебра': 'химия',
                    'бумеранг': 'гравитация', 'валентиниан i': 'валентиниан ii', 'валентиниан ii': 'валентиниан i',
                    'валентиниан iii': 'валентиниан i', 'водолей (созвездие)': 'евангелие от иоанна',
                    'геология': 'геоморфология', 'гравитация': 'бумеранг', 'группа крови': 'кровь',
                    'евангелие от иоанна': 'евангелие от луки', 'евангелие от луки': 'евангелие от марка',
                    'евангелие от марка': 'евангелие от иоанна', 'жасмин': 'эвкалипт', 'индейки': 'кошка',
                    'история игрушек': 'водолей (созвездие)', 'калория': 'холестерин', 'кома': 'эпилепсия',
                    'кошка': 'песец', 'кровь': 'кровяное давление', 'кровяное давление': 'кровь', 'лес':
                        'экосистема', 'песец': 'кошка', 'плод': 'жасмин', 'спирты': 'химия', 'фоссилии': 'геология',
                    'химическая реакция': 'химия', 'химический элемент': 'химия',
                    'химия': 'химический элемент', 'холера': 'аллергия', 'холестерин': 'калории', 'эвкалипт': 'лес',
                    'экосистема': 'биом', 'электрический двигатель': 'бензин', 'электролиз': 'химия',
                    'электрон': 'электролиз', 'эпидемиология': 'холера', 'эпилепсия': 'кома'}
    correct = 0
    wrong = 0
    for key in true_similar:
        target_article_title = key
        target_article = corpus[texts_mapping['ru2i'].get(target_article_title)]
        similars = search_similar(target_article, corpus_vecs)
        print(target_article_title, the_closest_article(similars), true_similar[key])
        if the_closest_article(similars) == true_similar[key]:
            correct += 1
        else:
            wrong += 1
    print('Correct: ', correct / len(true_similar), 'Wrong: ', wrong / len(true_similar))

#TODO: пусть работает и на русском, и на английском