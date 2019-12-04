'''
Принимаем на вход язык, путь к папке с корпусом, путь к модели с векторами и тип модели (пока есть только simple)
Сохраняем в той же папке pkl со списком векторов

python vectorize_corpus.py ru texts/ruwiki texts/titles_mapping.json models/ru.bin simple
python vectorize_corpus.py en texts/enwiki texts/titles_mapping.json models/en.bin simple
'''

import numpy as np
from gensim.models import KeyedVectors
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from json import load as jload, dump as jdump
import sys
import os
from pickle import dump as pdump
import argparse
from tqdm import tqdm

# TODO: векторизовать на выбор: либо по списку слов, либо по множеству
def vectorize_text(tokens, w2v):
    words = [token for token in tokens if token in w2v]  # составляем список токенов, которые есть в модели
    if not words:  # если ни одного слова нет в модели
        print('Я ничего не знаю из этих токенов: {}'.format(tokens), file=sys.stderr)
        return np.zeros(w2v.vector_size)  # возвращаем нули
    t_vecs = np.zeros((len(words), w2v.vector_size))  # заводим матрицу нужной размерности (кол-во слов, размерность предобученных векторв), состоящую из нулей
    for i, token in enumerate(words):  # для каждого слова
        t_vecs[i, :] = w2v.get_vector(token)  # получаем вектор из модели
    t_vec = np.sum(t_vecs, axis=0)  # суммируем вектора по столбцам
    t_vec = np.divide(t_vec, len(words))  # Computing average vector
    return t_vec  # возвращаем np.array размерноти (dim,)


# Составляем список с векторами текстов для всего корпуса
def vectorize_corpus(corpus, w2v):
    #TODO: делать np.array?
    corpus_vecs = [vectorize_text(text, w2v) for text in tqdm(corpus)]
    # список [[вектор текста 1], [вектор текста 2], [...]]
    return corpus_vecs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Векторизация корпуса и сохранение его в pkl в папке с текстами')
    parser.add_argument('lang', type=str, help='Язык, для которго разбираем, нужен для определения словаря в маппинге (ru/en)')
    parser.add_argument('texts_path', type=str, help='Папка, в которой лежат тексты в формате txt')
    parser.add_argument('mapping_path', type=str, help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('model_path', type=str, help='Папка, в которой лежит модель для векторизации корпуса')
    parser.add_argument('model_type', type=str, help='Краткое имя модели векторизации, чтобы не путать файлы. Будет использовано как имя pkl')
    args = parser.parse_args()

    i2lang = 'i2{}'.format(args.lang)
    lang2i = '{}2i'.format(args.lang)
    output_vecs_path = '{}/{}.pkl'.format(args.texts_path, args.model_type)
    lemmatized_path = '{}/lemmatized.json'.format(args.texts_path)

    # собираем лемматизированные тексты из lemmatized
    if not os.path.isfile(lemmatized_path):  # ничего ещё из этого корпуса не разбирали
        print('Ничего ещё не разбирали, нечего векторизовать!\nПожалуйста, сначала лемматизируйте тексты', file=sys.stderr)
        raise(SystemExit(0)) # заканчиваем работу
    else: # если существует уже разбор каких-то файлов
        lemmatized = jload(open(lemmatized_path, encoding='utf-8'))
        print('Понял, сейчас векторизуем', file=sys.stderr)

        #TODO: не векторизовать то, что уже есть, к этому добавлять новое
        #TODO: хранить вектора слоарём, не списком?

        # получаем список файлов
        texts_mapping = jload(open(args.mapping_path))
        #print(texts_mapping)

        corpus = []
        #print(texts_mapping[i2lang])
        for i in range(len(texts_mapping[i2lang])): # для каждого номера в маппинге
        # порядок текстов -- как в индексах
        #TODO: сделать генераторм?
            title = texts_mapping[i2lang].get(str(i))
            #print(title)
            text = lemmatized[title] # по номеру из маппинга берём название и находим его в леммах
            #print(text)
            corpus.append(text)
        corpus = np.array(corpus) # \\будет так меньше места занимать, чем список?
        #print(corpus.shape) #(54,)

        if args.model_path.split('.')[-1] == 'bin':
            #TODO: может быть ещё бинарник в архиве
            binary = True
        else:
            binary = False
        emb_model = KeyedVectors.load_word2vec_format(args.model_path, binary=binary)
        w2v = emb_model.wv  # TODO: у Пети был deprecation warning

        corpus_vecs = vectorize_corpus(corpus, w2v)  # векторизуем корпус
        #print(*(corpus_vecs))
        #print(len(corpus_vecs))

        pdump(corpus_vecs, open(output_vecs_path, 'wb'))