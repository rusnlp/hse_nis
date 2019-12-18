"""
Принимаем на вход язык, путь к папке с корпусом, путь к модели с векторами и тип модели
(пока есть только simple)
Сохраняем в той же папке pkl со списком векторов

python vectorize_corpus.py ru texts/ruwiki texts/titles_mapping.json models/ru.bin simple
python vectorize_corpus.py en texts/enwiki texts/titles_mapping.json models/en.bin simple
"""

import logging
import argparse
import os
import sys
import zipfile
from json import load as jload
from pickle import dump as pdump

import numpy as np
from numpy.linalg import norm
from gensim import models
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_embeddings(embeddings_file):
    # по расширению определяем формат модели
    # Бинарный формат word2vec:
    if embeddings_file.endswith('.bin.gz') or embeddings_file.endswith('.bin'):
        model = models.KeyedVectors.load_word2vec_format(embeddings_file, binary=True,
                                                         unicode_errors='replace')
    # Текстовый формат word2vec:
    elif embeddings_file.endswith('.txt.gz') or embeddings_file.endswith('.txt') \
            or embeddings_file.endswith('.vec.gz') or embeddings_file.endswith('.vec'):
        model = models.KeyedVectors.load_word2vec_format(
            embeddings_file, binary=False, unicode_errors='replace')

    # ZIP-архив из репозитория NLPL:
    elif embeddings_file.endswith('.zip'):
        with zipfile.ZipFile(embeddings_file, "r") as archive:
            # Loading and showing the metadata of the model:
            # metafile = archive.open('meta.json')
            # metadata = json.loads(metafile.read())
            # for key in metadata:
            #    print(key, metadata[key])
            # print('============')

            # Загрузка самой модели:
            stream = archive.open("model.bin")  # или model.txt, чтобы взглянуть на модель
            model = models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors='replace')
    else:
        # Native Gensim format?
        model = models.KeyedVectors.load(embeddings_file)
        # If you intend to train further: emb_model = models.Word2Vec.load(embeddings_file)

    model.init_sims(replace=True)  # Unit-normalizing the vectors (if they aren't already)
    return model


# TODO: векторизовать на выбор: либо по списку слов, либо по множеству
def vectorize_text(tokens, w2v):
    # составляем список токенов, которые есть в модели
    words = [token for token in tokens if token in w2v]
    if not words:  # если ни одного слова нет в модели
        print('Я ничего не знаю из этих токенов: {}'.format(tokens), file=sys.stderr)
        return np.zeros(w2v.vector_size)  # возвращаем нули
    # заводим матрицу нужной размерности (кол-во слов, размерность предобученных векторв),
    # состоящую из нулей
    t_vecs = np.zeros((len(words), w2v.vector_size))
    for i, token in enumerate(words):  # для каждого слова
        t_vecs[i, :] = w2v.get_vector(token)  # получаем вектор из модели
    t_vec = np.sum(t_vecs, axis=0)  # суммируем вектора по столбцам
    t_vec = np.divide(t_vec, len(words))  # считаем средний вектор

    return t_vec  # возвращаем np.array размерноти (dim,)


# Составляем список с векторами текстов для всего корпуса
def vectorize_corpus(corpus, w2v, emb_dim):
    # TODO: делать np.array?
    vectors = np.zeros((len(corpus), emb_dim))
    for i, text in tqdm(enumerate(corpus)):
        vectors[i, :] = vectorize_text(text, w2v)
    # матрица [[вектор текста 1], [вектор текста 2], [...]]
    vectors = vectors/norm(vectors, axis=1, keepdims=True) # нормализуем матрицу
    return vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Векторизация корпуса и сохранение его в pkl в папке с текстами')
    parser.add_argument('lang', type=str,
                        help='Язык, для которго разбираем, нужен для определения словаря в маппинге (ru/en)')
    parser.add_argument('texts_path', type=str,
                        help='Папка, в которой лежат тексты в формате txt')
    parser.add_argument('mapping_path', type=str,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('model_path', type=str,
                        help='Папка, в которой лежит модель для векторизации корпуса')
    parser.add_argument('model_type', type=str,
                        help='Краткое имя модели векторизации, чтобы не путать файлы. '
                             'Будет использовано как имя pkl')
    args = parser.parse_args()

    i2lang = 'i2{}'.format(args.lang)
    lang2i = '{}2i'.format(args.lang)
    output_vecs_path = '{}/{}.pkl'.format(args.texts_path, args.model_type)
    lemmatized_path = '{}/lemmatized.json'.format(args.texts_path)

    # собираем лемматизированные тексты из lemmatized
    if not os.path.isfile(lemmatized_path):  # ничего ещё из этого корпуса не разбирали
        print('Ничего ещё не разбирали, нечего векторизовать!'
              '\nПожалуйста, сначала лемматизируйте тексты', file=sys.stderr)
        raise (SystemExit(0))  # заканчиваем работу
    else:  # если существует уже разбор каких-то файлов
        lemmatized = jload(open(lemmatized_path, encoding='utf-8'))
        print('Понял, сейчас векторизуем', file=sys.stderr)

        # TODO: не векторизовать то, что уже есть, к этому добавлять новое
        # TODO: хранить вектора словарём, не списком?

        # получаем список файлов
        texts_mapping = jload(open(args.mapping_path))
        # print(texts_mapping)

        lemmatized_corpus = []
        # print(texts_mapping[i2lang])
        for nr in range(len(texts_mapping[i2lang])):  # для каждого номера в маппинге
            # порядок текстов -- как в индексах
            # TODO: сделать генератором?
            title = texts_mapping[i2lang].get(str(nr))
            # print(title)
            # по номеру из маппинга берём название и находим его в леммах
            lemmatized_text = lemmatized[title]
            # print(text)
            lemmatized_corpus.append(lemmatized_text)

        emb_model = load_embeddings(args.model_path)
        emb_dim = emb_model.vector_size
        emb_model = emb_model.wv  # TODO: у Пети был deprecation warning

        corpus_vecs = vectorize_corpus(lemmatized_corpus, emb_model, emb_dim)  # векторизуем корпус
        # print(*(corpus_vecs))
        # print(len(corpus_vecs))

        pdump(corpus_vecs, open(output_vecs_path, 'wb'))
