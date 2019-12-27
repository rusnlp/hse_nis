"""
Принимаем на вход язык, путь к папке с корпусом, путь к модели с векторами и путь, куда сохранять векторизованный корпус
(пока есть только simple)
Сохраняем pkl со списком векторов

python vectorize_corpus.py --lang=ru --lemmatized_path=texts/ruwiki/lemmatized.json --mapping_path=texts/titles_mapping.json --model_embeddings_path=models/ru.bin --output_embeddings_path=texts/ruwiki/simple.pkl
python vectorize_corpus.py --lang=en --lemmatized_path=texts/enwiki/lemmatized.json --mapping_path=texts/titles_mapping.json --model_embeddings_path=models/en.bin --output_embeddings_path=texts/enwiki/simple.pkl
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Векторизация корпуса и сохранение его в pkl')
    parser.add_argument('--lang', type=str, required=True,
                        help='Язык, для которго разбираем, нужен для определения словаря в маппинге (ru/en)')
    parser.add_argument('--lemmatized_path', type=str, required=True,
                        help='Путь к файлу json, в котором хранятся лемматизированные файлы')
    parser.add_argument('--mapping_path', type=str, required=True,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('--model_embeddings_path', type=str, required=True,
                        help='Путь к модели для векторизации корпуса')
    parser.add_argument('--output_embeddings_path', type=str, required=True,
                        help='Путь к файлу pkl, в котором будет лежать векторизованный корпус')
    parser.add_argument('--no_duplicates', type=int, default=0,
                        help='Брать ли для каждого типа в тексте вектор только по одному разу (0|1; default: 0)')

    return parser.parse_args()


class NotLemmatizedError(Exception):
    def __init__(self):
        self.text = 'Нечего векторизовать! Пожалуйста, сначала лемматизируйте тексты'

    def __str__(self):
        return repr(self.text)


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


def vectorize_text(tokens, w2v, no_duplicates):
    # возвращаем нормализованный np.array размерноти (dim,)
    words = [token for token in tokens if token in w2v]
    if not words:
        print('Я ничего не знаю из этих токенов: {}'.format(tokens), file=sys.stderr)
        return np.zeros(w2v.vector_size)  # возвращаем нули
    if no_duplicates:
        words = set(words)
    # заводим матрицу нужной размерности (кол-во слов, размерность предобученных векторв), состоящую из нулей
    t_vecs = np.zeros((len(words), w2v.vector_size))
    for i, token in enumerate(words):
        t_vecs[i, :] = w2v[token]
    t_vec = np.sum(t_vecs, axis=0)
    t_vec = np.divide(t_vec, len(words))
    vec = t_vec / norm(t_vec)
    return vec


# Составляем список с векторами текстов для всего корпуса
def vectorize_corpus(corpus, w2v, emb_dim, output_vecs_path, no_duplicates):
    # матрица [[вектор текста 1], [вектор текста 2], [...]]
    not_vectorized = []
    vectors = np.zeros((len(corpus), emb_dim))
    for i, text in tqdm(enumerate(corpus)):
        vector = vectorize_text(text, w2v, no_duplicates)
        if len(vector) != 0:  # для np.array нельзя просто if
            vectors[i, :] = vector
            pdump(vectors, open(output_vecs_path, 'wb'))
        else:
            not_vectorized.append(i)
            continue
    return vectors, not_vectorized


def main():
    args = parse_args()

    i2lang = 'i2{}'.format(args.lang)

    # собираем лемматизированные тексты из lemmatized
    if not os.path.isfile(args.lemmatized_path):  # ничего ещё из этого корпуса не разбирали
        raise NotLemmatizedError()

    else:  # если существует уже разбор каких-то файлов
        lemmatized = jload(open(args.lemmatized_path, encoding='utf-8'))
        print('Понял, сейчас векторизуем', file=sys.stderr)

        # TODO: не векторизовать то, что уже есть, к этому добавлять новое

        # получаем список файлов
        texts_mapping = jload(open(args.mapping_path))

        lemmatized_corpus = []
        for nr in range(len(texts_mapping[i2lang])):  # для каждого номера в маппинге
            # порядок текстов -- как в индексах
            title = texts_mapping[i2lang].get(str(nr))
            # по номеру из маппинга берём название и находим его в леммах
            lemmatized_text = lemmatized[title]
            lemmatized_corpus.append(lemmatized_text)

        emb_model = load_embeddings(args.model_embeddings_path)
        emb_dim = emb_model.vector_size
        emb_model = emb_model.wv

        _, not_vectorized = vectorize_corpus(lemmatized_corpus, emb_model, emb_dim, args.output_embeddings_path,
                                                                                    args.no_duplicates)
        if not_vectorized:
            print('Не удалось векторизовать следующие тексты:\n{}'.format('\n'.join(not_vectorized)), file=sys.stderr)


if __name__ == "__main__":
    main()
