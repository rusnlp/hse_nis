"""
Принимаем на вход язык, путь к папке с корпусом, путь к модели с векторами и путь, куда сохранять векторизованный корпус
Сохраняем pkl со списком векторов

python vectorize_corpus.py --lang=ru --lemmatized_path=texts/ruwiki/lemmatized.json --mapping_path=texts/titles_mapping.json --model_embeddings_path=models/ru.bin --output_embeddings_path=texts/ruwiki/simple.pkl
python vectorize_corpus.py --lang=en --lemmatized_path=texts/enwiki/lemmatized.json --mapping_path=texts/titles_mapping.json --model_embeddings_path=models/en.bin --output_embeddings_path=texts/enwiki/simple.pkl
"""

import argparse
import logging
import os
import sys
import zipfile
from json import load as jload
from pickle import dump as pdump, load as pload

from gensim import models
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def parse_args():
    """
    :return: объект со всеми аршументами (argparse.Namespace)
    """
    parser = argparse.ArgumentParser(
        description='Векторизация корпуса и сохранение матрицы нормализованных векторов в pkl')
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
    parser.add_argument('--forced', type=int, default=0,
                        help='Принудительно векторизовать весь корпус заново (0|1; default: 0)')

    return parser.parse_args()


class NotLemmatizedError(Exception):
    """
    По указанному пути не нашёлся json с лемматизированными тестами
    """
    def __init__(self):
        self.text = 'Нечего векторизовать! Пожалуйста, сначала лемматизируйте тексты'

    def __str__(self):
        return self.text


def create_lemmatized_corpus(mapping, i2lang_name, lemmatized, n_new):
    """
    :param mapping: маппинг текстов в индексы и наоборот (словарь словарей)
    :param i2lang_name: к какому словарю оращаться в маппинге (строка)
    :param lemmatized: заголовки и лемматизированные тексты (словарь)
    :param n_new: количество новых текстов (int)
    :return: леммы новых текстов в порядке из маппинга (список списков)
    """
    corpus = []
    # для каждого номера в маппинге от последнего в vectorized
    for nr in range(len(mapping[i2lang_name]) - n_new, len(mapping[i2lang_name])):
        # порядок текстов -- как в индексах
        title = mapping[i2lang_name].get(str(nr))
        # по номеру из маппинга берём название и находим его в леммах
        lemmatized_text = lemmatized[title]
        corpus.append(lemmatized_text)
    return corpus


def load_vectorized(output_embeddings_path, forced):
    """
    :param output_embeddings_path: путь к векторизованному корпусу (строка)
    :param forced: принудительная векторизация всех текстов заново (pseudo-boolean int)
    :return: матрица нормированных векторов или пустой список, если ничего ещё не векторизовали (np.array/список)
    """
    # если существует уже какой-то векторизованный корпус
    if os.path.isfile(output_embeddings_path) and not forced:
        vectorized = pload(open(output_embeddings_path, 'rb'))
        print('Уже что-то векторизовали!', file=sys.stderr)

    else:  # ничего ещё из этого корпуса не векторизовали или принудительно обновляем всё
        print('Ничего ещё не разбирали, сейчас будем.', file=sys.stderr)
        vectorized = []

    return vectorized


def load_embeddings(embeddings_file):
    """
    :param embeddings_file: путь к модели эмбеддингов (строка)
    :return: загруженная предобученная модель эмбеддингов (KeyedVectors)
    """
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
    """
    :param tokens: токены текста (список строк)
    :param w2v: модель векторизации (keyedvectors.wv)
    :param no_duplicates: векторизовать по множеству слов, а не по спику (pseudo-boolean int)
    :return: нормализованный вектор (np.array)
    """
    words = [token for token in tokens if token in w2v]

    if not words:
        print('Я ничего не знаю из этих токенов: {}'.format(tokens), file=sys.stderr)
        return np.zeros(w2v.vector_size)

    if no_duplicates:
        words = set(words)

    t_vecs = np.zeros((len(words), w2v.vector_size))

    for i, token in enumerate(words):
        t_vecs[i, :] = w2v[token]

    t_vec = np.sum(t_vecs, axis=0)
    t_vec = np.divide(t_vec, len(words))
    vec = t_vec / norm(t_vec)

    return vec


def vectorize_corpus(corpus, vectors, w2v, no_duplicates, starts_from=0):
    """
    :param corpus: токенизированные тексты (список списков строк)
    :param w2v: модель векторизации (keyedvectors.wv)
    :param no_duplicates: векторизовать по множеству слов, а не по спику (pseudo-boolean)
    :return vectors: матрица нормализованных векторов (np.array)
    :return not_vectorized: индексы текстов, которые не удалось векторизовать (список)
    """
    not_vectorized = []

    for i, text in tqdm(enumerate(corpus)):
        vector = vectorize_text(text, w2v, no_duplicates)

        if len(vector) != 0:  # для np.array нельзя просто if
            vectors[starts_from + i, :] = vector[:] # дописывам вектора новых текстов в конец

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
        lemmatized_dict = jload(open(args.lemmatized_path, encoding='utf-8'))
        print('Понял, сейчас векторизуем.', file=sys.stderr)

        # получаем список файлов
        texts_mapping = jload(open(args.mapping_path))

        old_vectorized = load_vectorized(args.output_embeddings_path, args.forced)

        n_new_texts = len(texts_mapping[i2lang]) - len(old_vectorized)  # появились ли новые номера в маппинге
        print('Новых текстов: {}'.format(n_new_texts), file=sys.stderr)

        if n_new_texts:
            lemmatized_corpus = create_lemmatized_corpus(texts_mapping, i2lang, lemmatized_dict, n_new_texts)

            emb_model = load_embeddings(args.model_embeddings_path)
            emb_dim = emb_model.vector_size
            emb_model = emb_model.wv

            # за размер нового корпуса принимаем длину маппинга
            new_vectorized = np.zeros((len(texts_mapping[i2lang]), emb_dim))

            # заполняем старые строчки, если они были
            for i, line in enumerate(old_vectorized):
                new_vectorized[i, :] = line

            new_vectorized, not_vectorized = vectorize_corpus(lemmatized_corpus, new_vectorized,
                                                              emb_model, args.no_duplicates,
                                                              starts_from=len(old_vectorized))
            pdump(new_vectorized, open(args.output_embeddings_path, 'wb'))

            if not_vectorized:
                print('Не удалось векторизовать следующие тексты:\n{}'.format('\n'.join(not_vectorized)), file=sys.stderr)


if __name__ == "__main__":
    main()
