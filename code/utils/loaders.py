'''
Загрузка всякой всячины (эмбеддинги, словари, проекции)
'''

from gensim import models
import logging
import numpy as np
import os
import sys
import zipfile
from pickle import load as pload
from json import load as jload

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_embeddings(embeddings_path):  # TODO: пометить, что чужое
    """
    :param embeddings_path: путь к модели эмбеддингов (строка)
    :return: загруженная предобученная модель эмбеддингов (KeyedVectors)
    """
    # Бинарный формат word2vec:
    if embeddings_path.endswith('.bin.gz') or embeddings_path.endswith('.bin'):
        model = models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True,
                                                         unicode_errors='replace')
    # Текстовый формат word2vec:
    elif embeddings_path.endswith('.txt.gz') or embeddings_path.endswith('.txt') \
            or embeddings_path.endswith('.vec.gz') or embeddings_path.endswith('.vec'):
        model = models.KeyedVectors.load_word2vec_format(
            embeddings_path, binary=False, unicode_errors='replace')

    # ZIP-архив из репозитория NLPL:
    elif embeddings_path.endswith('.zip'):
        with zipfile.ZipFile(embeddings_path, "r") as archive:
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
        model = models.KeyedVectors.load(embeddings_path)
        # If you intend to train further: emb_model = models.Word2Vec.load(embeddings_file)

    model.init_sims(replace=True)  # Unit-normalizing the vectors (if they aren't already)
    return model


def load_projection(projection_path):
    projection = np.loadtxt(projection_path, delimiter=',')
    return projection


def load_bidict(bidict_path):
    '''читаем словарь пар в словарь'''
    lines = open(bidict_path, encoding='utf-8').read().splitlines()
    bidict = {line.split()[0]: line.split()[1] for line in lines}
    # print(len(lines))
    return bidict


def load_vectorized(output_vectors_path, forced):
    """загрузка матрицы с векторами корпуса, если есть"""
    # если существует уже какой-то векторизованный корпус
    if os.path.isfile(output_vectors_path) and not forced:
        vectorized = pload(open(output_vectors_path, 'rb'))
        print('Уже что-то векторизовали!', file=sys.stderr)

    else:  # ничего ещё из этого корпуса не векторизовали или принудительно обновляем всё
        print('Ничего ещё не разбирали, сейчас будем.', file=sys.stderr)
        vectorized = []

    return vectorized


def load_mapping(mapping_path):
    """ключи маппинга делаем снова числами"""
    raw_mapping = jload(open(mapping_path))
    mapping = {}
    for dict_name in raw_mapping:
        map_dict = raw_mapping[dict_name]
        # print(map_dict)
        if dict_name.endswith('i'):  # маппинг названий в индексы
            mapping[dict_name] = {k: int(v) for k, v in map_dict.items()}
        elif dict_name.startswith('i'):  # маппинг индексов в названия
            mapping[dict_name] = {int(k): v for k, v in map_dict.items()}
    return mapping


def load_article_data(article_data_path):
    '''получаем словарь хеш: название, ссылка'''
    lines = open(article_data_path, encoding='utf-8').read().splitlines()
    article_data = {line.split('\t')[0]:
                    {'real_title': line.split('\t')[1], 'url': line.split('\t')[2]} for line in lines}
    # print(article_data)
    return article_data


