#!/usr/bin/python3.6
# coding: utf-8

"""
Запускала без командной строки, просто скрипт в пайчарме.
Нужны русская и английская модели и двуязычный словарь ru-en_lem.txt
https://github.com/ltgoslo/diachronic_armed_conflicts/blob/master/helpers.py
"""

import numpy as np
from tqdm import tqdm
import logging
import zipfile
import json
from gensim import models
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def load_embeddings(modelfile):
    if modelfile.endswith('.txt.gz') or modelfile.endswith('.txt'):
        model = models.KeyedVectors.load_word2vec_format(modelfile, binary=False,
                                                         unicode_errors='replace')
    elif modelfile.endswith('.bin.gz') or modelfile.endswith('.bin'):
        model = models.KeyedVectors.load_word2vec_format(modelfile, binary=True,
                                                         unicode_errors='replace')
    # ZIP archive from the NLPL vector repository:
    elif modelfile.endswith('.zip'):
        with zipfile.ZipFile(modelfile, "r") as archive:
            # Loading and showing the metadata of the model:
            metafile = archive.open('meta.json')
            metadata = json.loads(metafile.read())
            for key in metadata:
                print(key, metadata[key])
            print('============')

            # Loading the model itself:
            stream = archive.open("model.bin")  # or model.txt, if you want to look at the model
            model = models.KeyedVectors.load_word2vec_format(stream, binary=True,
                                                             unicode_errors='replace')
    else:
        # model = models.Word2Vec.load(modelfile)
        model = models.KeyedVectors.load(modelfile)  # For newer models
    model.init_sims(replace=True)
    return model


def normalequation(data, target, lambda_value, vector_size):
    regularizer = 0
    if lambda_value != 0:  # Regularization term
        regularizer = np.eye(vector_size + 1)
        regularizer[0, 0] = 0
        regularizer = np.mat(regularizer)
    # Normal equation:
    theta = np.linalg.pinv(data.T * data + lambda_value * regularizer) * data.T * target
    return theta


def learn_projection(src_vectors, tar_vectors, lmbd=1.0, save2file=None):
    src_vectors = np.mat([[i for i in vec] for vec in src_vectors])
    tar_vectors = np.mat([[i for i in vec] for vec in tar_vectors])
    m = len(src_vectors)
    x = np.c_[np.ones(m), src_vectors]  # Adding bias term to the source vectors

    num_features = src_vectors.shape[1]

    # Build initial zero transformation matrix
    learned_projection = np.zeros((num_features, x.shape[1]))
    learned_projection = np.mat(learned_projection)

    for component in tqdm(range(0, num_features)):  # Iterate over input components
        y = tar_vectors[:, component]  # True answers
        # Computing optimal transformation vector for the current component
        cur_projection = normalequation(x, y, lmbd, num_features)

        # Adding the computed vector to the transformation matrix
        learned_projection[component, :] = cur_projection.T

    if save2file:
        # Saving matrix to file:
        np.save(save2file, learned_projection)
    return learned_projection


def predict(src_word, src_embedding, tar_emdedding, projection, topn=10):
    test = np.mat(src_embedding[src_word])  # нашли вектор слова в исходной модели
    test = np.c_[1.0, test]  # Adding bias term
    predicted_vector = np.dot(projection, test.T)
    predicted_vector = np.squeeze(np.asarray(predicted_vector))
    # Our predictions:
    # нашли ближайшие в другой модели
    nearest_neighbors = tar_emdedding.most_similar(positive=[predicted_vector], topn=topn)
    return nearest_neighbors, predicted_vector


if __name__ == "__main__":
    # add command line arguments
    # this is probably the easiest way to store args for downstream
    parser = ArgumentParser()
    parser.add_argument('--spath', required=True, help="Path to the source model")
    parser.add_argument('--tpath', required=True, help="Path to the target model")
    parser.add_argument('--dic', help="Path to the bilingual dictionary", default="ru-en_lem.txt")
    parser.add_argument('--lmbd', action='store', type=float, default=1.0)
    parser.add_argument('--out', help="File to save the matrix to", default="proj.npy")
    args = parser.parse_args()

    src_model_path = args.spath
    tar_model_path = args.tpath
    bidict_path = args.dic

    src_model = load_embeddings(src_model_path)
    tar_model = load_embeddings(tar_model_path)

    # выбираем пары слов в двуязычном словаре, которые есть в обеих моделях
    lines = open(bidict_path, encoding='utf-8').read().splitlines()
    print('Number of lines in the dictionary:', len(lines))
    learn_pairs = []
    not_learn_pairs = []
    for line in tqdm(lines):
        pair = line.split()
        if pair[0] in src_model.vocab and pair[1] in tar_model.vocab:
            learn_pairs.append((pair[0], pair[1]))
        elif pair[0] not in src_model.vocab and pair[1] not in tar_model.vocab:
            not_learn_pairs.append((pair[0], pair[1]))
    print('Pairs to learn a transformation on:', len(learn_pairs))
    print('Skipped pairs:', len(not_learn_pairs))

    # open('models/muse_bidicts/ru-en_lem_clean.txt', 'w', encoding='utf-8').
    # write('\n'.join(['{}\t{}'.format(pair[0], pair[1]) for pair in learn_pairs]))
    # слова, на которых не обучались, но можем получить для них вектор
    # print([word for word in tqdm(src_model.vocab) if word not in [line.split()[0]
    # for line in tqdm(lines)]])

    dim = src_model.vector_size

    assert src_model.vector_size == tar_model.vector_size

    # делаем парные матрицы
    source_matrix = np.zeros((len(learn_pairs), dim))
    target_matrix = np.zeros((len(learn_pairs), dim))
    for nr, pair in tqdm(enumerate(learn_pairs)):
        source_matrix[nr, :] = src_model[pair[0]]
        target_matrix[nr, :] = tar_model[pair[1]]
    print(source_matrix.shape)
    print(target_matrix.shape)

    # pdump(source_matrix, open('models/ru_clean_lem.pkl', 'wb'))
    # pdump(target_matrix, open('models/en_clean_lem.pkl', 'wb'))

    # source_matrix = pload(open('models/ru_clean_lem.pkl', 'rb'))
    # print(source_matrix.shape)
    # target_matrix = pload(open('models/en_clean_lem.pkl', 'rb'))
    # print(target_matrix.shape)

    # обучаем модель на парных матрицах
    proj = learn_projection(source_matrix, target_matrix, lmbd=args.lmbd, save2file=args.out)
    print(proj.shape)

    # слово из двуязычного словаря
    candidates = predict('человек_NOUN', src_model, tar_model, proj)
    print(candidates[0])

    # слово не из двуязычного словаря, но из модели
    candidates = predict('шнурочек_NOUN', src_model, tar_model, proj)
    print(candidates[0])
