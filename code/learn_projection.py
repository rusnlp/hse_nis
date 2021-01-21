'''
Обучение матрицы трансформаций
https://github.com/ltgoslo/diachronic_armed_conflicts/blob/master/helpers.py

python learn_projection.py --src_model_path=../models/en_w2v_tok.bin.gz --tar_model_path=../models/cross_muse_orig.bin.gz --proj_path=../models/en_muse_tok.proj
python learn_projection.py --src_model_path=../models/ru_w2v_tok.bin.gz --tar_model_path=../models/cross_muse_orig.bin.gz --proj_path=../models/ru_muse_tok.proj
'''

import argparse
import numpy as np
from tqdm import tqdm

from utils.loaders import load_embeddings


def parse_args():
    parser = argparse.ArgumentParser(
        description='Обучение матрицы линейной трансформации')
    parser.add_argument('--src_model_path', type=str, required=True,
                        help='Путь к эмбеддингам исходного языка')
    parser.add_argument('--tar_model_path', type=str, required=True,
                        help='Путь к эмбеддингам целевого языка')
    parser.add_argument('--proj_path', type=str, required=True,
                        help='Путь, по которому будет сохранена проекция')
    return parser.parse_args()


def create_learn_pairs(src_model, tar_model):
    '''собираем слова, которые есть в обеих моделях'''
    src_vocab = set(src_model.vocab.keys())
    tar_vocab = set(tar_model.vocab.keys())
    pair_words = src_vocab & tar_vocab
    # print(len(pair_words), pair_words)
    return pair_words


def create_parallel_matrices(src_model, tar_model, words):
    '''параллельные матрицы на основе слов, которые есть в обеих моделях'''
    dim = src_model.vector_size
    # делаем парные матрицы
    source_matrix = np.zeros((len(words), dim))
    target_matrix = np.zeros((len(words), dim))
    for i, word in tqdm(enumerate(words), desc='Creating parallel matrices'):
        source_matrix[i, :] = src_model[word]
        target_matrix[i, :] = tar_model[word]
    print(source_matrix.shape)
    print(target_matrix.shape)

    # pdump(source_matrix, open('models/ru_clean_lem.pkl', 'wb'))
    # pdump(target_matrix, open('models/en_clean_lem.pkl', 'wb'))
    return source_matrix, target_matrix


def create_bidict_matrices(src_model, tar_model):
    '''объединение поиска общих слов и создания параллелльных матриц'''
    pair_words = create_learn_pairs(src_model, tar_model)
    src_matrix, tar_matrix = create_parallel_matrices(src_model, tar_model, pair_words)
    return src_matrix, tar_matrix, pair_words


def normalequation(data, target, lambda_value, vector_size):
    regularizer = 0
    if lambda_value != 0:  # Regularization term
        regularizer = np.eye(vector_size + 1)
        regularizer[0, 0] = 0
        regularizer = np.mat(regularizer)
    # Normal equation:
    theta = np.linalg.pinv(data.T * data + lambda_value * regularizer) * data.T * target
    return theta


def learn_projection(src_vectors, tar_vectors, embed_size, lmbd=1.0, proj_path=None):
    src_vectors = np.mat([[i for i in vec] for vec in src_vectors])
    tar_vectors = np.mat([[i for i in vec] for vec in tar_vectors])
    m = len(src_vectors)
    x = np.c_[np.ones(m), src_vectors]  # Adding bias term to the source vectors

    num_features = embed_size

    # Build initial zero transformation matrix
    learned_projection = np.zeros((num_features, x.shape[1]))
    learned_projection = np.mat(learned_projection)

    # Iterate over input components
    for component in tqdm(range(0, num_features), desc='Projecting'):
        y = tar_vectors[:, component]  # True answers
        # Computing optimal transformation vector for the current component
        cur_projection = normalequation(x, y, lmbd, num_features)

        # Adding the computed vector to the transformation matrix
        learned_projection[component, :] = cur_projection.T

    if proj_path:
        # Saving matrix to file:
        np.savetxt(proj_path, learned_projection, delimiter=',')
    return learned_projection


def main():
    args = parse_args()
    src_model = load_embeddings(args.src_model_path)
    tar_model = load_embeddings(args.tar_model_path)
    dim = src_model.vector_size

    src_matrix, tar_matrix, _ = create_bidict_matrices(src_model, tar_model)

    proj = learn_projection(src_matrix, tar_matrix, dim, lmbd=1.0, proj_path=args.proj_path)
    print('Проекция размерности {} сохранена в {}'.format(proj.shape, args.proj_path))


if __name__ == '__main__':
    main()
