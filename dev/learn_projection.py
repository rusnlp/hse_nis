'''
Обучение матрицы трансформаций
https://github.com/ltgoslo/diachronic_armed_conflicts/blob/master/helpers.py

python learn_projection.py --src_model_path=models/ru.bin --tar_model_path=models/en.bin --bidict_path=words/ru-en_lem.txt --proj_path=words/ru-en_proj.txt
'''

import argparse
import numpy as np
from tqdm import tqdm

from utils.loaders import load_embeddings, load_bidict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Обучение матрицы линейной трансформации')
    parser.add_argument('--src_model_path', type=str, required=True,
                        help='Путь к эмбеддингам исходного языка')
    parser.add_argument('--tar_model_path', type=str, required=True,
                        help='Путь к эмбеддингам целевого языка')
    parser.add_argument('--bidict_path', type=str, required=True,
                        help='Путь к двуязычному словарю, по которому будет строиться проекция')
    parser.add_argument('--proj_path', type=str, required=True,
                        help='Путь, по которому будет сохранена проекция')
    return parser.parse_args()


def create_learn_pairs(src_model, tar_model, bidict):
    '''собираем пары, которые точно есть в моделях'''
    learn_pairs = [(pair[0], pair[1]) for pair in bidict.items() if pair[0] in src_model.vocab and pair[1] in tar_model.vocab]
    # print(len(learn_pairs), learn_pairs)
    return learn_pairs


def create_parallel_matrices(src_model, tar_model, pairs):
    '''параллельные матрицы на основе пар, которые точно есть в моделях'''
    dim = src_model.vector_size
    # делаем парные матрицы
    source_matrix = np.zeros((len(pairs), dim))
    target_matrix = np.zeros((len(pairs), dim))
    for i, pair in tqdm(enumerate(pairs)):
        source_matrix[i, :] = src_model[pair[0]]
        target_matrix[i, :] = tar_model[pair[1]]
    print(source_matrix.shape)
    print(target_matrix.shape)

    # pdump(source_matrix, open('models/ru_clean_lem.pkl', 'wb'))
    # pdump(target_matrix, open('models/en_clean_lem.pkl', 'wb'))
    return source_matrix, target_matrix


def create_bidict_matrices(src_model, tar_model, bidict_path):
    '''объединение загружки параллельного словаря и создания параллелльных матриц'''
    bidict = load_bidict(bidict_path)
    learn_pairs = create_learn_pairs(src_model, tar_model, bidict)
    src_matrix, tar_matrix = create_parallel_matrices(src_model, tar_model, learn_pairs)
    return src_matrix, tar_matrix, learn_pairs



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

    for component in tqdm(range(0, num_features)):  # Iterate over input components
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

    src_matrix, tar_matrix, _ = create_bidict_matrices(src_model, tar_model, args.bidict_path)

    proj = learn_projection(src_matrix, tar_matrix, dim, lmbd=1.0, proj_path=args.proj_path)
    print('Проекция размерности {} сохранена в {}'.format(proj.shape, args.proj_path))


if __name__ == '__main__':
    main()
