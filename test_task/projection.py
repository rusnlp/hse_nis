'''
Обучение матрицы трансформаций
'''

src_model_path = 'models/ru.bin'
tar_model_path = 'models/en.bin'
bidict_path = 'words/ru-en_lem.txt'
proj_path = 'words/ru-en_proj.txt'

import numpy as np
from tqdm import tqdm
from vectorization import load_embeddings, create_bidict_matrices


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
    src_model = load_embeddings(src_model_path)
    tar_model = load_embeddings(tar_model_path)
    dim = src_model.vector_size

    src_matrix, tar_matrix, _ = create_bidict_matrices(src_model, tar_model, bidict_path)

    proj = learn_projection(src_matrix, tar_matrix, dim, lmbd=1.0, proj_path=proj_path)
    print('Проекция размерности {} сохранена в {}'.format(proj.shape, proj_path))


if __name__ == '__main__':
    main()