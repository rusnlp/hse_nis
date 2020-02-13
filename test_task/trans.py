'''
Запускала без командной строки, просто скрипт в пайчарме
'''
'''
https://github.com/ltgoslo/diachronic_armed_conflicts/blob/master/helpers.py
'''

import numpy as np
from pickle import load as pload, dump as pdump
from tqdm import tqdm
from gensim import models
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


src_model_path = 'models/ru.bin'
tar_model_path = 'models/en.bin'
bidict_path = 'models/muse_bidicts/ru-en_lem.txt'

def load_embeddings(modelfile):
    if modelfile.endswith('.txt.gz') or modelfile.endswith('.txt'):
        model = models.KeyedVectors.load_word2vec_format(modelfile, binary=False)
    elif modelfile.endswith('.bin.gz') or modelfile.endswith('.bin'):
        model = models.KeyedVectors.load_word2vec_format(modelfile, binary=True)
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



def learn_projection(src_vectors, tar_vectors, embed_size, lmbd=1.0, save2file=None):
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

    if save2file:
        # Saving matrix to file:
        np.savetxt(save2file, learned_projection, delimiter=',')
    return learned_projection


def predict(src_word, src_embedding, tar_emdedding, projection, topn=10):
    test = np.mat(src_embedding[src_word]) # нашли вектор слова
    test = np.c_[1.0, test]  # Adding bias term
    predicted_vector = np.dot(projection, test.T)
    predicted_vector = np.squeeze(np.asarray(predicted_vector))
    # Our predictions:
    nearest_neighbors = tar_emdedding.most_similar(positive=[predicted_vector], topn=topn) # нашли ближайшие в другой модели
    return nearest_neighbors, predicted_vector


src_model = load_embeddings(src_model_path)
tar_model = load_embeddings(src_model_path)

lines = open(bidict_path, encoding='utf-8').read().splitlines()
print(len(lines))
learn_pairs = []
not_learn_pairs = []
for line in tqdm(lines):
    pair = line.split()
    if pair[0] in src_model.vocab and pair[1] in tar_model.vocab:
        learn_pairs.append((pair[0], pair[1]))
    elif pair[0] not in src_model.vocab and pair[1] not in tar_model.vocab:
        not_learn_pairs.append((pair[0], pair[1]))
print(len(learn_pairs), learn_pairs)
print(len(not_learn_pairs), not_learn_pairs)
#open('models/muse_bidicts/ru-en_lem_clean.txt', 'w', encoding='utf-8').write('\n'.join(['{}\t{}'.format(pair[0], pair[1]) for pair in learn_pairs]))
# слова, на которых не обучались, но можем получить для них вектор
# print([word for word in tqdm(src_model.vocab) if word not in [line.split()[0] for line in tqdm(lines)]])

dim = src_model.vector_size

# делаем парные матрицы
source_matrix = np.zeros((len(learn_pairs), dim))
target_matrix = np.zeros((len(learn_pairs), dim))
for i, pair in tqdm(enumerate(learn_pairs)):
    source_matrix[i, :] = src_model[pair[0]]
    target_matrix[i, :] = tar_model[pair[1]]
print(source_matrix.shape)
print(target_matrix.shape)

#pdump(source_matrix, open('models/ru_clean_lem.pkl', 'wb'))
#pdump(target_matrix, open('models/en_clean_lem.pkl', 'wb'))

# source_matrix = pload(open('models/ru_clean_lem.pkl', 'rb'))
# print(source_matrix.shape)
# target_matrix = pload(open('models/en_clean_lem.pkl', 'rb'))
# print(target_matrix.shape)

proj = learn_projection(source_matrix, target_matrix, dim, lmbd=1.0, save2file='prj.txt')
print(proj.shape)

# слово из двуязычного словаря
candidates = predict('человек_NOUN', src_model, tar_model, proj)
print(candidates)

# слово не из двуязычного словаря, но из модели
candidates = predict('шнурочек_NOUN', src_model, tar_model, proj)
print(candidates)
