'''
https://github.com/ltgoslo/diachronic_armed_conflicts/blob/master/helpers.py
'''

import numpy as np
from pickle import load
from tqdm import tqdm
from gensim import models
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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


src_model = load_embeddings('models/ru.bin')
tar_model = load_embeddings('models/en.bin')

embed_size = 300
src_vectors = load(open('models/ru_clean_lem.pkl', 'rb'))
print(src_vectors.shape)
tar_vectors = load(open('models/en_clean_lem.pkl', 'rb'))
print(tar_vectors.shape)

proj = learn_projection(src_vectors, tar_vectors, embed_size, lmbd=1.0, save2file='prj.txt')
print(proj.shape)

candidates = predict('человек_NOUN', src_model, tar_model, proj)
print(candidates)

