import numpy as np
from tqdm import tqdm
from gensim import models
import zipfile
import logging
import sys
from numpy.linalg import norm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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


def load_bidict(bidict_path):
    '''читаем словарь пар'''
    lines = open(bidict_path, encoding='utf-8').read().splitlines()
    bidict = [line.split() for line in tqdm(lines)]
    print(len(lines))
    return bidict


def create_learn_pairs(src_model, tar_model, bidict):
    '''собираем пары, которые точно есть в моделях'''
    learn_pairs = [(pair[0], pair[1]) for pair in bidict if pair[0] in src_model.vocab and pair[1] in tar_model.vocab]
    print(len(learn_pairs), learn_pairs)
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


def load_projection(proj_path):
    projection = np.loadtxt(proj_path, delimiter=',')
    print(projection.shape)
    return projection


def project_vec(src_vec, projection):
    '''Проецируем вектор'''
    test = np.mat(src_vec)
    test = np.c_[1.0, test]  # Adding bias term
    predicted_vec = np.dot(projection, test.T)
    predicted_vec = np.squeeze(np.asarray(predicted_vec))
    return predicted_vec


def predict_projection_word(src_word, src_embedding, tar_emdedding, projection, topn=10):
    '''По слову предсказываем переводы и вектор ближайшего'''
    src_vec = src_embedding[src_word]
    predicted_vec = project_vec(src_vec, projection)
    # нашли ближайшие в другой модели
    nearest_neighbors = tar_emdedding.most_similar(positive=[predicted_vec], topn=topn)
    return nearest_neighbors, predicted_vec


def projection_vectorize_text(tokens, w2v, no_duplicates, projection):
    '''вкторизазия текста матрицей трансформации'''
    words = [token for token in tokens if token in w2v]
    # если прилетел пустой текст, то он так и останется пустым просто
    if not words:
        print('Я ничего не знаю из этих токенов: {}'.format(tokens), file=sys.stderr)
        return np.zeros(w2v.vector_size)  # TODO: возвращать вектор какого-то слова

    if no_duplicates:
        words = set(words)

    t_vecs = np.zeros((len(words), w2v.vector_size))

    for i, token in enumerate(words):
        src_vec = w2v[token]
        t_vecs[i, :] = project_vec(src_vec, projection)

    t_vec = np.sum(t_vecs, axis=0)
    t_vec = np.divide(t_vec, len(words))
    # TODO: уже нормализованные же, да?

    return t_vec


def model_vectorize_text(tokens, w2v, no_duplicates):
    '''вкторизазия текста моделью (бейзлайн, мьюз, векмап)'''
    words = [token for token in tokens if token in w2v]
    # если прилетел пустой текст, то он так и останется пустым просто

    if not words:
        print('Я ничего не знаю из этих токенов: {}'.format(tokens), file=sys.stderr)
        return np.zeros(w2v.vector_size)  # TODO: возвращать вектор какого-то слова

    if no_duplicates:
        words = set(words)

    t_vecs = np.zeros((len(words), w2v.vector_size))

    for i, token in enumerate(words):
        t_vecs[i, :] = w2v[token]

    t_vec = np.sum(t_vecs, axis=0)
    t_vec = np.divide(t_vec, len(words))
    vec = t_vec / norm(t_vec)

    return vec


def vectorize_corpus(corpus, vectors, w2v, no_duplicates, starts_from=0, type='model', projection=None):
    """векторизация всего корпуса. Если матрицей, то в model будет матрица трансформаци"""
    not_vectorized = []

    for i, text in tqdm(enumerate(corpus)):

        if type == 'model':  # если векторизуем предобученной моделью
            vector = model_vectorize_text(text, w2v, no_duplicates)
        elif type == 'projection':
            vector = projection_vectorize_text(text, w2v, no_duplicates, projection)

        if len(vector) != 0:  # для np.array нельзя просто if
            vectors[starts_from + i, :] = vector[:]  # дописывам вектора новых текстов в конец

        else:
            not_vectorized.append(i)
            continue

    return vectors, not_vectorized