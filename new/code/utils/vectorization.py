import numpy as np
from tqdm import tqdm
from numpy.linalg import norm

from utils.loaders import load_embeddings, load_projection, load_bidict
from utils.preprocessing import translate_line


class UnknownMethodError(Exception):
    """
    Указан неизвестный метод векторизации
    """
    def __init__(self):
        self.text = 'Неправильно указан метод векторизации! ' \
                    'Варианты: "model", "projection", "translation"'

    def __str__(self):
        return self.text


class BaseVectorizer():
    def __init__(self, embeddings_path, no_duplicates):
        self.embeddings_file = embeddings_path
        self.no_duplicates = no_duplicates
        self.model = load_embeddings(embeddings_path)
        self.dim = self.model.vector_size
        self.empty = np.zeros(self.dim)  # TODO: возвращать вектор какого-то слова

    def __str__(self):
        return '{}:\nModel: {}\nDim: {}'.format(type(self), self.embeddings_file, self.dim)

    def get_words(self, tokens):
        words = [token for token in tokens if token in self.model]
        # если прилетел пустой текст, то он так и останется пустым просто

        if self.no_duplicates:
            words = set(words)

        return words

    def get_norm_vec(self, vec):
        vec = vec / norm(vec)
        return vec

    def get_mean_vec(self, vecs, words):
        vec = np.sum(vecs, axis=0)
        vec = np.divide(vec, len(words))  # TODO: почему не np.mean? Другая длина words?
        # vec = np.mean(vec, axis=0)
        return vec

    def get_norm_mean_vec(self, vecs, words):
        vec = self.get_mean_vec(vecs, words)
        vec = self.get_norm_vec(vec)
        return vec

    def base_vectorize_text(self, tokens):
        # простая векторизация моделью
        words = self.get_words(tokens)

        if not words:
            # print('Я ничего не знаю из этих токенов: {}'.format(tokens), file=sys.stderr)
            return self.empty

        t_vecs = np.zeros((len(words), self.dim))
        for i, token in enumerate(words):
            t_vecs[i, :] = self.model[token]
        t_vec = self.get_norm_mean_vec(t_vecs, words)

        return t_vec


class ModelVectorizer(BaseVectorizer):
    '''векторизация текста моделью (бейзлайн, мьюз, векмап)'''

    def __init__(self, embeddings_path, no_duplicates):
        super(ModelVectorizer, self).__init__(embeddings_path, no_duplicates)

    def vectorize_text(self, tokens):
        t_vec = self.base_vectorize_text(tokens)
        return t_vec


class ProjectionVectorizer(BaseVectorizer):
    '''векторизация текста матрицей трансформации'''

    def __init__(self, embeddings_path, projection_path, no_duplicates):
        super(ProjectionVectorizer, self).__init__(embeddings_path, no_duplicates)
        self.projection = load_projection(projection_path)

    def project_vec(self, src_vec):
        '''Проецируем вектор'''

        test = np.mat(src_vec)
        test = np.c_[1.0, test]  # Adding bias term
        predicted_vec = np.dot(self.projection, test.T)
        predicted_vec = np.squeeze(np.asarray(predicted_vec))
        # print('Проецирую и нормализую вектор')
        return predicted_vec

    def predict_projection_word(self, src_word, tar_emdedding, topn=10):
        '''По слову предсказываем переводы и трансформированный вектор'''
        src_vec = self.model[src_word]
        predicted_vec = self.project_vec(src_vec)
        # нашли ближайшие в другой модели
        nearest_neighbors = tar_emdedding.most_similar(positive=[predicted_vec], topn=topn)
        return nearest_neighbors, predicted_vec

    def vectorize_text(self, tokens):
        '''векторизация текста матрицей трансформации'''

        words = self.get_words(tokens)

        if not words:
            # print('Я ничего не знаю из этих токенов: {}'.format(tokens), file=sys.stderr)
            return self.empty

        t_vecs = np.zeros((len(words), self.dim))
        for i, token in enumerate(words):
            src_vec = self.model[token]
            t_vecs[i, :] = self.project_vec(src_vec)
        t_vec = self.get_mean_vec(t_vecs, words)
        t_vec = self.get_norm_vec(t_vec)

        return t_vec


class TranslationVectorizer(BaseVectorizer):
    '''Перевод русского текста на английский и обычная векторизация'''

    def __init__(self, embeddings_path, bidict_path, no_duplicates):
        super(TranslationVectorizer, self).__init__(embeddings_path, no_duplicates)
        self.bidict = load_bidict(bidict_path)
        # print(self.bidict)

    def translate_text(self, text, trans_dict):
        '''Переводим лемматизировнный русский корпус по лемматизированному двуязычному словарю'''
        # print(text)
        translated_text = translate_line(text, trans_dict)
        # print(translated_text)
        return translated_text

    # переведённый текст векторизуем базовой функцией
    def vectorize_text(self, tokens):
        translated_tokens = self.translate_text(tokens, self.bidict)
        t_vec = self.base_vectorize_text(translated_tokens)
        return t_vec


def build_vectorizer(direction, method, embeddings_path, no_duplicates,
                     projection_path='', bidict_path=''):
    if direction == 'tar':  # для целевого языка всегда векторизуем моделью
        vectorizer = ModelVectorizer(embeddings_path, no_duplicates)

    else:  # для исходного языка уже определяем метод
        if method == 'model':  # если векторизуем предобученной моделью
            vectorizer = ModelVectorizer(embeddings_path, no_duplicates)
        elif method == 'projection':
            vectorizer = ProjectionVectorizer(embeddings_path, projection_path, no_duplicates)
        elif method == 'translation':
            vectorizer = TranslationVectorizer(embeddings_path, bidict_path, no_duplicates)
        else:
            raise UnknownMethodError()

    print(vectorizer)
    return vectorizer


def vectorize_corpus(corpus, vectorizer):
    """векторизация всего корпуса. Если матрицей, то в model будет матрица трансформаци"""
    not_vectorized = []
    corp_vectors = {}

    for name, text in tqdm(corpus.items()):
        vector = vectorizer.vectorize_text(text)
        # print(vector.shape)
        # print(vectorizer.empty.shape)
        if not np.array_equal(vector, vectorizer.empty):  # если это не зарезервированный вектор
            corp_vectors[name] = vector

        else:
            not_vectorized.append(name)
            continue

    return corp_vectors, not_vectorized


def save_text_vectors(vectors, output_path):
    # TODO: подгружать эту модель в словарь, чтобы можно было его менять
    vec_str = '{} {}'.format(len(vectors), len(list(vectors.values())[0]))
    for word, vec in vectors.items():
        vec_str += '\n{} {}'.format(word, ' '.join(str(v) for v in vec))  #TODO: так можно? так сделать везде?
    open(output_path, 'w', encoding='utf-8').write(vec_str)
    print('Сохранил вектора в {}'.format(output_path))
