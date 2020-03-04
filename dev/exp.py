"""
Чтобы пересоздать матрицу векторов для русского, get_new_vectors = True
"""
get_new_vectors = False

from pickle import load as pload, dump as pdump
from json import load as jload
import numpy as np
from numpy.linalg import norm
from utils.vectorization import build_vectorizer, vectorize_corpus
from vectorization_pipeline import get_lemmatized_corpus

mapping = jload(open('texts/titles_mapping.json', encoding='utf-8'))

if get_new_vectors:
    lemmatized_dict = jload(open('texts/ru_pos_lemmatized.json', encoding='utf-8'))
    lemmatized_corpus = get_lemmatized_corpus(mapping, 'i2ru', lemmatized_dict, 54)
    vectorizer = build_vectorizer('src', 'projection', no_duplicates=0, embeddings_path='models/ru.bin', projection_path='words/ru-en_proj.txt')
    pr, _ = vectorize_corpus(lemmatized_corpus, np.zeros((54, vectorizer.dim)), vectorizer)
    pdump(pr, open('ru_vectors.pkl', 'wb'))
else:
    pr = pload(open('ru_vectors.pkl', 'rb'))


# отсюда начинаются странности
ru = mapping['ru2i']['кошка.txt']
a = pr[ru]
print(a.shape)
print(a.tolist())
print(norm(a))
print([True for e in a if e**2>1]) # есть ли элементы, которые по модулю больше 1
print(np.dot(a, a))

print('-'*30)

b = a/norm(a)
print(b.tolist())
print(norm(b))
print(np.dot(b, b))
