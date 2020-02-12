from tqdm import tqdm
from gensim.models import KeyedVectors
import logging
from json import dump as jdump, load as jload
import numpy as np
from pickle import dump as pdump, load as pload

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

'''
# Добавим в пары слов пос-теги
from tqdm import tqdm
from ufal.udpipe import Model, Pipeline
from preprocess import unify_sym, process
src_udpipe_path = 'models/ru.udpipe'
tar_udpipe_path = 'models/en.udpipe'
bidict_path = 'models/muse_bidicts/ru-en.txt'
lem_bidict_path = 'models/muse_bidicts/ru-en_lem.txt'

src_udpipe_model = Model.load(src_udpipe_path)
src_pipeline = Pipeline(src_udpipe_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
tar_udpipe_model = Model.load(tar_udpipe_path)
tar_pipeline = Pipeline(tar_udpipe_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

lines = open(bidict_path, encoding='utf-8').read().lower().splitlines()
lem_pairs = []
for line in tqdm(lines):
    pair = line.strip().split()
    src_lem = unify_sym(pair[0])
    src_lem = process(src_pipeline, src_lem, keep_pos=True, keep_punct=False, keep_stops=False)
    tar_lem = unify_sym(pair[1])
    tar_lem = process(tar_pipeline, tar_lem, keep_pos=True, keep_punct=False, keep_stops=False)
    if src_lem and tar_lem:
        lem_pairs.append('{}\t{}'.format(src_lem[0], tar_lem[0]))
print(len(lem_pairs))
lem_pairs = set(lem_pairs)
print(len(lem_pairs))
open(lem_bidict_path, 'w', encoding='utf-8').write('\n'.join(list(lem_pairs)))
'''

'''
# убираем pos-теги из моделей
source_word_vec = KeyedVectors.load_word2vec_format(source_word_vec_file, binary=True)
source_word_vec_clean = {word.lower().split('_')[:-1][0]: source_word_vec[word].tolist() for word in tqdm(source_word_vec.vocab)}
# TODO: конвертировать в keyedvectors
print('Saving...')
jdump(source_word_vec_clean, open('models/ru_clean.json', 'w'))
print('Saved!')

target_word_vec = KeyedVectors.load_word2vec_format(target_word_vec_file, binary=True)
target_word_vec_clean = {word.lower().split('_')[:-1][0]: target_word_vec[word].tolist() for word in tqdm(target_word_vec.vocab)}
print('Saving...')
jdump(target_word_vec_clean, open('models/en_clean.json', 'w'))
print('Saved!')


# source_vocab = [word.lower().split('_')[:-1] for word in source_word_vec.vocab.keys()]
# print(source_vocab)
# target_vocab = [word.lower().split('_')[:-1] for word in target_word_vec.vocab.keys()]
# print(target_vocab)
'''


source_word_vec_file = 'models/ru.bin'
source_word_vec = KeyedVectors.load_word2vec_format(source_word_vec_file, binary=True)
target_word_vec_file = 'models/en.bin'
target_word_vec = KeyedVectors.load_word2vec_format(target_word_vec_file, binary=True)

path = 'models/muse_bidicts/ru-en_lem.txt'
lines = open(path, encoding='utf-8').read().splitlines()
print(len(lines))
learn_pairs = []
for line in tqdm(lines):
    pair = line.split()
    if pair[0] in source_word_vec.vocab and pair[1] in target_word_vec.vocab:
        learn_pairs.append((pair[0], pair[1]))
print(len(learn_pairs))
print(learn_pairs)
open('models/muse_bidicts/ru-en_lem_clean.txt', 'w', encoding='utf-8').write('\n'.join(['{}\t{}'.format(pair[0], pair[1]) for pair in learn_pairs]))

dim = source_word_vec.vector_size

# делаем парные матрицы
source_matrix = np.zeros((len(learn_pairs), dim))
target_matrix = np.zeros((len(learn_pairs), dim))
for i, pair in tqdm(enumerate(learn_pairs)):
    source_matrix[i, :] = source_word_vec[pair[0]]
    target_matrix[i, :] = target_word_vec[pair[1]]
print(source_matrix.shape)
print(target_matrix.shape)
pdump(source_matrix, open('models/ru_clean_lem.pkl', 'wb'))
pdump(target_matrix, open('models/en_clean_lem.pkl', 'wb'))
