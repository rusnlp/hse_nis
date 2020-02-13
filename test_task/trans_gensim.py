'''
Запускала без командной строки, просто скрипт в пайчарме
'''

from gensim.models import translation_matrix, TranslationMatrix
from gensim.models import KeyedVectors
from pickle import dump as pdump, load as pload
from json import load as jload

bidict_path = 'models/muse_bidicts/ru-en_lem_clean.txt'
source_word_vec_file = 'models/ru.bin'
target_word_vec_file = 'models/en.bin'
transmat_path = 'models/ru_en_lem_trans_gensim'


lines = open(bidict_path, encoding='utf-8').read().splitlines()
pairs = [tuple(line.split()) for line in lines]
print(len(pairs))

source_word_vec = KeyedVectors.load_word2vec_format(source_word_vec_file, binary=True)
target_word_vec = KeyedVectors.load_word2vec_format(target_word_vec_file, binary=True)


transmat = translation_matrix.TranslationMatrix(source_word_vec, target_word_vec, pairs)
transmat.train(pairs)
print("the shape of translation matrix is: ", transmat.translation_matrix.shape)
transmat.save(transmat_path)

transmat = TranslationMatrix.load(transmat_path)
#vecs = transmat.source_lang_vec.vectors
#print(vecs)
#print(vecs.shape)


# Для слова из двуязычного словаря всё работает
translated = transmat.translate('человек_NOUN', 5)
print(translated)
trw = translated['человек_NOUN'][0]
print(trw)
idx = transmat.target_lang_vec.index2word.index(trw)
print(idx)
print(transmat.target_lang_vec.vectors[idx])

print('='*50)

# слово не из двуязычного словаря, но из модели
translated = transmat.translate('шнурочек_NOUN', 5)
print(translated)
trw = translated['шнурочек_NOUN'][0]
print(trw)
idx = transmat.target_lang_vec.index2word.index(trw)
print(idx)
print(transmat.target_lang_vec.vectors[idx])



vec = source_word_vec['человек_NOUN']
# translated_vec = transmat.translate(vec, 5)
# на вектор -- KeyError

# translated_vec = transmat.apply_transmat(vec)
'''
Traceback (most recent call last):
  File "hse_nis/test_task/trans_gensim.py", line 41, in <module>
    translated_vec = transmat.apply_transmat(vec)
  File "C:\Program Files\Python36\lib\site-packages\gensim\models\translation_matrix.py", line 277, in apply_transmat
    return Space(np.dot(words_space.mat, self.translation_matrix), words_space.index2word)
AttributeError: 'numpy.ndarray' object has no attribute 'mat'
'''

# translated_vec = transmat.apply_transmat([vec])
'''
AttributeError: 'list' object has no attribute 'mat'
'''
