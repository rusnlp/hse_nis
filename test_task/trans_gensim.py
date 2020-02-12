from gensim.models import translation_matrix, TranslationMatrix
from gensim.models import KeyedVectors
from pickle import dump as pdump, load as pload
from json import load as jload

# path = 'models/muse_bidicts/ru-en_lem_clean.txt'
# lines = open(path, encoding='utf-8').read().splitlines()
# pairs = [tuple(line.split()) for line in lines]
# print(len(pairs))

# source_word_vec_file = 'models/ru.bin'
# source_word_vec = KeyedVectors.load_word2vec_format(source_word_vec_file, binary=True)

# target_word_vec_file = 'models/en.bin'
# target_word_vec = KeyedVectors.load_word2vec_format(target_word_vec_file, binary=True)

'''
transmat = translation_matrix.TranslationMatrix(source_word_vec, target_word_vec, pairs)
transmat.train(pairs)
print ("the shape of translation matrix is: ", transmat.translation_matrix.shape)
transmat.save('models/ru_en_lem_trans_gensim')
'''

transmat = TranslationMatrix.load('models/ru_en_lem_trans_gensim')
#vecs = transmat.source_lang_vec.vectors
#print(vecs)
#print(vecs.shape)
translated = transmat.translate('человек_NOUN', 1)
print(translated)
trw = translated['человек_NOUN'][0]
print(trw)
idx = transmat.target_lang_vec.index2word.index(trw)
print(idx)
print(transmat.target_lang_vec.vectors[idx])


lemmatized = jload(open('RU.json', encoding='utf-8'))
print(lemmatized)