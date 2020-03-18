from gensim.models import KeyedVectors
path = 'models/ru.bin'
print('loading')
bin_model = KeyedVectors.load_word2vec_format(path, binary=True)
print('saving')
bin_model.save_word2vec_format('ru.vec', binary=False)
