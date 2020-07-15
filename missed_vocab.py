from json import dump, load
from os import mkdir
from test_task.utils.loaders import load_embeddings

vocab_path = 'words/vocab/'

try:
    mkdir(vocab_path)
except:
    pass

en_simple_path = 'models/en.bin'
en_vecmap_path = 'models/en_vecmap.vec'
en_muse_path = 'models/en_muse.vec'

ru_simple_path = 'models/ru.bin'
ru_vecmap_path = 'models/ru_vecmap.vec'
ru_muse_path = 'models/ru_muse.vec'

en_simple_vocab_path = vocab_path+'en_simple_vocab.txt'
en_vecmap_vocab_path = vocab_path+'en_vecmap_vocab.txt'
en_muse_vocab_path = vocab_path+'en_muse_vocab.txt'

ru_simple_vocab_path = vocab_path+'ru_simple_vocab.txt'
ru_vecmap_vocab_path = vocab_path+'ru_vecmap_vocab.txt'
ru_muse_vocab_path = vocab_path+'ru_muse_vocab.txt'

vocab_diffs_path = vocab_path+'vocab_diffs.json'


# СОХРАНИТЬ СЛОВАРИ ИЗ МОДЕЛЕЙ
# def save_vocab(model_path, vocab_path):
#     en_simple = load_embeddings(model_path)
#     en_simple_words = [word for word in en_simple.vocab]
#     print(len(en_simple_words))
#     open(vocab_path, 'w', encoding='utf-8').write('\n'.join(en_simple_words))
#     print()
#
#
# save_vocab(en_simple_path, en_simple_vocab_path)
# save_vocab(ru_simple_path, ru_simple_vocab_path)
# save_vocab(en_vecmap_path, en_vecmap_vocab_path)
# save_vocab(ru_vecmap_path, ru_vecmap_vocab_path)
# save_vocab(en_muse_path, en_muse_vocab_path)
# save_vocab(ru_muse_path, en_muse_vocab_path)


# СОХРАНИТЬ РАЗНИЦЫ В JSON
# def load_vocab(path, clean=True):
#     raw = open(path, encoding='utf-8').read().lower().splitlines()
#     if clean:
#         raw = [line.split('_')[0] for line in raw]
#     vocab = set(raw)
#     print(len(vocab))
#     return vocab
#
#
# en_simple_vocab = load_vocab(en_simple_vocab_path, clean=True)
# ru_simple_vocab = load_vocab(ru_simple_vocab_path, clean=True)
#
# en_muse_vocab = load_vocab(en_muse_vocab_path)
# ru_muse_vocab = load_vocab(ru_muse_vocab_path)
#
# print()
#
# vocab_diffs = {}
#
# en_sim_muse = en_simple_vocab - en_muse_vocab
# print(len(en_sim_muse))
# vocab_diffs['en sim-muse'] = list(en_sim_muse)
#
# en_muse_sim = en_muse_vocab - en_simple_vocab
# print(len(en_muse_sim))
# vocab_diffs['en muse-sim'] = list(en_muse_sim)
#
# ru_sim_muse = ru_simple_vocab - ru_muse_vocab
# print(len(ru_sim_muse))
# vocab_diffs['ru sim-muse'] = list(ru_sim_muse)
#
# ru_muse_sim = ru_muse_vocab - ru_simple_vocab
# print(len(ru_muse_sim))
# vocab_diffs['ru muse-sim'] = list(ru_muse_sim)
#
# dump(vocab_diffs, open(vocab_diffs_path, 'w', encoding='utf-8'))


# vocab_diffs = load(open(vocab_diffs_path, encoding='utf-8'))
# print(vocab_diffs.keys())
# for k in vocab_diffs.keys():
#     vocab_diff = vocab_diffs[k]
#     vocab_diff_path = vocab_path + k.replace(' ', '_') + '.txt'
#     open(vocab_diff_path, 'w', encoding='utf-8').write('\n'.join(vocab_diff))
