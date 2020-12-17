"""
Достаём словарь модели в файл, чтобы можно было на него посмотреть
"""

from os import mkdir
from utils.loaders import load_embeddings

models_dir = '../models/'
vocab_dir = '../words/'

try:
    mkdir(vocab_dir)
except:
    pass


def save_vocab(model_path, vocab_path):
    model = load_embeddings(model_path)
    words = [word for word in model.vocab]
    print(len(words))
    open(vocab_path, 'w', encoding='utf-8').write('\n'.join(words))
    print()


if __name__ == '__main__':
    cross_model_path = models_dir + 'cross_muse.bin.gz'
    cross_vocab_path = vocab_dir + 'cross_muse_vocab.txt'

    save_vocab(cross_model_path, cross_vocab_path)
