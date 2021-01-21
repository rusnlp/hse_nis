"""
Достаём словарь модели в файл, чтобы можно было на него посмотреть

python save_vocab.py --model_paths=../models/cross_muse_orig.bin.gz --vocab_paths=../words2/cross_muse_orig_vocab.txt
"""

import argparse
from utils.loaders import load_embeddings, create_dir, split_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description='Сохранение словаря модели в текст')
    parser.add_argument('--model_paths', type=str, required=True,
                        help='Пути к моделям (можно перечислить через +)')
    parser.add_argument('--vocab_paths', type=str, required=True,
                        help='Пути к файлам со словарями моделей (можно перечислить через +)')

    return parser.parse_args()


def save_vocab(model_path, vocab_path):
    model = load_embeddings(model_path)
    words = list(model.vocab.keys())
    create_dir(vocab_path)
    print('Размер словаря: {} слов'.format(len(words)))
    open(vocab_path, 'w', encoding='utf-8').write('\n'.join(words))


def main():
    args = parse_args()
    model_paths = args.model_paths.split('+')
    vocab_paths = split_paths(args.vocab_paths, model_paths)

    for model_path, vocab_path in zip(model_paths, vocab_paths):
        save_vocab(model_path, vocab_path)


if __name__ == '__main__':
    main()