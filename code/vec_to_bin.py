"""
python vec_to_bin.py --vec_path=../models/en_muse.vec+../models/ru_muse.vec --bin_path=../models/en_muse.bin.gz+../models/ru_muse.bin.gz --remove_source=1
"""

import argparse
import logging
import os
from utils.loaders import load_embeddings, split_paths

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Преобразование векторов из текстового формата в бинарный')
    parser.add_argument('--vec_paths', type=str, required=True,
                        help='Пути к векторам в текстовом формате (можно перечислить через +)')
    parser.add_argument('--bin_paths', type=str, required=True,
                        help='Пути к векторам в бинарном формате (можно перечислить через +)')
    parser.add_argument('--remove_source', type=int, default=0,
                        help='Удалять ли исходный файл с векторами в текстовом формате (default: 0)')

    return parser.parse_args()


def main():
    args = parse_args()
    vec_paths = args.vec_paths.split('+')
    bin_paths = split_paths(args.bin_paths, vec_paths)

    for vec_path, bin_path in zip(vec_paths, bin_paths):
        logging.info('Конвертирую текстовый формат w2v в бинарный из {} в {}'.format(vec_path, bin_path))
        model = load_embeddings(vec_path)
        model.save_word2vec_format(bin_path, binary=True)
        if args.remove_source:
            logging.info('Удаляю исходный файл {}'.format(vec_path))
            os.remove(vec_path)


if __name__ == '__main__':
    main()
