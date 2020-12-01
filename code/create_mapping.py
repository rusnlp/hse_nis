"""
"""

import argparse
from json import load, dump
import os
from utils.preprocessing import clean_ext


def parse_args():
    parser = argparse.ArgumentParser(
        description='Создание маппинга индексов в заголовки и обратно и сохранение его в json')
    parser.add_argument('--texts_path', type=str, required=True,
                        help='Папка, в которой лежат тексты')
    parser.add_argument('--lang', type=str, required=True,
                        help='Язык, для которого разбираем; '
                             'нужен для определения словаря в маппинге (ru/en')
    parser.add_argument('--mapping_path', type=str, required=True,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('--forced', type=int, default=0,
                        help='Принудительно пересоздать весь маппинг (0|1; default: 0)')

    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isfile(args.mapping_path) and not args.forced:
        mapping = load(open(args.mapping_path, 'r', encoding='utf-8'))
        print('Уже есть какой-то маппинг!')
        print('\t'.join(['{}: {} объекта'.format(k, len(v)) for k, v in mapping.items()]))
    else:
        mapping = {}
        print('Маппинга ещё нет, сейчас будет')

    i2lang = 'i2{}'.format(args.lang)
    lang2i = '{}2i'.format(args.lang)
    files = [file for file in os.listdir(args.texts_path)]
    mapping[i2lang] = {i: clean_ext(file) for i, file in enumerate(files)}
    mapping[lang2i] = {clean_ext(file): i for i, file in enumerate(files)}
    print('Новый маппинг:')
    print('\t'.join(['{}: {} объекта'.format(k, len(v)) for k, v in mapping.items()]))
    dump(mapping, open(args.mapping_path, 'w', encoding='utf-8'))


if __name__ == '__main__':
    main()
