"""
python create_mapping.py --texts_path=../texts_conf/texts/ru_conllu+../texts_conf/texts/en_conllu --lang=ru+en --mapping_path=../texts_conf/mapping.json
"""

import argparse
from json import load, dump
import os
from utils.preprocessing import clean_ext


def parse_args():
    parser = argparse.ArgumentParser(
        description='Создание маппинга индексов в заголовки и обратно и сохранение его в json')
    parser.add_argument('--texts_path', type=str, required=True,
                        help='Путь к текстам в формате conllu (можно перечислить через +)')
    parser.add_argument('--lang', type=str, required=True,
                        help='Языки, для которых разбираем (можно перечислить через +); '
                             'нужен для определения словаря в маппинге (ru/en')
    parser.add_argument('--mapping_path', type=str, required=True,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('--forced', type=int, default=0,
                        help='Принудительно пересоздать весь маппинг (0|1; default: 0)')

    return parser.parse_args()


def split_paths(joint_path, texts_paths):
    # делим пути по + или задаём столько пустых, сколько пришло папок с текстами
    if joint_path:
        paths = joint_path.split('+')
    else:
        paths = [''] * len(texts_paths)
    return paths


def main():
    args = parse_args()

    texts_paths = args.texts_path.split('+')
    langs = split_paths(args.lang, texts_paths)

    if os.path.isfile(args.mapping_path) and not args.forced:
        mapping = load(open(args.mapping_path, 'r', encoding='utf-8'))
        print('Уже есть какой-то маппинг!')
        print('\t'.join(['{}: {} объекта'.format(k, len(v)) for k, v in mapping.items()]))
    else:
        mapping = {}
        print('Маппинга ещё нет, сейчас будет')

    for texts_path, lang in zip(texts_paths, langs):
        print('Добавляю язык {}'.format(lang))
        i2lang = 'i2{}'.format(lang)
        lang2i = '{}2i'.format(lang)
        files = [file for file in os.listdir(texts_path)]
        mapping[i2lang] = {i: clean_ext(file) for i, file in enumerate(files)}
        mapping[lang2i] = {clean_ext(file): i for i, file in enumerate(files)}

    print('Новый маппинг:')
    print('\n'.join(['{}: {} объекта'.format(k, len(v)) for k, v in mapping.items()]))
    dump(mapping, open(args.mapping_path, 'w', encoding='utf-8'))


if __name__ == '__main__':
    main()
