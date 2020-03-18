'''
Перевод текстов и сохранение переведённых в новую папку

python translate_corpus.py --src_texts_path=texts/ruwiki --translated_texts_path=texts/ruwiki_translated --udpipe_path=../models/ru.udpipe --bidict_path=../words/ru-en_lem.txt --lemmatized_path=texts/trans_pos_lemmatized.json
'''

import argparse
import os
import sys
from json import dump as jdump

from tqdm import tqdm
from ufal.udpipe import Model, Pipeline

from utils.loaders import load_bidict
from utils.preprocessing import process_unified, translate_line


def parse_args():
    parser = argparse.ArgumentParser(
        description='Перевод корпуса по двуязычному словарю и сохранение файлов, а также '
                    'создание словаря лемматизированных текстов')
    parser.add_argument('--src_texts_path', type=str, required=True,
                        help='Папка, в которой лежат тексты')
    parser.add_argument('--translated_texts_path', type=str, required=True,
                        help='Папка, в которой будут сохранятся переведённые тексты')
    parser.add_argument('--udpipe_path', type=str, required=True,
                        help='Путь к модели udpipe для обработки корпуса')
    parser.add_argument('--bidict_path', type=str, required=True,
                        help='Путь к файлу txt, содержащему пары слово-перевод')
    parser.add_argument('--keep_pos', type=int, default=1,
                        help='Возвращать ли леммы, помеченные pos-тегами (0|1; default: 1)')
    parser.add_argument('--keep_stops', type=int, default=0,
                        help='Сохранять ли слова, получившие тег функциональной части речи '
                             '(0|1; default: 0)')
    parser.add_argument('--keep_punct', type=int, default=0,
                        help='Сохранять ли знаки препинания (0|1; default: 0)')
    parser.add_argument('--lemmatized_path', type=str,
                        help='Путь, куда сохранится json с лемматизированными текстами '
                             '(0|1; default: 0)')
    return parser.parse_args()


def create_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def translate_text(src_lines, trans_dict, pipeline, keep_pos, keep_punct, keep_stops):
    translated_text = []

    for line in src_lines:
        #TODO: проверять оба варианта
        # лемматизируем текст перед переводом, чтобы больше слов нашлось в словаре
        src_lems = process_unified(line, pipeline, keep_pos, keep_punct, keep_stops)
        translated_line = translate_line(src_lems, trans_dict)

        if translated_line:  # если что-то перевелось
            translated_text.extend(translated_line)

    return translated_text


def translate_corpus(texts_path, translated_path, files, trans_dict,
                     pipeline, keep_pos, keep_punct, keep_stops):
    create_dir(translated_path)  # создаём папку для переводов, если её ещё нет

    lemmatized = {}
    not_translated = []

    for i, file in tqdm(enumerate(files)):
        text = open('{}/{}'.format(texts_path, file), encoding='utf-8').\
            read().lower().strip().splitlines()
        translated_text = translate_text(text, trans_dict, pipeline, keep_pos, keep_punct, keep_stops)

        if translated_text:
            lemmatized[file] = translated_text
            open('{}/{}'.format(translated_path, file), 'w', encoding='utf-8').\
                write(' '.join(translated_text))

        else:
            not_translated.append(file)
            print('Не удалось перевести!')

    return lemmatized, not_translated


def main():
    args = parse_args()

    bidict = load_bidict(args.bidict_path)

    all_files = [f for f in os.listdir(args.src_texts_path) if f.endswith('.txt')]

    udpipe_model = Model.load(args.udpipe_path)
    process_pipeline = Pipeline(
        udpipe_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

    lemmatized_texts, not_translated_texts = translate_corpus(
        args.src_texts_path, args.translated_texts_path, all_files, bidict, process_pipeline,
        args.keep_pos, args.keep_punct, args.keep_stops)

    jdump(lemmatized_texts, open(args.lemmatized_path, 'w', encoding='utf-8'))

    print(not_translated_texts)
    if not_translated_texts:
        print('Не удалось перевести следующие файлы:\n{}'.
              format('\n'.join(not_translated_texts)), file=sys.stderr)


if __name__ == '__main__':
    main()
