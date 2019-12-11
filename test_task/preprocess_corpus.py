"""
Принимаем на вход путь к папке с корпусом и путь к модели
Сохраняем в той же папке json с лемматизированныеми текстами по названиям файлов

python preprocess_corpus.py texts/ruwiki models/ru.udpipe
python preprocess_corpus.py texts/enwiki models/en.udpipe
python preprocess_corpus.py texts/enwiki models/en.udpipe --keep_stops=True
"""

import argparse
import os
import sys
from json import dump as jdump, load as jload

from tqdm import tqdm
from ufal.udpipe import Model, Pipeline

from preprocess import unify_sym, process, stop_pos


def get_udpipe_lemmas(string, keep_pos, keep_punct, keep_stops):
    # принимает строку, возвращает список токенов
    res = unify_sym(string.strip())
    output = process(process_pipeline, res, keep_pos, keep_punct)
    # убираем всё, что получило стоп-тег
    if not keep_stops:
        clean_output = [lem for lem in output if lem.split('_')[-1] not in stop_pos]
        return clean_output
    return output


# Превращаем список токенов в список лемм с pos-тегами (этот формат нужен для предобученной модели с rusvectores)


def process_corpus(texts_path, files, keep_pos, keep_punct, keep_stops):
    for file in tqdm(files):  # для каждого файла в списке файлов (с прогресс-баром)
        # читаем файл и приводим его к нижнему регистру
        text = open('{}/{}'.format(texts_path, file), encoding='utf-8').read().lower().strip().splitlines()
        # превращаем список токенов в список лемм с pos-тегами # как обрабатывает слова с ударениями?
        lems = []  # придётся экстендить, поэтому без генератора \\есть способ?
        for line in text:
            line_lems = get_udpipe_lemmas(line, keep_pos, keep_punct, keep_stops)
            if line_lems:  # если не пустая строка
                lems.extend(line_lems)
        lemmatized[file] = lems
        # TODO: может быть пустой список, или вряд ли?
        # print('\nЛемматизировал {}'.format(file))
    # ничего не возвращаем, только добавляем в lemmatized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Лемматизация корпуса и сохранение его в json')
    parser.add_argument('texts_path', type=str,
                        help='Папка, в которой лежат тексты в формате txt')
    parser.add_argument('model_path', type=str,
                        help='Папка, в которой лежит модель udpipe для обработки корпуса')
    parser.add_argument('--keep_pos', type=bool, default=True,
                        help='Возвращать ли леммы, помеченные pos-тегами (default: True)')
    parser.add_argument('--keep_stops', type=bool, default=False,
                        help='Сохранять ли слова, получившие тег функциональной части речи (default: False)')
    parser.add_argument('--keep_punct', type=bool, default=False,
                        help='Сохранять ли знаки препинания (default: False)')
    args = parser.parse_args()

    lemmatized_path = '{}/lemmatized.json'.format(args.texts_path)

    files = [f for f in os.listdir(args.texts_path) if f.endswith('.txt')]
    print(len(files), files, file=sys.stderr)

    # TODO: сделать возможность принудительного обновления
    # TODO: страховку от одинаковых названий
    if os.path.isfile(lemmatized_path):  # если существует уже разбор каких-то файлов
        lemmatized = jload(open(lemmatized_path, encoding='utf-8'))
        print('Уже что-то разбирали!', file=sys.stderr)
    else:  # ничего ещё из этого корпуса не разбирали
        lemmatized = {}
        print('Ничего ещё не разбирали, сейчас будем', file=sys.stderr)

    new_files = [file for file in files if file.endswith('txt') and file not in lemmatized]
    # если txt, который ещё не разбирали
    print('Новых текстов: {}'.format(len(new_files)), file=sys.stderr)
    if new_files:  # если есть, что разобрать
        model = Model.load(args.model_path)
        process_pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        process_corpus(args.texts_path, new_files,
                       keep_pos=args.keep_pos, keep_punct=args.keep_punct, keep_stops=args.keep_stops)
        # TODO: сделать защиту от прерывания, или ну её?
        jdump(lemmatized, open(lemmatized_path, 'w', encoding='utf-8'))
