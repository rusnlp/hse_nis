"""
Принимаем на вход путь к папке с корпусом и путь к модели
Сохраняем json с лемматизированныеми текстами по названиям файлов

python preprocess_corpus.py --texts_path=texts/ruwiki --udpipe_path=models/ru.udpipe --lemmatized_path=texts/ruwiki/lemmatized.json --forced=1
python preprocess_corpus.py --texts_path=texts/enwiki --udpipe_path=models/en.udpipe --lemmatized_path=texts/enwiki/lemmatized.json
"""

import argparse
import os
import sys
from json import dump as jdump, load as jload

from tqdm import tqdm
from ufal.udpipe import Model, Pipeline

from preprocess import unify_sym, process, stop_pos


def parse_args():
    parser = argparse.ArgumentParser(
        description='Лемматизация корпуса и сохранение его в json')
    parser.add_argument('--texts_path', type=str, required=True,
                        help='Папка, в которой лежат тексты в формате txt')
    parser.add_argument('--udpipe_path', type=str, required=True,
                        help='Путь к модели udpipe для обработки корпуса')
    parser.add_argument('--lemmatized_path', type=str, required=True,
                        help='Путь к файлу json, в который будут сохраняться лемматизированные файлы. '
                             'Если файл уже существует, он будет пополняться')
    parser.add_argument('--keep_pos', type=int, default=1,
                        help='Возвращать ли леммы, помеченные pos-тегами (0|1; default: 1)')
    parser.add_argument('--keep_stops', type=int, default=0,
                        help='Сохранять ли слова, получившие тег функциональной части речи (0|1; default: 0)')
    parser.add_argument('--keep_punct', type=int, default=0,
                        help='Сохранять ли знаки препинания (0|1; default: 0)')
    parser.add_argument('--forced', type=int, default=0,
                        help='Принудительно лемматизировать весь корпус заново (0|1; default: 0)')

    return parser.parse_args()


def get_udpipe_lemmas(pipeline, text_string, keep_pos, keep_punct, keep_stops):
    # принимает строку, возвращает список токенов
    text_string = unify_sym(text_string.strip())
    output = process(pipeline, text_string, keep_pos, keep_punct)
    if not keep_stops:
        clean_output = [lem for lem in output if lem.split('_')[-1] not in stop_pos]
        return clean_output
    return output


# Превращаем список строк дока в список лемм с pos-тегами (этот формат нужен для предобученной модели с rusvectores)
def process_text(pipeline, text, keep_pos, keep_punct, keep_stops):
    # превращаем список строк в список лемм с pos-тегами
    lems = []  # придётся экстендить, а с генератором плохо читается
    for line in text:
        line_lems = get_udpipe_lemmas(pipeline, line, keep_pos, keep_punct, keep_stops)
        if line_lems:  # если не пустая строка
            lems.extend(line_lems)
    return lems


def process_corpus(pipeline, lemmatized, texts_path, files, keep_pos, keep_punct, keep_stops):
    # добавляем в словарь lemmatized, возвращаем lemmatized и список файлов, которые не получилось лемматизировать
    not_lemmatized = []
    for file in tqdm(files):
        text = open('{}/{}'.format(texts_path, file), encoding='utf-8').read().lower().strip().splitlines()
        lems = process_text(pipeline, text, keep_pos, keep_punct, keep_stops)
        if lems:
            lemmatized[file] = lems
        else:  # с текстом что-то не так, и там не остаётся нормальных лемм
            not_lemmatized.append('{}/{}'.format(texts_path, file))
            continue
    return lemmatized, not_lemmatized


def main():
    args = parse_args()

    files = [f for f in os.listdir(args.texts_path) if f.endswith('.txt')]

    if os.path.isfile(args.lemmatized_path) and not args.forced:  # если существует уже разбор каких-то файлов
        lemmatized = jload(open(args.lemmatized_path, encoding='utf-8'))
        print('Уже что-то разбирали!', file=sys.stderr)
    else:  # ничего ещё из этого корпуса не разбирали или принудительно обновляем всё
        lemmatized = {}
        print('Ничего ещё не разбирали, сейчас будем', file=sys.stderr)

    new_files = [file for file in files if file.endswith('txt') and file not in lemmatized]
    print('Новых текстов: {}'.format(len(new_files)), file=sys.stderr)
    if new_files:
        model = Model.load(args.udpipe_path)
        pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

        lemmatized, not_lemmatized = process_corpus(pipeline, lemmatized, args.texts_path, new_files,
                                                    keep_pos=args.keep_pos, keep_punct=args.keep_punct,
                                                    keep_stops=args.keep_stops)

        jdump(lemmatized, open(args.lemmatized_path, 'w', encoding='utf-8'))

        if not_lemmatized:
            print('Не удалось разобрать следующие файлы:\n{}'.format('\n'.join(not_lemmatized)), file=sys.stderr)


if __name__ == "__main__":
    main()
