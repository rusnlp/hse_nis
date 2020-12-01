"""
Принимаем на вход путь к папке с корпусом и путь к модели
Сохраняем json с лемматизированныеми текстами по названиям файлов
"""

import argparse
import os
import sys
from json import dump as jdump, load as jload

from tqdm import tqdm
from ufal.udpipe import Model, Pipeline

from utils.arguments import check_args
from utils.preprocessing import process_unified


def parse_args():
    parser = argparse.ArgumentParser(
        description='Лемматизация корпуса и сохранение его в json')
    parser.add_argument('--texts_path', type=str, required=True,
                        help='Папка, в которой лежат тексты')
    parser.add_argument('--udpipe_path', type=str,
                        help='Путь к модели udpipe для обработки корпуса')
    parser.add_argument('--lemmatized_path', type=str, required=True,
                        help='Путь к файлу json, в который будут сохраняться лемматизированные файлы.'
                             ' Если файл уже существует, он будет пополняться')
    parser.add_argument('--keep_pos', type=int, default=1,
                        help='Возвращать ли леммы, помеченные pos-тегами (0|1; default: 1)')
    parser.add_argument('--keep_stops', type=int, default=0,
                        help='Сохранять ли слова, получившие тег функциональной части речи '
                             '(0|1; default: 0)')
    parser.add_argument('--keep_punct', type=int, default=0,
                        help='Сохранять ли знаки препинания (0|1; default: 0)')
    parser.add_argument('--forced', type=int, default=0,
                        help='Принудительно лемматизировать весь корпус заново (0|1; default: 0)')
    parser.add_argument('--preprocessed', type=int, default=0,
                        help='Лежат ли в папке уже предобработанные тексты (0|1; default: 0)')
    return parser.parse_args()


def load_lemmatized(lemmatized_path, forced):
    """
    :param lemmatized_path: путь к словарю заголовков и лемм (строка)
    :param forced: принудительно лемматизировать весь корпус заново (pseudo-boolean int)
    :return: словарь заголовков и лемм
    """
    # если существует уже разбор каких-то файлов
    if os.path.isfile(lemmatized_path) and not forced:
        lemmatized = jload(open(lemmatized_path, encoding='utf-8'))
        print('Уже что-то разбирали!', file=sys.stderr)

    else:  # ничего ещё из этого корпуса не разбирали или принудительно обновляем всё
        lemmatized = {}
        print('Ничего ещё не разбирали, сейчас будем', file=sys.stderr)

    return lemmatized


def collect_texts(lemmatized, texts_path, files):
    '''собираем тексты из папки в словарь'''
    for file in tqdm(files):
        text = open('{}/{}'.format(texts_path, file), encoding='utf-8').\
            read().lower().strip().split()
        lemmatized[file] = text
    return lemmatized


def process_text(pipeline, text_lines, keep_pos, keep_punct, keep_stops):
    """
    :param pipeline: пайплайн udpipe
    :param text_lines: строки документа на лемматизацию (список строк, могут быть пустые списки)
    :param keep_pos: оставлять ли pos-теги (pseudo-boolean int)
    :param keep_punct: оставлять ли пунктуацию (pseudo-boolean int)
    :param keep_stops: оставлять ли стоп-слова (pseudo-boolean int)
    :return: леммы текста в виде "токен_pos" (список строк)
    """
    text_lems = []  # придётся экстендить, а с генератором плохо читается

    for line in text_lines:
        line_lems = process_unified(line, pipeline, keep_pos, keep_punct, keep_stops)
        if line_lems:  # если не пустая строка
            text_lems.extend(line_lems)

    return text_lems


def process_corpus(pipeline, lemmatized, texts_path, files, keep_pos, keep_punct, keep_stops):
    """
    :param pipeline: пайплайн udpipe
    :param lemmatized: заголовки и лемматизированные тексты (словарь)
    :param texts_path: путь к папке с текстами (строка)
    :param files: заголовки текстов, которые надо лемматизировать (список строк)
    :param keep_pos: оставлять ли pos-теги (pseudo-boolean int)
    :param keep_punct: оставлять ли пунктуацию (pseudo-boolean int)
    :param keep_stops: оставлять ли стоп-слова (pseudo-boolean int)
    :return lemmatized: обновленный словарь заголовков и лемм (словарь)
    :return not_lemmatized: заголовки текстов, которые не удалось лемматизировать (список строк)
    """
    not_lemmatized = []

    for file in tqdm(files):
        text = open('{}/{}'.format(texts_path, file), encoding='utf-8').\
            read().lower().strip().splitlines()
        text_lems = process_text(pipeline, text, keep_pos, keep_punct, keep_stops)

        if text_lems:
            lemmatized[file] = text_lems

        else:  # с текстом что-то не так, и там не остаётся нормальных лемм
            not_lemmatized.append(file)
            continue

    return lemmatized, not_lemmatized


def main():
    args = parse_args()

    preprocessed_required = {
        0: ['udpipe_path']
    }
    # Проверяем, всё ли указали для непредобработанных статей
    check_args(args, 'preprocessed', preprocessed_required)

    all_files = [f for f in os.listdir(args.texts_path)]

    lemmatized_dict = load_lemmatized(args.lemmatized_path, args.forced)

    new_files = [file for file in all_files if file not in lemmatized_dict]
    print('Новых текстов: {}'.format(len(new_files)), file=sys.stderr)

    if new_files:

        if args.preprocessed:  # если файлы уже предобработаны
            full_lemmatized_dict = collect_texts(lemmatized_dict, args.texts_path, new_files)

        else:
            udpipe_model = Model.load(args.udpipe_path)
            process_pipeline = Pipeline(
                udpipe_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

            full_lemmatized_dict, not_lemmatized_texts = process_corpus(
                process_pipeline, lemmatized_dict, args.texts_path, new_files,
                keep_pos=args.keep_pos, keep_punct=args.keep_punct, keep_stops=args.keep_stops)

            if not_lemmatized_texts:
                print('Не удалось разобрать следующие файлы:\n{}'.
                      format('\n'.join(not_lemmatized_texts)), file=sys.stderr)

        jdump(full_lemmatized_dict, open(args.lemmatized_path, 'w', encoding='utf-8'))


if __name__ == "__main__":
    main()
