'''
Принимаем на вход путь к папке с корпусом и путь к модели
Сохраняем в той же папке json с лемматизированныеми текстами по названиям файлов

python preprocess_corpus.py ru texts/ruwiki models/ru.udpipe
python preprocess_corpus.py en texts/enwiki models/en.udpipe
'''

from preprocess import unify_sym, process, stop_pos
from ufal.udpipe import Model, Pipeline
from tqdm import tqdm
from json import dump as jdump, load as jload
import os
import sys
import argparse

def get_udpipe_lemmas(string, keep_pos=True, keep_punct=False, keep_stops=False):
    # принимает строку, возвращает список токенов
    res = unify_sym(string.strip())
    output = process(process_pipeline, res, keep_pos, keep_punct)
    if not keep_stops:
        clean_output = [lem for lem in output if lem.split('_')[-1] not in stop_pos] # убираем всё, что получило стоп-тег
        return clean_output
    return output
# Превращаем список токенов в список лемм с pos-тегами (этот формат нужен для предобученной модели с rusvectores)


def process_corpus(texts_path, files):
    for file in tqdm(files):  # для каждого файла в списке файлов (с прогресс-баром)
        text = open('{}/{}'.format(texts_path, file), encoding='utf-8').read().lower().strip().splitlines()  # читаем файл и приводим его к нижнему регистру
        # превращаем список токенов в список лемм с pos-тегами # как обрабатывает слова с ударениями?
        lems = []  # придётся экстендить, поэтому без генератора \\есть способ?
        for line in text:
            line_lems = get_udpipe_lemmas(line) #убираем стоп-слова
            if line_lems:  # если не пустая строка
                lems.extend(line_lems)
        lemmatized[file] = lems
        #TODO: может быть пустой список, или вряд ли?
        #print('\nЛемматизировал {}'.format(file))
    # ничего не возвращаем, только добавляем в lemmatized

if __name__ == "__main__":
    #TODO: pos-теги и стоп-слова в опциональные аргументы
    parser = argparse.ArgumentParser(description='Лемматизация корпуса и сохранение его в json')
    parser.add_argument('texts_path', type=str, help='Папка, в которой лежат тексты в формате txt')
    parser.add_argument('model_path', type=str, help='Папка, в которой лежит модель udpipe для обработки корпуса')
    args = parser.parse_args()

    lemmatized_path = '{}/lemmatized.json'.format(args.texts_path)

    # TODO: walk тоже нельзя?
    files = next(os.walk(args.texts_path))[-1]
    print(len(files),  files, file=sys.stderr)

    #TODO: сделать возможность принудительного обновления
    #TODO: страховку от одинаковых названий
    if os.path.isfile(lemmatized_path): # если существует уже разбор каких-то файлов
        lemmatized = jload(open(lemmatized_path, encoding='utf-8'))
        print('Уже что-то разбирали!', file=sys.stderr)
    else: # ничего ещё из этого корпуса не разбирали
        lemmatized = {}
        print('Ничего ещё не разбирали, сейчас будем', file=sys.stderr)

    new_files = [file for file in files if file.split('.')[-1] == 'txt' and file not in lemmatized] # если txt, который ещё не разбирали
    print('Новых текстов: {}'.format(len(new_files)), file=sys.stderr)
    if new_files: # если есть, что разобрать
        model = Model.load(args.model_path)
        process_pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        process_corpus(args.texts_path, new_files)
        # TODO: сделать защиту от прерывания, или ну её?
        jdump(lemmatized, open(lemmatized_path, 'w', encoding='utf-8'))
