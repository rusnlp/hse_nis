"""
Считаем, что все тексты, которые ищем, априори добавлены в корпус, предобработаны,
вектора для всего построены

Подаём название текста, язык, путь к векторизованному корпусу, путь к маппингу, можно подать
кол-во ближайших статей
Получаем n ближайших записей в виде списка кортежей (заголовок, близость) -- напечатем рейтинг,
если не сделали verbose=False

python monolang_search.py --target_article_path=кровь --lang=ru
--mapping_path=texts/titles_mapping.json --corpus_embeddings_path=texts/ruwiki/simple.pkl --top=10
python monolang_search.py --target_article_path=blood --lang=en
--mapping_path=texts/titles_mapping.json --corpus_embeddings_path=texts/enwiki/simple.pkl --top=10

python monolang_search.py --target_article_path=texts/ruwiki/бензин.txt --lang=ru
--mapping_path=texts/titles_mapping.json --corpus_embeddings_path=texts/ruwiki/simple.pkl --top=10
--included=0 --udpipe_path=models/ru.udpipe --model_embeddings_path=models/ru.bin
python monolang_search.py --target_article_path=texts/enwiki/gasoline.txt --lang=en
--mapping_path=texts/titles_mapping.json --corpus_embeddings_path=texts/enwiki/simple.pkl --top=10
--included=0 --udpipe_path=models/en.udpipe --model_embeddings_path=models/en.bin
"""

import argparse
from json import load as jload
from pickle import load as pload

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Векторизация корпуса и сохранение его в pkl в папке с текстами')
    parser.add_argument('--target_article_path', type=str, required=True,
                        help='Путь к статье в формате txt, для которой ищем ближайшие.'
                             '\nЕсли статья из корпуса, то только назание без формата')
    parser.add_argument('--lang', type=str, required=True,
                        help='Язык, для которого разбираем, '
                             'нужен для определения словаря в маппинге (ru/en)')
    parser.add_argument('--mapping_path', type=str, required=True,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('--corpus_embeddings_path', type=str, required=True,
                        help='Путь к файлу pkl, в котором лежит векторизованный корпус')
    parser.add_argument('--top', type=int, default=1,
                        help='Сколько близких статeй возвращать (default: 1; -1 for all)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Принтить ли рейтинг (0|1; default: 0)')
    # ДЛЯ ВНЕШНИХ ТЕКСТОВ:
    parser.add_argument('--included', type=int, default=1,
                        help='Включена ли статья в корпус (0|1; default: 1)')
    parser.add_argument('--udpipe_path', type=str, default='',
                        help='Папка, в которой лежит модель udpipe для обработки нового текста')
    parser.add_argument('--keep_pos', type=int, default=1,
                        help='Возвращать ли леммы, помеченные pos-тегами (0|1; default: 1)')
    parser.add_argument('--keep_stops', type=int, default=0,
                        help='Сохранять ли слова, получившие тег функциональной части речи '
                             '(0|1; default: 0)')
    parser.add_argument('--keep_punct', type=int, default=0,
                        help='Сохранять ли знаки препинания (0|1; default: 0)')
    parser.add_argument('--model_embeddings_path', type=str, default='',
                        help='Папка, в которой лежит модель для векторизации корпуса')
    parser.add_argument('--no_duplicates', type=int, default=0,
                        help='Брать ли для каждого типа в тексте вектор только по одному разу '
                             '(0|1; default: 0)')

    return parser.parse_args()


class NotIncludedError(Exception):
    def __init__(self):
        self.text = 'Такого текста нет в корпусе! Пожалуйста, измените значение параметра included'

    def __str__(self):
        return self.text


class NoModelProvided(Exception):
    def __init__(self):
        self.text = 'Пожалуйста, укажите пути к моделям для лемматизации и векторизации текста!'

    def __str__(self):
        return self.text


def prepare_new_article(article_path, udpipe_path, model_embeddings_path,
                        keep_pos, keep_punct, keep_stops, no_duplicates):
    from ufal.udpipe import Model, Pipeline
    from preprocess_corpus import process_text
    from vectorize_corpus import load_embeddings, vectorize_text

    udpipe_model = Model.load(udpipe_path)
    pipeline = Pipeline(udpipe_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

    text = open(article_path, encoding='utf-8').read().lower().strip().splitlines()
    lems = process_text(pipeline, text, keep_pos, keep_punct, keep_stops)

    emb_model = load_embeddings(model_embeddings_path)
    emb_model = emb_model.wv

    article_vec = vectorize_text(lems, emb_model, no_duplicates)

    return article_path, article_vec


# Для каждого текста в корпусе считаем косинусную близость к данному
def search_similar(target_vec, corpus_vecs):  # подаём текст как вектор и векторизованный корпус
    sim_vecs = np.dot(corpus_vecs, target_vec)
    # словарь {индекс текста в корпусе: близость к данному}
    similars = {i: sim for i, sim in enumerate(sim_vecs)}

    return similars


# Ранжируем тексты по близости и принтим красивый списочек
# принимаем словарь близостей
def make_rating(target_article, similars, verbose, n, included, texts_mapping, i2lang):
    # сортируем словарь по значениям в порядке убывания:
    # сортируем список кортежей (key, value) по value
    sorted_simkeys = sorted(similars, key=similars.get, reverse=True)
    # similars: [i, j, ...]
    similars_list = [(texts_mapping[i2lang].get(str(simkey)), similars[simkey])
                     for simkey in sorted_simkeys]
    # [(i_title, sim), (j_title, sim), (...)]
    if included:  # если статья включена в корпус
        # на 0 индексе всегда будет сама статья, если она из корпуса
        similars_list = similars_list[1:]

    if n == -1:  # нужен вывод всех статей
        if verbose:
            print('\nРейтинг статей по близости к {}:'.format(n, target_article))
            for i, sim_item in enumerate(similars_list[:n]):
                print('{}. {} ({})'.format(i + 1, sim_item[0], sim_item[1]))
        return similars_list

    else:
        if verbose:
            print('\nТоп-{} ближайших статей к {}:'.format(n, target_article))
            for i, sim_item in enumerate(similars_list[:n]):
                print('{}. {} ({})'.format(i + 1, sim_item[0], sim_item[1]))
        return similars_list[:n]  # если нужна будет одна статья, вернётся список с одним элементом


def main():
    args = parse_args()

    i2lang = 'i2{}'.format(args.lang)
    lang2i = '{}2i'.format(args.lang)
    texts_mapping = jload(open(args.mapping_path))
    corpus_vecs = pload(open(args.corpus_embeddings_path, 'rb'))

    if args.included:
        target_article_title = args.target_article_path.lower()
        target_article = '{}.txt'.format(target_article_title)
        target_article_id = texts_mapping[lang2i].get(target_article)

        if not target_article_id:
            raise NotIncludedError

        target_article_vec = corpus_vecs[target_article_id]

    else:
        if not args.udpipe_path or not args.model_embeddings_path:
            raise NoModelProvided

        target_article, target_article_vec = prepare_new_article(
            args.target_article_path, args.udpipe_path, args.model_embeddings_path, args.keep_pos,
            args.keep_punct, args.keep_stops, args.no_duplicates)

    similars = search_similar(target_article_vec, corpus_vecs)
    make_rating(target_article, similars, n=args.top, verbose=args.verbose, included=args.included,
                texts_mapping=texts_mapping, i2lang=i2lang)


if __name__ == "__main__":
    main()
