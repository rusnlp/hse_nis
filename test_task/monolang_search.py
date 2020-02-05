"""
Считаем, что все тексты, которые ищем, априори добавлены в корпус, предобработаны, вектора для всего построены

Подаём название текста, язык, путь к векторизованному корпусу, путь к маппингу, можно подать кол-во ближайших статей
Получаем n ближайших записей в виде списка кортежей (заголовок, близость)
Напечатем рейтинг, если не поставили verbose=False

python monolang_search.py --target_article_path=кровь --lang=ru --mapping_path=texts/titles_mapping.json --corpus_embeddings_path=texts/ruwiki/simple.pkl --top=10
python monolang_search.py --target_article_path=blood --lang=en --mapping_path=texts/titles_mapping.json --corpus_embeddings_path=texts/enwiki/simple.pkl --top=10

python monolang_search.py --target_article_path=texts/ruwiki/бензин.txt --lang=ru --mapping_path=texts/titles_mapping.json --corpus_embeddings_path=texts/ruwiki/simple.pkl --top=10 --included=0 --udpipe_path=models/ru.udpipe --model_embeddings_path=models/ru.bin
python monolang_search.py --target_article_path=texts/enwiki/gasoline.txt --lang=en --mapping_path=texts/titles_mapping.json --corpus_embeddings_path=texts/enwiki/simple.pkl --top=10 --included=0 --udpipe_path=models/en.udpipe --model_embeddings_path=models/en.bin
"""

import argparse
from json import load as jload
from pickle import load as pload

import numpy as np


def parse_args():
    """
    :return: объект со всеми аршументами (argparse.Namespace)
    """
    parser = argparse.ArgumentParser(
        description='Ранжирование статей на основе косинусной близости векторов')
    parser.add_argument('--target_article_path', type=str, required=True,
                        help='Путь к статье в формате txt, для которой ищем ближайшие.'
                             '\nЕсли статья из корпуса, то только назание без формата')
    parser.add_argument('--lang', type=str, required=True,
                        help='Язык, для которго разбираем, нужен для определения словаря '
                             'в маппинге (ru/en)')
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
    """
    Указали included=True для текста, которого нет в маппинге
    """
    def __init__(self):
        self.text = 'Текста с таким названием нет в корпусе! ' \
                    'Пожалуйста, измените значение параметра included'

    def __str__(self):
        return self.text


class NoModelProvided(Exception):
    """
    Забыли указать путь к модели лемматизации или векторизации
    """
    def __init__(self):
        self.text = 'Пожалуйста, укажите пути к моделям для лемматизации и векторизации текста.'

    def __str__(self):
        return self.text


def prepare_new_article(article_path, udpipe_path, model_embeddings_path,
                        keep_pos, keep_punct, keep_stops, no_duplicates):
    """
    :param article_path: путь к новому тексту (строка)
    :param udpipe_path: путь к модели udpipe для лемматизации (строка)
    :param model_embeddings_path: путь к модели для векторизации (строка)
    :param keep_pos: оставлять ли pos-теги (pseudo-boolean int)
    :param keep_punct: оставлять ли пунктуацию (pseudo-boolean int)
    :param keep_stops: оставлять ли стоп-слова (pseudo-boolean int)
    :param no_duplicates: векторизовать по множеству слов, а не по спику (pseudo-boolean)
    :return: вектор нового текста (np.array)
    """
    import logging
    from ufal.udpipe import Model, Pipeline
    from preprocess_corpus import process_text
    from vectorize_corpus import load_embeddings, vectorize_text

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    article_title = article_path.split('/')[-1]

    logging.info('Загружаю модель {} для обработки нового текста'.format(udpipe_path))
    udpipe_model = Model.load(udpipe_path)
    logging.info('Загрузил модель {}'.format(udpipe_path))
    process_pipeline = Pipeline(udpipe_model, 'tokenize', Pipeline.DEFAULT,
                                Pipeline.DEFAULT, 'conllu')

    logging.info('Лемматизирую текст {}'.format(article_path))
    text = open(article_path, encoding='utf-8').read().lower().strip().splitlines()
    lems = process_text(process_pipeline, text, keep_pos, keep_punct, keep_stops)
    logging.info('Лемматризировал текст {}, приступаю к векторизации'.format(article_path))

    emb_model = load_embeddings(model_embeddings_path)
    emb_model = emb_model.wv

    article_vec = vectorize_text(lems, emb_model, no_duplicates)

    return article_title, article_vec


def search_similar(vector, vectors):
    """
    :param vector: вектор текста (np.array)
    :param vectors: матрица векторов корпуса (np.array)
    :return: индексы текстов и близость к данному (словарь)
    """
    sim_vecs = np.dot(vectors, vector)
    sim_dict = {i: sim for i, sim in enumerate(sim_vecs)}

    return sim_dict


def make_rating(target_article, sim_dict, verbose, n, included, mapping, i2lang_name):
    """
    :param target_article: заголовок статьи, для которой ищем ближайшие (строка)
    :param sim_dict: индексы статей в корпусе и близость к данной (словарь)
    :param verbose: принтить ли рейтинг (pseudo-boolean int)
    :param n: сколько ближайших статей получать (int)
    :param included: включена ли статья в корпус (pseudo-boolean int)
    :param mapping: маппинг заголовков в индексы и обратно (словарь словарей)
    :param i2lang_name: название словаря индекс-заголовок в маппинге (строка)
    :return: индексы текстов и близость к данному в порядке убывания (список кортежей)
    """
    sorted_simkeys = sorted(sim_dict, key=sim_dict.get, reverse=True)
    sim_list = [(mapping[i2lang_name].get(str(simkey)), sim_dict[simkey])
                for simkey in sorted_simkeys]

    if included:
        # на 0 индексе всегда будет сама статья, если она из корпуса
        sim_list.pop(0)

    if n == -1:  # нужен вывод всех статей
        if verbose:
            print('\nРейтинг статей по близости к {}:'.format(n, target_article))
            for i, sim_item in enumerate(sim_list[:n]):
                print('{}. {} ({})'.format(i + 1, sim_item[0], sim_item[1]))
        return sim_list

    else:
        if verbose:
            print('\nТоп-{} ближайших статей к {}:'.format(n, target_article))
            for i, sim_item in enumerate(sim_list[:n]):
                print('{}. {} ({})'.format(i + 1, sim_item[0], sim_item[1]))
        return sim_list[:n]  # если нужна будет одна статья, вернётся список с одним элементом


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
    rating = make_rating(target_article, similars, args.verbose, args.top, args.included,
                         texts_mapping, i2lang)


if __name__ == "__main__":
    main()
