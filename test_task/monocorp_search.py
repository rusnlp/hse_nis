"""
Считаем, что все тексты, которые ищем, априори добавлены в корпус, предобработаны,
вектора для всего построены

Подаём название текста, язык, путь к векторизованному корпусу, путь к маппингу,
можно подать кол-во ближайших статей
Получаем n ближайших записей в виде списка кортежей (заголовок, близость)
Напечатем рейтинг, если не поставили verbose=False
"""

import argparse
from json import load as jload
from pickle import load as pload

import numpy as np
from utils.arguments import check_args
from utils.loaders import load_article_data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Ранжирование статей на основе косинусной близости векторов')
    parser.add_argument('--target_article_path', type=str, required=True,
                        help='Путь к статье, для которой ищем ближайшие.'
                             '\nЕсли статья из корпуса, то только назание')
    parser.add_argument('--lang', type=str, required=True,
                        help='Язык, для которого разбираем; '
                             'нужен для определения словаря в маппинге (ru/en/cross')
    parser.add_argument('--mapping_path', type=str, required=True,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('--corpus_vectors_path', type=str, required=True,
                        help='Путь к файлу pkl, в котором лежит векторизованный корпус')
    parser.add_argument('--top', type=int, default=1,
                        help='Сколько близких статeй возвращать (default: 1; -1 for all)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Принтить ли рейтинг (0|1; default: 0)')
    parser.add_argument('--with_url', type=int, default=0,
                        help='Добавлять ссылки к заголовкам (0|1; default: 0)')
    parser.add_argument('--url_mapping_path', type=str,
                        help='Путь к файлу маппинга заголовков в ссылки')
    # ДЛЯ ВНЕШНИХ ТЕКСТОВ:
    parser.add_argument('--included', type=int, default=1,
                        help='Включена ли статья в корпус (0|1; default: 1)')
    # ДЛЯ ПРЕДОБРАБОТКИ
    parser.add_argument('--udpipe_path', type=str,
                        help='Папка, в которой лежит модель udpipe для обработки нового текста')
    parser.add_argument('--keep_pos', type=int, default=1,
                        help='Возвращать ли леммы, помеченные pos-тегами (0|1; default: 1)')
    parser.add_argument('--keep_stops', type=int, default=0,
                        help='Сохранять ли слова, получившие тег функциональной части речи '
                             '(0|1; default: 0)')
    parser.add_argument('--keep_punct', type=int, default=0,
                        help='Сохранять ли знаки препинания (0|1; default: 0)')
    # ДЛЯ ВЕКТОРИЗАЦИИ
    parser.add_argument('--method', type=str,
                        help='Метод векторизации (model/translation/projection)')
    parser.add_argument('--embeddings_path', type=str,
                        help='Папка, в которой лежит модель для векторизации текста')
    parser.add_argument('--bidict_path', type=str,
                        help='Путь к двуязычному словарю в формате txt')
    parser.add_argument('--projection_path', type=str,
                        help='Путь к матрице трансформации в формате txt')
    parser.add_argument('--no_duplicates', type=int, default=0,
                        help='Брать ли для каждого типа в тексте вектор только по одному разу '
                             '(0|1; default: 0)')

    return parser.parse_args()


class NotIncludedError(Exception):
    """
    Указали included=True для текста, которого нет в маппинге
    """
    def __init__(self):
        self.text = 'Такого текста нет в корпусе! ' \
                    'Пожалуйста, измените значение параметра included'

    def __str__(self):
        return self.text


class NoModelProvided(Exception):
    """
    Забыли указать путь к модели лемматизации или векторизации
    """
    def __init__(self):
        self.text = 'Пожалуйста, укажите пути ' \
                    'к моделям для лемматизации и векторизации текста.'

    def __str__(self):
        return self.text


def prepare_new_article(article_path, udpipe_path, keep_pos, keep_punct, keep_stops,
                        method, embeddings_path, no_duplicates, projection_path, bidict_path):
    """
    :param article_path: путь к новому тексту (строка)
    :param udpipe_path: путь к модели udpipe для лемматизации (строка)
    :param keep_pos: оставлять ли pos-теги (pseudo-boolean int)
    :param keep_punct: оставлять ли пунктуацию (pseudo-boolean int)
    :param keep_stops: оставлять ли стоп-слова (pseudo-boolean int)
    :param method:
    :param embeddings_path: путь к модели для векторизации (строка)
    :param no_duplicates: векторизовать по множеству слов, а не по спику (pseudo-boolean)
    :param projection_path:
    :param bidict_path:
    :return: вектор нового текста (np.array)
    """
    import logging
    from ufal.udpipe import Model, Pipeline
    from preprocess_corpus import process_text
    from utils.vectorization import build_vectorizer

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

    vectorizer = build_vectorizer('src', method, embeddings_path, no_duplicates,
                                  projection_path, bidict_path)

    article_vec = vectorizer.vectorize_text(lems)

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


def get_lang(title, mapping):
    '''ищем заголовок в словарях маппига и извлекаем язык '''
    lang_dicts = [key for key in mapping.keys() if key.endswith('2i') and 'cross' not in key]
    for lang_dict in lang_dicts:
        if title in mapping[lang_dict]:
            lang = lang_dict.split('2')[0]
            break  # будем надеться, что статей с одинаковым названием нет (хеши-то разные)
    return lang


def get_url_tail(target_article, article_data):
    '''получаем форматированную ссылку на статью, если есть'''
    try:  # если вобще подан article_data и в нём есть target_article
        url_tail = article_data[target_article].get('url', '')
        if url_tail:
            url_tail = ' = {}'.format(url_tail)
        return url_tail
    except:
        return ''


def verbose_rating(target_article, mapping, result_dicts, n, article_data=None):
    verbosed = 'Рейтинг статей по близости к {} ({}):{}\n'\
        .format(target_article, get_lang(target_article, mapping), get_url_tail(target_article, article_data))
    for i, result_dict in enumerate(result_dicts[:n]):
        verbosed += '{}. {} ({}): {}{}\n'.\
              format(i + 1, result_dict['title'], result_dict['lang'], result_dict['sim'], get_url_tail(result_dict['title'], article_data))
    return verbosed


def make_rating(target_article, sim_dict, verbose, n, included, mapping, i2lang_name,
                with_url=0, article_data=None):
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

    if included:  # на 0 индексе всегда будет сама статья, если она из корпуса
        sim_list.pop(0)

    result_dicts = []  # словари с данными ближайших статей
    missed_urls = []  # список статей, пропущенных в hash_title_url
    for i, (title, sim) in enumerate(sim_list):
        result_dict = {'title': title, 'sim': sim, 'lang': get_lang(title, mapping)}
        if with_url:
            try:
                result_dict['url'] = article_data[title].get('url')
                result_dict['real_title'] = article_data[title].get('real_title')
            except KeyError:
                # print(False, title)
                missed_urls.append((result_dict['title'], result_dict['lang']))
        result_dicts.append(result_dict)

    if n == -1:  # нужен вывод всех статей
        n = len(sim_list)

    # вывод всё равно придётся делать для получения в automatic_search
    verbosed_rating = verbose_rating(target_article, mapping, result_dicts, n, article_data)
    if verbose:
        print(verbosed_rating)

    # если нужна будет одна статья, вернётся список с одним элементом
    return result_dicts[:n], verbosed_rating, missed_urls


def main(target_article_path, lang, mapping_path, corpus_vectors_path,
         top=1, verbose=1, included=1, udpipe_path='', keep_pos=1, keep_stops=0, keep_punct=0,
         method='', embeddings_path='', bidict_path='', projection_path='', no_duplicates=0,
         with_url=0, url_mapping_path=''):

    i2lang = 'i2{}'.format(lang)
    lang2i = '{}2i'.format(lang)
    texts_mapping = jload(open(mapping_path))
    corpus_vecs = pload(open(corpus_vectors_path, 'rb'))

    article_data = None
    if with_url:
        article_data = load_article_data(url_mapping_path)

    if included:
        target_article = target_article_path
        target_article_id = texts_mapping[lang2i].get(target_article)

        if target_article_id is None:  # просто not нельзя, т.к. есть статьи с id 0
            print(texts_mapping['en2i'])
            print(target_article)
            raise NotIncludedError

        target_article_vec = corpus_vecs[target_article_id]

    else:
        target_article, target_article_vec = prepare_new_article(
            target_article_path, udpipe_path, keep_pos, keep_punct, keep_stops,
            method, embeddings_path, no_duplicates, projection_path, bidict_path)

    similars = search_similar(target_article_vec, corpus_vecs)
    rating, verbosed_rating, missed_urls = make_rating(target_article, similars, verbose, top, included,
                         texts_mapping, i2lang, with_url, article_data)

    return rating, verbosed_rating, missed_urls


if __name__ == "__main__":
    args = parse_args()

    # всё ли указали для прикрепления ссылок
    url_required = {
        1: ['url_mapping_path']
    }
    check_args(args, 'with_url', url_required)

    # Проверяем, всё ли указали для внешней статьи
    included_required = {
        0: ['udpipe_path', 'embeddings_path', 'method']
    }

    check_args(args, 'included', included_required)

    if not args.included:
        model_required = {'model': ['embeddings_path'],
                          'translation': ['embeddings_path', 'bidict_path'],
                          'projection': ['embeddings_path', 'projection_path']
                          }
        check_args(args, 'method', model_required)


    rating, verbosed_rating = main(args.target_article_path, args.lang, args.mapping_path,
                                   args.corpus_vectors_path, args.top, args.verbose, args.included,
                                   args.udpipe_path, args.keep_pos, args.keep_stops, args.keep_punct,
         args.method, args.embeddings_path, args.bidict_path, args.projection_path, args.no_duplicates,
                                   args.with_url, args.url_mapping_path)
