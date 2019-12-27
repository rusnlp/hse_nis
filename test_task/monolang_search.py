"""
Считаем, что все тексты, которые ищем, априори добавлены в корпус, предобработаны,
вектора для всего построены

Подаём название текста, язык, путь к папке с текстами, путь к маппингу, тип модели, которой векторизовали,
можно подать кол-во ближайших статей
Получаем n ближайших записей в виде списка кортежей (заголовок, близость) -- напечатем рейтинг,
если не сделали verbose=False

python "mono-lang_search.py" кровь ru texts/ruwiki texts/titles_mapping.json simple --included=True
python "mono-lang_search.py" blood en texts/enwiki texts/titles_mapping.json simple --included=True
python "mono-lang_search.py" blood en texts/enwiki texts/titles_mapping.json simple --top=10
"""

import argparse
from json import load as jload
from pickle import load as pload

import numpy as np
from numpy.linalg import norm


# Для каждого текста в корпусе считаем косинусную близость к данному
def search_similar(target_vec, corpus_vecs):  # подаём текст как вектор и векторизованный корпус
    target_vec = target_vec/norm(target_vec) # нормализуем вектор текста
    sim_vecs = np.dot(corpus_vecs, target_vec)
    # словарь {индекс текста в корпусе: близость к данному}
    similars = {i:sim for i, sim in enumerate(sim_vecs)}

    # print(similars)
    return similars


# Ранжируем тексты по близости и принтим красивый списочек
def make_rating(similars, verbose, n, included):  # принимаем словарь близостей
    # сортируем словарь по значениям в порядке убывания: сортируем список кортежей (key, value) по value
    sorted_simkeys = sorted(similars, key=similars.get, reverse=True)
    # similars: [i, j, ...]
    similars_list = [(texts_mapping[i2lang].get(str(simkey)), similars[simkey]) for simkey in sorted_simkeys]
    # [(i_title, sim), (j_title, sim), (...)]
    if included:  # если статья включена в корпус
        # на 0 индексе всегда будет сама статья, если она из корпуса
        similars_list = similars_list[1:]
    if verbose:
        print('\nТоп-{} ближайших статей к {}:'.format(n, target_article))
        for i, sim_item in enumerate(similars_list[:n]):
            print('{}. {} ({})'.format(i + 1, sim_item[0], sim_item[1]))
    return similars_list[:n]  # если нужна будет одна статья, вернётся список с одним элементом


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Векторизация корпуса и сохранение его в pkl в папке с текстами')
    parser.add_argument('target_article_path', type=str,
                        help='Путь к статье в формате txt, для которой ищем ближайшие.'
                             '\nЕсли статья из корпуса, то только назание без формата')
    parser.add_argument('lang', type=str,
                        help='Язык, для которго разбираем, нужен для определения словаря в маппинге (ru/en)')
    parser.add_argument('texts_path', type=str,
                        help='Папка, в которой лежат тексты в формате txt')
    parser.add_argument('mapping_path', type=str,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('model_type', type=str,
                        help='Краткое имя модели векторизации, чтобы не путать файлы. '
                             'Будет использовано как имя pkl')
    parser.add_argument('--included', type=bool, default=False,
                        help='Включена ли статья в корпус (default: False)')
    parser.add_argument('--top', type=int, default=1,
                        help='Сколько близких статeй возвращать (default: 1)')
    # TODO: сделать вариант для возвращения всех
    parser.add_argument('--verbose', type=bool, default=True,
                        help='Принтить ли рейтинг (default: True)')
    args = parser.parse_args()

    i2lang = 'i2{}'.format(args.lang)
    lang2i = '{}2i'.format(args.lang)
    vecs_path = '{}/{}.pkl'.format(args.texts_path, args.model_type)
    texts_mapping = jload(open(args.mapping_path))
    corpus_vecs = pload(open(vecs_path, 'rb'))

    if args.included:
        target_article_title = args.target_article_path.lower()  # на всякий приведём к нижнему регистру
        target_article = '{}.txt'.format(target_article_title)
        target_article_id = texts_mapping[lang2i].get(target_article)
        # print(target_article, target_article_id)
        target_article_vec = corpus_vecs[target_article_id]
    else:
        pass

    similars = search_similar(target_article_vec, corpus_vecs)
    # print(similars)
    make_rating(similars, n=args.top, verbose=args.verbose, included=args.included)
