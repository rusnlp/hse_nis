"""
Считаем, что все тексты, которые ищем, априори добавлены в корпус, предобработаны,
вектора для всего построены

Подаём название текста, язык, путь к векторизованному корпусу, путь к маппингу,
можно подать кол-во ближайших статей
Получаем n ближайших записей в виде списка кортежей (заголовок, близость)
Напечатем рейтинг, если не поставили verbose=False

python monocorp_search.py --target_article_path=aist_2012_c6bc0383ea448fcb7e5f45ac85a1afb2d12505ef --mapping_path=../texts_conf/mapping.json --corpus_vectors_path=../models/common_lem_muse_orig.bin.gz --top=10 --text_sim_treshold=0.5 --with_url=1 --url_mapping_path=../texts_conf/hash_title_url.tsv
python monocorp_search.py --target_article_path=aist_2012_c6bc0383ea448fcb7e5f45ac85a1afb2d12505ef --mapping_path=../texts_conf/mapping.json --corpus_vectors_path=../models/common_tok_muse_orig.bin.gz --top=10 --text_sim_treshold=0.5 --with_url=1 --url_mapping_path=../texts_conf/hash_title_url.tsv
"""

import argparse

from utils.arguments import check_args
from utils.loaders import load_embeddings, load_mapping, load_article_data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Ранжирование статей на основе косинусной близости векторов')
    parser.add_argument('--target_article_path', type=str, required=True,
                        help='Путь к статье, для которой ищем ближайшие.'
                             '\nЕсли статья из корпуса, то только назание')
    parser.add_argument('--lang', type=str, default='cross',
                        help='Язык, для которого разбираем; '
                             'нужен для определения словаря в маппинге (ru|en|cross; default: 1')
    parser.add_argument('--mapping_path', type=str, required=True,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('--corpus_vectors_path', type=str, required=True,
                        help='Путь к файлу pkl, в котором лежит векторизованный корпус')
    parser.add_argument('--top', type=int, default=1,
                        help='Сколько близких статeй возвращать (default: 1; -1 for all)')
    parser.add_argument('--text_sim_treshold', type=float, default=0,
                        help='Порог близости для статей (default: 0)')
    parser.add_argument('--task_sim_treshold', type=float, default=0.7,
                        help='Порог близости для задач (default: 0.7)')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Принтить ли рейтинг (0|1; default: 0)')
    parser.add_argument('--with_url', type=int, default=0,
                        help='Добавлять ссылки к заголовкам (0|1; default: 0)')
    parser.add_argument('--url_mapping_path', type=str,
                        help='Путь к файлу маппинга заголовков в ссылки')
    # ДЛЯ ВНЕШНИХ ТЕКСТОВ:
    parser.add_argument('--included', type=int, default=1,
                        help='Включена ли статья в корпус (0|1; default: 1)')
    return parser.parse_args()


class NotIncludedError(Exception):
    """
    Указали included=True для текста, которого нет в маппинге
    """
    def __init__(self):
        self.text = 'Такого текста нет в корпусе! Пожалуйста, измените значение параметра included'

    def __str__(self):
        return self.text


class NoSimilarError(Exception):
    """
    Не нашлось близких статей
    """
    def __init__(self):
        self.text = 'В корпусе не нашлось близких статей. ' \
                    'Пожалуйста, попробуйте снизить порог text_sim_treshold'

    def __str__(self):
        return self.text


class SimPaper:
    def __init__(self, title, sim, n):
        self.title = title
        self.sim = sim
        self.n = n
        self.lang = ''
        self.real_title = ''
        self.url = ''
        self.tasks = {}

    def __str__(self):
        return '{}\t{}\t{}\t{}'\
            .format(self.title, self.lang, self.real_title, self.url)

    def str_as_sim(self):
        return '{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(self.n, self.title, self.lang, self.sim, self.real_title, self.url,
                                               '|'.join(['{}({})'.format(task, sim) for task, sim in self.tasks.items()]))

    def str_as_tar(self):
        return '\t{}\t{}\t{}\t{}\t{}'.format(self.title, self.lang, self.real_title, self.url,
                                         '|'.join(['{}({})'.format(task, sim) for task, sim in self.tasks.items()]))


def search_similar(vector, model, top, sim_treshold, included):
    """
    :param vector: вектор текста (np.array)
    :param model: модель векторов корпуса
    :return: индексы текстов и близость к данному (словарь)
    """
    if included:  # первый результат будет лишний
        top += 1
    # sim_list = model.similar_by_vector(vector, top)
    sim_list = [(d, sim) for d, sim in model.similar_by_vector(vector, len(model.vocab))
                if not d.startswith('TASK::') and sim >= sim_treshold][:top]
    sim_dict = {title: SimPaper(title, sim, i) for i, (title, sim) in enumerate(sim_list)}

    return sim_dict


def get_tasks(title, model, sim_treshold):
    task_dict = {d: sim for d, sim in model.wv.most_similar(title, topn=len(model.vocab))
                if d.startswith('TASK::') and sim >= sim_treshold}
    return task_dict


def get_lang(title, mapping):
    '''ищем заголовок в словарях маппига и извлекаем язык '''
    lang2i_dicts = [key for key in mapping.keys() if key.endswith('2i') and 'cross' not in key]
    for lang_dict in lang2i_dicts:  # если найдётся в каком-нибудь из словарей lang2i
        if title in mapping[lang_dict]:
            lang = lang_dict.split('2')[0]  # извлекаем язык из названия словаря
            break  # будем надеться, что статей с одинаковым названием нет (хеши-то разные)
    return lang


def verbose_rating(tarpaper, simpapers):
    verbosed = tarpaper.str_as_tar()
    for simpaper in simpapers:
        verbosed += '\n'+simpaper.str_as_sim()
        # print(verbosed)

    return verbosed


def make_rating(target_article, sim_dict, task_sim_treshold, verbose, included, mapping, model,
                with_url=0, article_data=None):
    """
    :param target_article: заголовок статьи, для которой ищем ближайшие (строка)
    :param sim_dict: индексы статей в корпусе и близость к данной (словарь)
    :param verbose: принтить ли рейтинг (pseudo-boolean int)
    :param included: включена ли статья в корпус (pseudo-boolean int)
    :param mapping: маппинг заголовков в индексы и обратно (словарь словарей)
    :return: индексы текстов и близость к данному в порядке убывания (список кортежей)
    """

    simpapers = []  # список объектов SimPaper
    missed_urls = []  # список статей, пропущенных в hash_title_url

    for title, simpaper in sim_dict.items():
        simpaper.lang = get_lang(title, mapping)
        simpaper.tasks = get_tasks(simpaper.title, model, task_sim_treshold)
        if with_url:
            try:
                simpaper.url = article_data[title].get('url')
                simpaper.real_title = article_data[title].get('real_title')
            except KeyError:
                missed_urls.append(str(simpaper))
        simpapers.append(simpaper)

    if included:  # на 0 индексе всегда будет сама статья, если она из корпуса
        if len(simpapers) >= 2:
            tarpaper = simpapers[0]
            simpapers = simpapers[1:]
        else:
            raise NoSimilarError
    else:
        if len(simpapers) >= 1:
            tarpaper = SimPaper(target_article, '', '')
            tarpaper.lang = 'tarlang'  # TODO: просить вводить?
        else:
            raise NoSimilarError

    # вывод всё равно придётся делать для получения в automatic_search
    verbosed_rating = verbose_rating(tarpaper, simpapers)
    if verbose:
        print(verbosed_rating)

    # если нужна будет одна статья, вернётся список с одним элементом
    return simpapers, verbosed_rating, missed_urls


def main_search(target_article_path, lang2i_name, texts_mapping, corpus_model,
         top=1, text_sim_treshold=0, task_sim_treshold=0.7, verbose=1, included=1,
                with_url=0, url_mapping_path=''):

    article_data = None
    if with_url:
        article_data = load_article_data(url_mapping_path)

    if included:
        target_article = target_article_path

        if target_article not in texts_mapping[lang2i_name]:
            print(lang2i_name, target_article)
            raise NotIncludedError

        target_article_vec = corpus_model.get_vector(target_article)

    else:
        pass

    if top == -1:  # нужен вывод всех статей
        top = len(corpus_model.vocab)

    similars = search_similar(target_article_vec, corpus_model, top, text_sim_treshold, included)

    rating, verbosed_rating, missed_urls = make_rating(target_article, similars, task_sim_treshold,
                                                       verbose, included, texts_mapping, corpus_model,
                                                       with_url, article_data)

    return rating, verbosed_rating, missed_urls


def main():
    args = parse_args()

    # всё ли указали для прикрепления ссылок
    url_required = {
        1: ['url_mapping_path']
    }
    check_args(args, 'with_url', url_required)

    lang2i_name = '{}2i'.format(args.lang)
    texts_mapping = load_mapping(args.mapping_path)

    corpus_model = load_embeddings(args.corpus_vectors_path)

    rating, verbosed_rating, missed_urls = main_search(args.target_article_path, lang2i_name,
                                                       texts_mapping, corpus_model, args.top,
                                                       args.text_sim_treshold, args.task_sim_treshold,
                                                       args.verbose, args.included,
                                                       args.with_url, args.url_mapping_path)
    # print(rating)


if __name__ == "__main__":
    main()
