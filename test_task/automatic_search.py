"""
Автоматический запуск поиска для списка статей
"""

import argparse
import os
from tqdm import tqdm

import monocorp_search

def parse_args():
    parser = argparse.ArgumentParser(
        description='Поочерёдный поиск ближайших статей к заданным и сохранение результатов')
    parser.add_argument('--titles_path', type=str, required=True,
                        help='Список заголовков, для которых запускаем поиск')
    parser.add_argument('--mapping_path', type=str, required=True,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('--corpus_vectors_path', type=str, required=True,
                        help='Путь к файлу pkl, в котором лежит векторизованный корпус')
    parser.add_argument('--result_path', type=str, required=True,
                        help='Файл, куда сохранятся результаты поиска')
    parser.add_argument('--lang', type=str, default='cross',
                        help='Язык, для которого разбираем; нужен для определения словаря в маппинге')
    parser.add_argument('--top', type=int, default=1,
                        help='Сколько близких статeй возвращать (default: 1; -1 for all)')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Принтить ли рейтинг (0|1; default: 0)')
    parser.add_argument('--with_url', type=int, default=0,
                        help='Добавлять ссылки к заголовкам (0|1; default: 0)')
    parser.add_argument('--url_mapping_path', type=str,
                        help='Путь к файлу маппинга заголовков в ссылки')
    return parser.parse_args()



def create_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def main():
    args = parse_args()

    create_dir(args.result_path[:args.result_path.rfind('/')])

    titles = [line.split('\t')[0] for line in open(args.titles_path, encoding='utf-8').readlines()]

    results = []
    missed_urls_all = []
    for title in tqdm(titles):
        rating, verbosed_rating, missed_urls = monocorp_search.main(title, args.lang, args.mapping_path,
                                            args.corpus_vectors_path, args.top,
                                            verbose=args.verbose, with_url=args.with_url,
                                            url_mapping_path=args.url_mapping_path)
        results += [verbosed_rating]
        missed_urls_all += missed_urls

    formated_results = ['{}. {}'.format(i+1, result) for i, result in enumerate(results)]
    open(args.result_path, 'w', encoding='utf-8').write('\n\n'.join(formated_results))

    missed_urls_all = set(missed_urls_all)
    print('Нет в hash_title_urls {}: {}'.format(len(missed_urls_all), missed_urls_all))


if __name__ == '__main__':
    main()
