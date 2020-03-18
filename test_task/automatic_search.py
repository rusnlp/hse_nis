"""
Автоматический запуск поиска для списка статей
"""

import argparse
import os
import subprocess
from tqdm import tqdm


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
    parser.add_argument('--top', type=int,
                        help='Сколько близких статeй возвращать (default: 1; -1 for all)')
    return parser.parse_args()


def create_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def main():
    args = parse_args()

    create_dir(args.result_path[:args.result_path.rfind('/')])

    titles = [line.split()[0] for line in open(args.titles_path, encoding='utf-8').readlines()]

    results = []
    for title in tqdm(titles):
        command = '''python3 monocorp_search.py --target_article_path={} --lang={} \
        --mapping_path={} --corpus_vectors_path={}'''.\
            format(title, args.lang, args.mapping_path, args.corpus_vectors_path)
        if args.top:
            command += ' --top={}'.format(args.top)

        output = subprocess.getoutput(command)
        results += [output]

    formated_results = ['{}. {}'.format(i+1, result[1:]) for i, result in enumerate(results)]
    open(args.result_path, 'w', encoding='utf-8').write('\n\n'.join(formated_results))


if __name__ == '__main__':
    main()
