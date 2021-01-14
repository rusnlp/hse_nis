"""
Пайплайн для одноязычной и кросс-языковой векторизации корпусов.
Для кросс-языковой векторизации собирает общий маппинг и сохраняет в pkl общую маторицу векторов
"""

import argparse
import os
import sys
from tqdm import tqdm
from json import dump as jdump

from utils.loaders import load_vectorized, load_mapping
from utils.preprocessing import get_text, clean_ext
from utils.vectorization import build_vectorizer, vectorize_corpus, save_text_vectors
from utils.arguments import check_args


def parse_args():
    parser = argparse.ArgumentParser(
        description='Пайплайн для векторизации двух корпусов любой моделью '
                    'и составления общей матрицы и общего маппинга')
    parser.add_argument('--src_texts_path', type=str, required=True,
                        help='Путь к текстам на исходном языке в формате conllu')
    parser.add_argument('--tar_texts_path', type=str, required=True,
                        help='Путь к текстам на целевом языке в формате conllu')
    parser.add_argument('--lemmatize', type=int, required=True,
                        help='Брать ли леммы текстов (0/1)')
    parser.add_argument('--keep_pos', type=int, default=0,
                        help='Возвращать ли леммы, помеченные pos-тегами (0|1; default: 0)')
    parser.add_argument('--keep_stops', type=int, default=0,
                        help='Сохранять ли слова, получившие тег функциональной части речи '
                             '(0|1; default: 0)')
    parser.add_argument('--keep_punct', type=int, default=0,
                        help='Сохранять ли знаки препинания (0|1; default: 0)')
    parser.add_argument('--join_propn', type=int, default=0,
                        help='Склеивать ли именованные сущности (0|1; default: 0)')
    parser.add_argument('--join_token', type=str, default='::',
                        help='Как склеивать именованные сущности (default: ::)')
    parser.add_argument('--unite', type=int, default=1,
                        help='Убирать ли деление на предложения (0|1; default: 1)')
    parser.add_argument('--lang', type=str, default='cross',
                        help='Язык, для которого векторизуем; '
                             'нужен для определения словаря в маппинге (ru/en/cross, default: cross')
    parser.add_argument('--direction', type=str,
                        help='Направление перевода векторов при кросс-языковой векторизации (ru-en)')
    parser.add_argument('--method', type=str, required=True,
                        help='Метод векторизации (model/translation/projection)')
    parser.add_argument('--mapping_path', type=str, required=True,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('--src_embeddings_path', type=str,
                        help='Путь к модели векторизации для исходного языка')
    parser.add_argument('--tar_embeddings_path', type=str,
                        help='Путь к модели векторизации для целевого языка')
    parser.add_argument('--common_output_vectors_path', type=str,
                        help='Путь к pkl, в котором лежит объединённый векторизованный корпус')
    parser.add_argument('--src_output_vectors_path', type=str,
                        help='Путь к pkl, в котором лежит '
                             'уже векторизованный корпус на исходном языке')
    parser.add_argument('--tar_output_vectors_path', type=str,
                        help='Путь к pkl, в котором лежит '
                             'уже векторизованный корпус на целевом языке')
    parser.add_argument('--bidict_path', type=str,
                        help='Путь к двуязычному словарю в формате txt')
    parser.add_argument('--projection_path', type=str,
                        help='Путь к матрице трансформации в формате txt')
    parser.add_argument('--src_mis_path', type=str,
                        help='Путь к файлу с ошибками векторизации текстов на исходном языке')
    parser.add_argument('--tar_mis_path', type=str,
                        help='Путь к файлу с ошибками векторизации текстов на целевом языке')
    parser.add_argument('--no_duplicates', type=int, default=0,
                        help='Брать ли для каждого типа в тексте вектор только по одному разу '
                             '(0|1; default: 0)')

    return parser.parse_args()


def get_corpus(texts_path, lemmatize, keep_pos, keep_punct, keep_stops, join_propn, join_token, unite):
    """собираем файлы conllu в словарь {файл: список токенов}"""
    texts = {}
    for file in tqdm(os.listdir(texts_path), desc='Collecting'):
        text = open('{}/{}'.format(texts_path, file), encoding='utf-8').read().strip()
        preprocessed = get_text(text, lemmatize, keep_pos, keep_punct, keep_stops,
                                join_propn, join_token, unite)
        texts[clean_ext(file)] = preprocessed

    return texts


def main_onelang(direction, texts_path, lemmatize, keep_pos, keep_punct, keep_stops,
                 join_propn, join_token, unite,
                 embeddings_path, output_vectors_path, method, no_duplicates, projection_path,
                 bidict_path, mis_path):
    """делаем словарь векторов для корпуса"""
    # собираем тексты из conllu
    text_corpus = get_corpus(texts_path, lemmatize, keep_pos, keep_punct, keep_stops,
                             join_propn, join_token, unite)

    # для tar всегда загружаем верисю model
    vectorizer = build_vectorizer(direction, method, embeddings_path, no_duplicates,
                                  projection_path, bidict_path)

    vec_corpus, not_vectorized = vectorize_corpus(text_corpus, vectorizer)

    if output_vectors_path:
        save_text_vectors(vec_corpus, output_vectors_path)

    if not_vectorized:
        print('Не удалось векторизовать текстов: {}'.format(len(not_vectorized)), file=sys.stderr)
        open(mis_path, 'w', encoding='utf-8').write('\n'.join(not_vectorized))

    return vec_corpus


def to_common(common2i, i2common, common_vectors, vectors):
    '''добавляем корпус и заголовки в общий словарь и общий маппинг'''
    common_vectors.update(vectors)
    start_from = len(common_vectors)
    for i, title in tqdm(enumerate(vectors.keys())):
        common2i[title] = i + start_from
        i2common[i + start_from] = title

    return common_vectors, common2i, i2common


def main():
    args = parse_args()

    texts_mapping = load_mapping(args.mapping_path)

    # для кросс-языковой векторизации должно быть указано направление и путь к общей матрице векторов
    lang_required = {'cross': ['direction', 'common_output_vectors_path']}
    check_args(args, 'lang', lang_required)

    # для кроссязыковой векторизации
    if args.lang == 'cross':
        model_required = {'model': ['src_embeddings_path', 'tar_embeddings_path'],
                          'translation': ['tar_embeddings_path', 'bidict_path'],
                          'projection': ['src_embeddings_path', 'tar_embeddings_path', 'projection_path']
                          }
        check_args(args, 'method', model_required)

        if args.method == 'translation':
            args.src_embeddings_path = args.tar_embeddings_path

        directions = {d: lang for d, lang in zip(['src', 'tar'], args.direction.split('-'))}
        print(directions)

        print('Векторизую src')
        src_vectors = main_onelang('src', args.src_texts_path, args.lemmatize, args.keep_pos,
                                    args.keep_punct, args.keep_stops,
                                    args.join_propn, args.join_token, args.unite,
                                    args.src_embeddings_path, args.src_output_vectors_path, args.method,
                                    args.no_duplicates, args.projection_path, args.bidict_path, args.src_mis_path)

        print('Векторизую tar')
        tar_vectors = main_onelang('tar', args.tar_texts_path, args.lemmatize, args.keep_pos,
                                    args.keep_punct, args.keep_stops,
                                    args.join_propn, args.join_token, args.unite,
                                    args.tar_embeddings_path, args.tar_output_vectors_path, args.method,
                                    args.no_duplicates, args.projection_path, args.bidict_path, args.tar_mis_path)

        # собираем общие словарь и маппинг

        common_vectors = {}
        common2i = {}
        i2common = {}

        common_vectorized, common2i, i2common = to_common(common2i, i2common,
                                                          common_vectors, tar_vectors)

        common_vectorized, common2i, i2common = to_common(common2i, i2common,
                                                          common_vectors, src_vectors)

        save_text_vectors(common_vectors, args.common_output_vectors_path)

        texts_mapping['cross2i'] = common2i
        texts_mapping['i2cross'] = i2common
        jdump(texts_mapping, open(args.mapping_path, 'w', encoding='utf-8'))

        # print(i2common)
        # print(common2i)


if __name__ == "__main__":
    main()
