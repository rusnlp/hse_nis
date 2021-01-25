"""
python vectorization_pipeline.py --texts_paths=../texts_conf/texts/en_conllu+../texts_conf/texts/ru_conllu --lemmatize=1 --embeddings_path=../models/cross_muse_orig.bin.gz --common_vectors_path=../models/common_lem_muse_orig.bin.gz --mapping_path=../texts_conf/mapping.json --mis_path=../texts_conf/texts/mis_lem_en_orig.txt+../texts_conf/texts/mis_lem_ru_orig.txt --task_path=../words/nlpub.tsv --task_column=terms_short --task_mis_path=../words/mis_lem_nlpub_orig.txt
python vectorization_pipeline.py --texts_paths=../texts_conf/texts/en_conllu+../texts_conf/texts/ru_conllu --lemmatize=0 --embeddings_path=../models/cross_muse_orig.bin.gz --common_vectors_path=../models/common_tok_muse_orig.bin.gz --mapping_path=../texts_conf/mapping.json --mis_path=../texts_conf/texts/mis_tok_en_orig.txt+../texts_conf/texts/mis_tok_ru_orig.txt --task_path=../words/nlpub.tsv --task_column=terms_short --task_mis_path=../words/mis_tok_nlpub_orig.txt
"""

import argparse
import logging
from json import dump as jdump

from utils.loaders import load_embeddings, save_w2v, load_task_terms, load_mapping, split_paths
from utils.preprocessing import get_corpus
from utils.vectorization import vectorize_corpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Векторизация корпусов в общую модель')
    parser.add_argument('--texts_paths', type=str, required=True,
                        help='Путь к текстам в формате conllu (можно перечислить через +)')
    parser.add_argument('--lemmatize', type=int, required=True,
                        help='Брать ли леммы текстов (0|1)')
    parser.add_argument('--keep_pos', type=int, default=0,
                        help='Возвращать ли слова, помеченные pos-тегами (0|1; default: 0)')
    parser.add_argument('--keep_punct', type=int, default=0,
                        help='Сохранять ли знаки препинания (0|1; default: 0)')
    parser.add_argument('--keep_stops', type=int, default=0,
                        help='Сохранять ли слова, получившие тег функциональной части речи '
                             '(0|1; default: 0)')
    parser.add_argument('--join_propn', type=int, default=0,
                        help='Склеивать ли именованные сущности (0|1; default: 0)')
    parser.add_argument('--join_token', type=str, default='::',
                        help='Как склеивать именованные сущности (default: ::)')
    parser.add_argument('--unite', type=int, default=1,
                        help='Убирать ли деление на предложения (0|1; default: 1)')
    parser.add_argument('--no_duplicates', type=int, default=0,
                        help='Брать ли для каждого типа в тексте вектор только по одному разу '
                             '(0|1; default: 0)')
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Путь к модели векторизации')
    parser.add_argument('--common_vectors_path', type=str, required=True,
                        help='Путь к файлу, в котором лежит объединённый векторизованный корпус')
    parser.add_argument('--dir_vectors_paths', type=str,
                        help='Пути к файлам с векторизованными корпусами из одной директории'
                             ' (можно перечислить через +)')
    parser.add_argument('--mapping_path', type=str, required=True,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('--mis_path', type=str,
                        help='Путь к файлу с ошибками векторизации текстов '
                             '(можно перечислить через +)')
    parser.add_argument('--task_path', type=str, required=False,
                        help='Путь к файлу с темами из NLPub')
    parser.add_argument('--task_column', type=str, default='terms',
                        help='Какую колонку со словами брать из NLPub (default: terms)')
    parser.add_argument('--task_mis_path', type=str,
                        help='Путь к файлу с ошибками векторизации задач NLPub')

    return parser.parse_args()


def main_onepath(texts_path, lemmatize, keep_pos, keep_punct, keep_stops, join_propn, join_token,
                 unite, embed_model, no_duplicates, dir_vectors_path, mis_path):
    """делаем словарь векторов для папки"""
    # собираем тексты из conllu
    text_corpus = get_corpus(texts_path, lemmatize, keep_pos, keep_punct, keep_stops,
                             join_propn, join_token, unite)

    vec_corpus, not_vectorized = vectorize_corpus(text_corpus, embed_model, no_duplicates)

    if dir_vectors_path:
        save_w2v(vec_corpus, dir_vectors_path)

    if not_vectorized:
        logging.info('Not vectorized texts: {}'.format(len(not_vectorized)))
        if mis_path:
            open(mis_path, 'w', encoding='utf-8').write('\n'.join(not_vectorized))

    return vec_corpus


def to_common(common2i, i2common, common_vectors, vectors):
    """добавляем корпус и заголовки в общий словарь и общий маппинг"""
    start_from = len(common_vectors)
    for i, title in enumerate(vectors.keys()):
        common2i[title] = i + start_from
        i2common[i + start_from] = title
    common_vectors.update(vectors)
    return common_vectors, common2i, i2common


def main():
    args = parse_args()

    texts_paths = args.texts_paths.split('+')
    dir_vectors_paths = split_paths(args.dir_vectors_paths, texts_paths)
    mis_paths = split_paths(args.mis_path, texts_paths)

    embed_model = load_embeddings(args.embeddings_path)
    common_vectors = {}
    common2i = {}
    i2common = {}

    for texts_path, dir_vectors_path, mis_path in zip(texts_paths, dir_vectors_paths, mis_paths):
        logging.info('Vectorizing {}...'.format(texts_path))
        text_vectors = main_onepath(texts_path, args.lemmatize, args.keep_pos, args.keep_punct,
                                    args.keep_stops, args.join_propn, args.join_token, args.unite, embed_model, args.no_duplicates,
                                    dir_vectors_path, mis_path)

        common_vectors, common2i, i2common = to_common(common2i, i2common, common_vectors, text_vectors)

    if args.task_path:
        task_terms = load_task_terms(args.task_path, args.task_column)
        task_vectors, not_vectorized = vectorize_corpus(task_terms, embed_model)
        if not_vectorized:
            logging.info('Not vectorized tasks: {}'.format(len(not_vectorized)))
            if args.task_mis_path:
                open(args.task_mis_path, 'w', encoding='utf-8').write('\n'.join(not_vectorized))
        common_vectors, common2i, i2common = to_common(common2i, i2common, common_vectors, task_vectors)

    save_w2v(common_vectors, args.common_vectors_path)

    texts_mapping = load_mapping(args.mapping_path)
    texts_mapping['cross2i'] = common2i
    texts_mapping['i2cross'] = i2common
    jdump(texts_mapping, open(args.mapping_path, 'w', encoding='utf-8'))


if __name__ == "__main__":
    main()
