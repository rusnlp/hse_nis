"""
python vectorization_pipeline.py --src_lemmatized_path=texts/ru_lemmatized.json --tar_lemmatized_path=texts/en_lemmatized.json --direction=ru-en --method=model --mapping_path=texts/titles_mapping.json --src_embeddings_path=models/ru_vecmap.vec --tar_embeddings_path=models/en_vecmap.vec --src_output_embeddings_path=texts/ru_vecmap.pkl --tar_output_embeddings_path=texts/en_vecmap.pkl --common_output_embeddings_path=texts/common_vecmap.pkl --forced=1
python vectorization_pipeline.py --src_lemmatized_path=texts/ru_lemmatized.json --tar_lemmatized_path=texts/en_lemmatized.json --direction=ru-en --method=projection --mapping_path=texts/titles_mapping.json --src_embeddings_path=models/ru.bin --tar_embeddings_path=models/en.bin --src_output_embeddings_path=texts/ru_projection.pkl --tar_output_embeddings_path=texts/en_projection.pkl --common_output_embeddings_path=texts/common_projection.pkl --projection_path=words/ru-en_proj.txt --forced=1
python vectorization_pipeline.py --src_lemmatized_path=texts/ru_lemmatized.json --tar_lemmatized_path=texts/en_lemmatized.json --direction=ru-en --method=translation --mapping_path=texts/titles_mapping.json --src_embeddings_path=models/ru.bin --tar_embeddings_path=models/en.bin --src_output_embeddings_path=texts/ru_trans.pkl --tar_output_embeddings_path=texts/en_trans.pkl --common_output_embeddings_path=texts/common_trans.pkl --bidict_path=words/ru-en_lem.txt --forced=1
"""

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from json import load as jload, dump as jdump
from pickle import dump as pdump, load as pload

from vectorization import build_vectorizer, vectorize_corpus


def parse_args():
    parser = argparse.ArgumentParser(
        description='Пайплайн для векторизации двух корпусов любой моделью и составления общей матрицы и общего маппинга')
    parser.add_argument('--src_lemmatized_path', type=str, required=True,
                        help='Путь к лемматизованным текстам на исходном языке в формате json')
    parser.add_argument('--tar_lemmatized_path', type=str, required=True,
                        help='Путь к лемматизованным текстам на целевом языке в формате json')
    parser.add_argument('--direction', type=str, required=True,
                        help='Направление перевода векторов (ru-en)')
    parser.add_argument('--method', type=str, required=True,
                        help='Метод векторизации (model/translation/projection)')
    parser.add_argument('--mapping_path', type=str, required=True,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('--src_embeddings_path', type=str, required=True,
                        help='Путь к модели векторизации для исходного языка')
    parser.add_argument('--tar_embeddings_path', type=str, required=True,
                        help='Путь к модели векторизации для целевого языка')
    parser.add_argument('--src_output_embeddings_path', type=str, required=True,
                        help='Путь к pkl, в котором лежит уже векторизованный корпус на исходном языке')
    parser.add_argument('--tar_output_embeddings_path', type=str, required=True,
                        help='Путь к pkl, в котором лежит уже векторизованный корпус на целевом языке')
    parser.add_argument('--common_output_embeddings_path', type=str, required=True,
                        help='Путь к pkl, в котором лежит объединённый векторизованный корпус')
    parser.add_argument('--bidict_path', type=str, default='',
                        help='Путь к двуязычному словарю в формате txt')
    parser.add_argument('--projection_path', type=str, default='',
                        help='Путь к матрице трансформации в формате txt')
    parser.add_argument('--no_duplicates', type=int, default=0,
                        help='Брать ли для каждого типа в тексте вектор только по одному разу '
                             '(0|1; default: 0)')
    parser.add_argument('--forced', type=int, default=0,
                        help='Принудительно векторизовать весь корпус заново (0|1; default: 0)')

    return parser.parse_args()


class NotLemmatizedError(Exception):
    """
    По указанному пути не нашёлся json с лемматизированными тестами
    """
    def __init__(self):
        self.text = 'Нечего векторизовать! Пожалуйста, сначала лемматизируйте тексты'

    def __str__(self):
        return self.text


def get_lemmatized_corpus(mapping, i2lang_name, lemmatized, n_new):
    """собираем корпус лемматизированных текстов и [], если в маппинге есть, а в lemmatized нет"""
    corpus = []
    # для каждого номера в маппинге от последнего в vectorized
    for nr in range(len(mapping[i2lang_name]) - n_new, len(mapping[i2lang_name])):
        # порядок текстов -- как в индексах
        title = mapping[i2lang_name].get(str(nr))
        # по номеру из маппинга берём название и находим его в леммах, если нет -- []
        lemmatized_text = lemmatized.get(title, [])
        corpus.append(lemmatized_text)
    return corpus


def load_vectorized(output_embeddings_path, forced):
    """загрузка матрицы с векторами корпуса, если есть"""
    # если существует уже какой-то векторизованный корпус
    if os.path.isfile(output_embeddings_path) and not forced:
        vectorized = pload(open(output_embeddings_path, 'rb'))
        print('Уже что-то векторизовали!', file=sys.stderr)

    else:  # ничего ещё из этого корпуса не векторизовали или принудительно обновляем всё
        print('Ничего ещё не разбирали, сейчас будем.', file=sys.stderr)
        vectorized = []

    return vectorized


def main_onelang(direction, lang, texts_mapping, lemmatized_path, embeddings_path,
                 output_embeddings_path, method, no_duplicates, projection_path, bidict_path, forced):
    i2lang = 'i2{}'.format(lang)

    # собираем лемматизированные тексты из lemmatized
    if not os.path.isfile(lemmatized_path):  # ничего ещё из этого корпуса не разбирали
        raise NotLemmatizedError()

    else:  # если существует уже разбор каких-то файлов
        lemmatized_dict = jload(open(lemmatized_path, encoding='utf-8'))
        print('Понял, сейчас векторизуем.', file=sys.stderr)

        # подгружаем старое, если было
        old_vectorized = load_vectorized(output_embeddings_path, forced)

        # появились ли новые номера в маппинге
        n_new_texts = len(texts_mapping[i2lang]) - len(old_vectorized)
        print('Новых текстов: {}'.format(n_new_texts), file=sys.stderr)

        if n_new_texts:
            # собираем всё, что есть лемматизированного и нелемматизированного
            lemmatized_corpus = get_lemmatized_corpus(texts_mapping, i2lang, lemmatized_dict,
                                                      n_new_texts)
            # for i in lemmatized_corpus:
            #     print(i)

            # для tar всегда загружаем верисю model
            vectorizer = build_vectorizer(direction, method, embeddings_path, no_duplicates,
                                          projection_path, bidict_path)

            # за размер нового корпуса принимаем длину маппинга
            new_vectorized = np.zeros((len(texts_mapping[i2lang]), vectorizer.dim))

            # заполняем старые строчки, если они были
            for nr, line in enumerate(old_vectorized):
                new_vectorized[nr, :] = line
            # print(new_vectorized)
            # print(new_vectorized.shape)

            new_vectorized, not_vectorized = vectorize_corpus(
                lemmatized_corpus, new_vectorized, vectorizer, starts_from=len(old_vectorized))
            pdump(new_vectorized, open(output_embeddings_path, 'wb'))

            if not_vectorized:
                print('Не удалось векторизовать следующие тексты:\n{}'.
                      format('\n'.join(not_vectorized)), file=sys.stderr)

            return new_vectorized

        else:  # если не нашлось новых текстов
            return old_vectorized


def to_common(texts_mapping, common2i, i2common, common_vectorized, vectorized, lang, start_from=0):
    '''добавляем корпус и заголовки в общую матрицу и общий маппинг'''
    for nr in tqdm(range(len(vectorized))):
        common_vectorized[nr + start_from, :] = vectorized[nr]

        title = texts_mapping['i2{}'.format(lang)][str(nr)]
        common2i[title] = nr + start_from
        i2common[nr + start_from] = title

    return common_vectorized, common2i, i2common


def main():
    args = parse_args()

    directions = {d: lang for d, lang in zip(['src', 'tar'], args.direction.split('-'))}
    print(directions)

    texts_mapping = jload(open(args.mapping_path))

    src_vectorized = main_onelang('src', directions['src'], texts_mapping, args.src_lemmatized_path,
                                  args.src_embeddings_path, args.src_output_embeddings_path, args.method,
                                  args.no_duplicates, args.projection_path, args.bidict_path, args.forced)
    # print(src_vectorized)
    tar_vectorized = main_onelang('tar', directions['tar'], texts_mapping, args.tar_lemmatized_path,
                                  args.tar_embeddings_path, args.tar_output_embeddings_path, args.method,
                                  args.no_duplicates, args.projection_path, args.bidict_path, args.forced)
    # print(tar_vectorized)

    # собираем общие матрицу и маппинг
    common_len = len(src_vectorized) + len(tar_vectorized)
    emb_dim = src_vectorized.shape[1]
    common_vectorized = np.zeros((common_len, emb_dim))
    print(common_vectorized.shape)

    common2i = {}
    i2common = {}

    common_vectorized, common2i, i2common = to_common(texts_mapping, common2i, i2common,
                                                      common_vectorized, tar_vectorized,
                                                      directions['tar'], start_from=0)
    common_vectorized, common2i, i2common = to_common(texts_mapping, common2i, i2common,
                                                      common_vectorized, src_vectorized,
                                                      directions['src'],
                                                      start_from=len(tar_vectorized))

    pdump(common_vectorized, open(args.common_output_embeddings_path, 'wb'))

    texts_mapping['common2i'] = common2i
    texts_mapping['i2common'] = i2common
    jdump(texts_mapping, open(args.mapping_path, 'w', encoding='utf-8'))

    print(i2common)
    print(common2i)


if __name__ == "__main__":
    main()
