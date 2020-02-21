import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from json import load as jload, dump as jdump
from pickle import dump as pdump, load as pload

from vectorization import load_embeddings, vectorize_corpus

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


def main_onelang(lang, texts_mapping, lemmatized_path, embeddings_path, output_embeddings_path):
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
            lemmatized_corpus = get_lemmatized_corpus(texts_mapping, i2lang, lemmatized_dict, n_new_texts)
            for i in lemmatized_corpus:
                print(i)

            emb_model = load_embeddings(embeddings_path)
            emb_dim = emb_model.vector_size
            emb_model = emb_model.wv

            # за размер нового корпуса принимаем длину маппинга
            new_vectorized = np.zeros((len(texts_mapping[i2lang]), emb_dim))

            # заполняем старые строчки, если они были
            for nr, line in enumerate(old_vectorized):
                new_vectorized[nr, :] = line
            print(new_vectorized)
            print(new_vectorized.shape)

            new_vectorized, not_vectorized = vectorize_corpus(
                lemmatized_corpus, new_vectorized, emb_model, no_duplicates,
                starts_from=len(old_vectorized))
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


if __name__ == "__main__":
    ru_model_path = 'models/ru.bin'
    en_model_path = 'models/en.bin'
    bidict_path = 'words/ru-en_lem.txt'
    proj_path = 'words/ru-en_proj.txt'
    mapping_path = 'texts/titles_mapping.json'
    dir = 'ru-en'.split('-')
    ru_lemmatized_path = 'texts/new_ru_lemmatized.json'
    en_lemmatized_path = 'texts/en_lemmatized.json'
    forced = False
    ru_output_embeddings_path = 'texts/ru_simple.pkl'
    en_output_embeddings_path = 'texts/en_simple.pkl'
    common_output_embeddings_path = 'texts/common_simple.pkl'
    no_duplicates = False
    type = 'simple'


    texts_mapping = jload(open(mapping_path))

    ru_vectorized = main_onelang(dir[0], texts_mapping, ru_lemmatized_path, ru_model_path, ru_output_embeddings_path)
    #print(ru_vectorized)
    en_vectorized = main_onelang(dir[1], texts_mapping, en_lemmatized_path, en_model_path, en_output_embeddings_path)
    #print(en_vectorized)

    # собираем общие матрицу и маппинг
    common_len = len(en_vectorized) + len(ru_vectorized)
    emb_dim = en_vectorized.shape[1]
    common_vectorized = np.zeros((common_len, emb_dim))
    print(common_vectorized.shape)

    common2i = {}
    i2common = {}

    # соберём в обратном порядке: сначала английские, потом русские
    common_vectorized, common2i, i2common = to_common(texts_mapping, common2i, i2common, common_vectorized, ru_vectorized, dir[0], start_from=0)
    common_vectorized, common2i, i2common = to_common(texts_mapping, common2i, i2common, common_vectorized, en_vectorized, dir[1], start_from=len(ru_vectorized))

    pdump(common_vectorized, open(common_output_embeddings_path, 'wb'))

    texts_mapping['common2i'] = common2i
    texts_mapping['i2common'] = i2common
    jdump(texts_mapping, open(mapping_path, 'w', encoding='utf-8'))


    print(common_vectorized[-1].tolist())
    print(en_vectorized[-1].tolist())


    print(i2common)
    print(common2i)