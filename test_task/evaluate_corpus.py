"""
python evaluate_corpus.py --lang=ru --corpus_embeddings_path=texts/ruwiki/simple.pkl --mapping_path=texts/titles_mapping.json --golden_standard=texts/ru_similar_titles.txt
python evaluate_corpus.py --lang=en --corpus_embeddings_path=texts/enwiki/simple.pkl --mapping_path=texts/titles_mapping.json --golden_standard=texts/en_similar_titles.txt
"""

import argparse
from json import load as jload
from pickle import load as pload
from tqdm import tqdm

from monolang_search import search_similar


def parse_args():
    parser = argparse.ArgumentParser(
        description='Векторизация корпуса и сохранение его в pkl в папке с текстами')
    parser.add_argument('--lang', type=str, required=True,
                        help='Заголовок статьи в формате txt (только назание, без формата), '
                             'для которой ищем ближайшие')
    parser.add_argument('--corpus_embeddings_path', type=str, required=True,
                        help='Путь к файлу pkl, в котором лежит векторизованный корпус')
    parser.add_argument('--mapping_path', type=str, required=True,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('--golden_standard', type=str, required=True,
                        help='Файл с наиболее близкими статьями')
    return parser.parse_args()


# Для каждого текста в корпусе получаем список ранжированных по близости индексов статей
def predict_sim(target_vec, corpus_vecs):  # подаём текст как вектор и векторизованный корпус
    similars = search_similar(target_vec, corpus_vecs)
    sorted_simkeys = sorted(similars, key=similars.get, reverse=True)
    return sorted_simkeys[1:]  # не 0, т.к. там он сам


def eval_acc(top, golden_standard_ids, corp_sims):
    intersections = [len({golden_standard_ids[i]} & set(corp_sims[i][:top])) for i in range(len(corp_sims))]
    acc = intersections.count(1) / len(intersections)
    return acc


def main():
    args = parse_args()

    lang2i = '{}2i'.format(args.lang)
    texts_mapping = jload(open(args.mapping_path))
    corpus_vecs = pload(open(args.corpus_embeddings_path, 'rb'))

    golden_standard_raw = (open(args.golden_standard, encoding='utf-8')).read().lower().splitlines()
    golden_standard_titles = {line.split('\t')[0]: line.split('\t')[1] for line in golden_standard_raw}
    golden_standard_ids = {texts_mapping[lang2i].get(art): texts_mapping[lang2i].get(sim_art) \
                           for art, sim_art in golden_standard_titles.items()}

    corp_sims = []  # списки предсказанного для каждого текста
    for i in tqdm(range(len(corpus_vecs))):
        target_vec = corpus_vecs[i]
        sim_ids = predict_sim(target_vec, corpus_vecs)
        corp_sims.append(sim_ids)

    acc_1 = eval_acc(1, golden_standard_ids, corp_sims)
    acc_5 = eval_acc(5, golden_standard_ids, corp_sims)
    acc_10 = eval_acc(10, golden_standard_ids, corp_sims)

    print('ТОП-1:\t{}\nТОП-5:\t{}\nТОП-10:\t{}'.format(acc_1, acc_5, acc_10))


if __name__ == "__main__":
    main()
