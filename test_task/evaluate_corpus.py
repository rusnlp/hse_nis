"""
python evaluate_corpus.py ru texts/ruwiki texts/titles_mapping.json texts/ru_similar_titles.txt simple
"""

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from json import load as jload
from pickle import load as pload
import argparse


# Для каждого текста в корпусе получаем индекс ближайшего к данному
# TODO: мы весь корпус считаем дважды, надо в табличку
def search_sim(target_vec, corpus_vecs):  # подаём текст как вектор и векторизованный корпус
    similars = {}  # словарь {индекс текста в корпусе: близость к данному}
    for i, v in enumerate(corpus_vecs):
        # вычисляем косинусную близость для данного вектора и вектора текста
        sim = cosine_similarity(target_vec.reshape(1, target_vec.shape[0]), v.reshape(1, v.shape[0]))
        # print(sim)
        similars[i] = sim[0][0]  # для индекса текста добавили его близость к данному в словарь
    # сортируем словарь по значениям в порядке убывания: сортируем список кортежей (key, value) по value
    sorted_simkeys = sorted(similars, key=similars.get, reverse=True)
    return sorted_simkeys[1:]  # не 0, т.к. там он сам


def check(i, top, if_top):
    if golden_standard_ids[i] in top[i]:
        if_top.append(1)
    else:
        if_top.append(0)


def eval_acc():
    # TODO: или три генератора были бы быстрее?
    if_1, if_5, if_10 = [], [], []
    for i in range(len(corpus_vecs)):
        check(i, corp_sims_1, if_1)
        check(i, corp_sims_5, if_5)
        check(i, corp_sims_10, if_10)
    acc_1 = if_1.count(1) / len(if_1)
    acc_5 = if_5.count(1) / len(if_5)
    acc_10 = if_10.count(1) / len(if_10)
    return acc_1, acc_5, acc_10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Векторизация корпуса и сохранение его в pkl в папке с текстами')
    parser.add_argument('lang', type=str,
                        help='Заголовок статьи в формате txt (только назание, без формата), '
                             'для которой ищем ближайшие')
    parser.add_argument('texts_path', type=str,
                        help='Папка, в которой лежат тексты в формате txt')
    parser.add_argument('mapping_path', type=str,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('golden_standard', type=str,
                        help='Файл с наиболее близкими статьями')
    parser.add_argument('model_type', type=str,
                        help='Краткое имя модели векторизации, чтобы не путать файлы. '
                             'Будет использовано как имя pkl')
    args = parser.parse_args()

    i2lang = 'i2{}'.format(args.lang)
    lang2i = '{}2i'.format(args.lang)
    vecs_path = '{}/{}.pkl'.format(args.texts_path, args.model_type)
    texts_mapping = jload(open(args.mapping_path))
    corpus_vecs = pload(open(vecs_path, 'rb'))

    golden_standard_raw = (open(args.golden_standard, encoding='utf-8')).read().lower().splitlines()
    golden_standard_titles = {line.split('\t')[0]: line.split('\t')[1] for line in golden_standard_raw}
    # print(golden_standard_titles)
    golden_standard_ids = {texts_mapping[lang2i].get(art): texts_mapping[lang2i].get(sim_art) \
                           for art, sim_art in golden_standard_titles.items()}
    # print(golden_standard_ids)

    corp_sims_1, corp_sims_5, corp_sims_10 = [], [], []
    for i in tqdm(range(len(corpus_vecs))):
        target_vec = corpus_vecs[i]
        sim_ids = search_sim(target_vec, corpus_vecs)
        # print(sim_id)
        # ещё не оценка
        corp_sims_1.append(sim_ids[:1])
        corp_sims_5.append(sim_ids[:5])
        corp_sims_10.append(sim_ids[:10])

    # print(corp_sims_1)
    # print(corp_sims_5)
    # print(corp_sims_10)

    acc_1, acc_5, acc_10 = eval_acc()
    print('ТОП-1:\t{}\nТОП-5:\t{}\nТОП-10:\t{}'.format(acc_1, acc_5, acc_10))
