"""
Оценка точности предсказания для всего корпуса:
верный ответ вверху, в первых 5, в первых 10 рейтинга.
Корпус должен быть предварительно лемматизирован и векторизован
"""

import argparse
from json import load as jload
from pickle import load as pload
from tqdm import tqdm

from monocorp_search import search_similar
from utils.arguments import arg_to_list


def parse_args():
    parser = argparse.ArgumentParser(
        description='Оценка качества поиска: проверка среди 1, 5, 10 ближайших)')
    parser.add_argument('--lang', type=str, required=True,
                        help='Язык, для которого разбираем; нужен для определения словаря в маппинге')
    parser.add_argument('--mapping_path', type=str, required=True,
                        help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('--corpus_vectors_path', type=str, required=True,
                        help='Путь к файлу pkl, в котором лежит векторизованный корпус')
    parser.add_argument('--golden_standard_path', type=str, required=True,
                        help='Файл с парами наиболее близких статей')
    parser.add_argument('--top_ns', type=str, default=[1, 5, 10],
                        help='Среди скольки ближайших искать совпадение '
                             '(перечисление чисел через запятую без пробела; default: [1, 5, 10]))')
    return parser.parse_args()


def get_golden_standart(path, mapping, lang2i_name):
    """
    :param path: путь к json с золотым стандартом (строка)
    :param mapping: маппинг заголовков в индексы и обратно (словарь словарей)
    :param lang2i_name: название словаря индекс-заголовок в маппинге (строка)
    :return: индексы заголовков в порядке близости к данному (список)
    """
    raw = (open(path, encoding='utf-8')).read().lower().splitlines()
    titles = {line.split('\t')[0]: line.split('\t')[1] for line in raw}
    ids = {mapping[lang2i_name].get(art): mapping[lang2i_name].get(sim_art)
           for art, sim_art in titles.items()}
    return ids


def predict_sim(vector, vectors):
    """
    :param vector: вектор текста (np.array)
    :param vectors: матрица векторов корпуса (np.array)
    :return: индексы текстов в порядке убывания близости к данному (список)
    """
    sim_dict = search_similar(vector, vectors)
    sorted_simkeys = sorted(sim_dict, key=sim_dict.get, reverse=True)

    return sorted_simkeys[1:]  # не 0, т.к. там сама статья


def eval_acc(top, golden_standard_ids, predicted_ids):
    """
    :param top: в скольких первых предсказаниях искать верное (int)
    :param golden_standard_ids: индексы "правильных" ближайших текстов (список)
    :param predicted_ids: отсортированные предсказанные индексы для каждого текста (список списков)
    :return: точность предсказания (float)
    """
    intersections = [len({golden_standard_ids[i]} & set(predicted_ids[i][:top]))
                     for i in range(len(predicted_ids))]
    acc = intersections.count(1) / len(intersections)
    return acc


def main():
    args = parse_args()
    args.top_ns = arg_to_list(args.top_ns)

    lang2i = '{}2i'.format(args.lang)
    texts_mapping = jload(open(args.mapping_path))
    corpus_vecs = pload(open(args.corpus_vectors_path, 'rb'))

    golden_standard_ids = get_golden_standart(args.golden_standard_path, texts_mapping, lang2i)

    corp_sims = []  # списки предсказанного для каждого текста

    for i in tqdm(range(len(corpus_vecs))):
        target_vec = corpus_vecs[i]
        sim_ids = predict_sim(target_vec, corpus_vecs)
        corp_sims.append(sim_ids)

    top_accuracies = [eval_acc(top_n, golden_standard_ids, corp_sims) for top_n in args.top_ns]

    top_strings = ['ТОП-{}:\t{}'.format(top_n, top_acc)
                   for top_n, top_acc in zip(args.top_ns, top_accuracies)]
    print('\n'.join(top_strings))


if __name__ == "__main__":
    main()
