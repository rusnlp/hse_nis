"""
Скрипт для объединения моделей в одну двуязычную и очищения словаря от мусора

python join_models.py --model_paths=../models/en_muse.bin.gz+../models/ru_muse.bin.gz --common_path=../models/cross_muse_orig.bin.gz --removed_path=../words/cross_muse_removed.txt --remove_source=0
"""

import argparse
import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm
from utils.loaders import load_embeddings, save_w2v
from utils.preprocessing import alphabet


def parse_args():
    parser = argparse.ArgumentParser(
        description='Слияние моделей эмбеддингов в одну и фильтрация словаря от мусора')
    parser.add_argument('--model_paths', type=str, required=True,
                        help='Пути к моделям, которые будем объединять (можно перечислить через +)')
    parser.add_argument('--common_path', type=str, required=True,
                        help='Путь к объединённым векторам')
    parser.add_argument('--removed_path', type=str, default='',
                        help='Путь к файлу с удалёнными токенами')
    parser.add_argument('--remove_source', type=int, default=0,
                        help='Удалять ли исходный файл с векторами в текстовом формате (default: 0)')

    return parser.parse_args()


def main():
    args = parse_args()

    model_paths = args.model_paths.split('+')
    models = [load_embeddings(model_path) for model_path in model_paths]

    # очищаем словари моделей от мусора и смотрим, есть ли пересечения в моделях
    cross_dict = defaultdict(list)
    removed = set()
    for mi, model in enumerate(models):
        trash = []
        for wi, w in tqdm(enumerate(model.vocab), desc='Cleaning vocabs'):
            # ищем пересечение хотя бы с одним символом
            if set(w) & set(alphabet['lat']+alphabet['cyr']):
                # запоминаем позицию в словаре, чтобф разрешать конфликты по частотности
                cross_dict[w].append((wi, model.get_vector(w)))
            else:
                trash.append(w)
        removed.update(trash)
        print('Из {} удалено токенов: {}'.format(model_paths[mi], len(trash)))

    # составляем новый словарь модели, разрешая конфикты
    common_vecs = {}
    for w, vs in tqdm(cross_dict.items(), desc='Joining vocabs'):
        if len(vs) == 1:
            common_vecs[w] = vs[0][1]
        else:
            # смотрим, какой вектор был с минимальным индексом -- т.е. с большей частотой
            pos = [v[0] for v in vs]
            min_pos = np.argmin(pos)
            common_vecs[w] = vs[min_pos][1]

    print('Всего удалено токенов: {}'.format(len(removed)))
    save_w2v(common_vecs, args.common_path)

    if args.removed_path:
        open(args.removed_path, 'w', encoding='utf-8').write('\n'.join(removed))

    if args.remove_source:
        for model_path in model_paths:
            os.remove(model_path)


if __name__ == '__main__':
    main()
