"""
На вход подается csv-файл c бинарными оценками семантической близости найденных статей
к статье запроса
На выходе получаем альфу Криппендорффа и среднюю оценку
"""

import argparse
import krippendorff
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description='Средняя оценка аннотаторов и коэффициент их согласия')
    parser.add_argument('--path', '-p', type=str, required=True, help='Путь к csv-файлу')
    return parser.parse_args()


def score(path):
    df = pd.read_csv(path, sep=',', header=None)
    total_mean = df.mean(axis=1).mean()
    score_matrix = df.T.values
    krip = krippendorff.alpha(score_matrix, level_of_measurement='ratio')
    print('Krippendorff\'s alpha coefficient:', round(krip, 3))
    print('Mean score:', round(total_mean, 3))
    return df


if __name__ == '__main__':
    args = parse_args()
    score(args.path)
