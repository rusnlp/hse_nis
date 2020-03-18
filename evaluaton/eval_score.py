"""
На вход подается csv-файл c бинарными оценками семантической близости найденных статей к статье запроса
На выходе получаем альфу Криппендорффа и среднюю оценку
"""

import argparse
import krippendorff
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description='Средняя оценка аннотаторов и коэффициент их согласия')
    parser.add_argument('--path', type=str, required=True, help='Путь к csv-файлу')
    return parser.parse_args()


def score(path):
    df = pd.read_csv(path, sep=',', header=None)
    mean = df.mean()
    total_mean = sum([i for i in mean])/len(mean)
    krip = krippendorff.alpha(df.T)
    print('Krippendorff\'s alpha coefficient:', krip)
    print('Mean value:', total_mean)
    return df


if __name__ == '__main__':
    args = parse_args()
    score(args.path)
