"""""
На вход принимается путь к таблице tsv. В ней обязательно есть 2 столбца: 
Target article - статья запроса 
Results - полученные ближайшие статьи
Также есть столбцы с оценками моделей. Их число может варьироваться и они должны иметь 
следующий вид:
"Muse 21", где "Muse" - название модели, "21" - id или порядковый номер аннотатора
На выходе получаем Krippendorff's alpha и среднюю оценку по каждой модели

python eval_score.py --path=C:/data/RUSNLP_MAP_EVAL.TSV
"""""

import argparse
import re
import krippendorff
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description='Средняя оценка аннотаторов и коэффициент их согласия для каждой модели')
    parser.add_argument('--path', type=str, required=True, help='Путь к файлу с оценками')
    return parser.parse_args()


def open_excel(path):
    df = pd.read_csv(path, sep='\t', header=0)
    return df


# находим названия моделей с помощью регулярных выражений
def find_models(path):
    dataframe = open_excel(path)
    columns = dataframe.columns
    model_names, column_names = set(), []
    for column in columns:
        if 'Unnamed' in column:  # чтобы не считывался какой-нибудь пустой столбец
            break
        name = re.findall('(.+) [1-9]+$', column)
        model_names.update(name)
        if name:
            column_names += [column]
    return model_names, column_names


def mean_and_krip(path):
    models, columns = find_models(path)
    df = open_excel(path)
    for model in models:
        model_columns = []
        for column in columns:
            if model in column:
                model_columns += [column]
        model_df = df[model_columns]
        models_mean = model_df.mean()
        krip = krippendorff.alpha(model_df.T)
        print('Krippendorff\'s alpha coefficient for {}:'.format(model), krip)
        total_mean = models_mean.sum() / len(models_mean)
        print('Mean value for {}:'.format(model), total_mean)


if __name__ == '__main__':
    args = parse_args()
    mean_and_krip(args.path)
