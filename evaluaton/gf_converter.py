"""На вход принимаются:
1) path - путь к директории с csv-файлами результатов гугл-опроса формата "5-8.csv"(т.е. ответы для статей запроса 5-8)
2) path2 - путь к директории для сохранения результатов
На выходе получаем 4 отдельных csv-таблицы по каждому поисковику"""

import os
import pandas as pd
import re
import numpy as np

path = "C:/Users/79850/Desktop/учеба/НИС/гугл-формы"
path2 = "C:/Users/79850/Desktop/учеба/НИС/models_eval"


for root, dirs, files in os.walk(path):
    g_forms = files

searchers = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]

for form in g_forms:
    form_df = pd.read_csv(path + '/' + form)
    article_1 = int(re.findall('^([0-9]+)-', form)[0])
    article_4 = int(re.findall('-([0-9]+)\.csv', form)[0])
    art_range = np.arange(article_1, article_4 + 1)  # диапазон статей из названия файла типа "9-12.csv"
    print(art_range)
    answ = len(form_df.columns)
    for i in range(4):
        searcher_answers = form_df.iloc[:, range(2+i, answ, 4)]
        for j in range(len(searcher_answers.columns)):
            searcher_answers.rename(columns={searcher_answers.columns[j]: art_range[j]}, inplace=True)
        searchers[i] = searchers[i].append(searcher_answers, sort=False)

# print(searchers[0].T)


for i in range(4):
    searchers[i].T.to_csv(path2+"/{}.csv".format(i+1), index=False, sep=',', header=None)
