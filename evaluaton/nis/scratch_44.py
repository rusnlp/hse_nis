#превращаем выводы командной строки в датафрейм с ссылками, если они есть
#если ссылок нет, оставляем название статьи
import pandas as pd
import re

path = 'Muse_results.txt'  # вход
path2 = "musecsv.csv"  # выход, csv таблица

with open (path, 'r', encoding='utf-8') as file:
    results = file.read()
#print(results)
target_articles = re.findall('Топ-5 ближайших статей к ([^:]+):', results)
result1 = re.findall('1\. ([^ ]+) ', results)
result2 = re.findall('2\. ([^ ]+) ', results)
result3 = re.findall('3\. ([^ ]+) ', results)
result4 = re.findall('4\. ([^ ]+) ', results)
result5 = re.findall('5\. ([^ ]+) ', results)
df = pd.DataFrame({'target': target_articles, '1': result1, '2': result2, '3': result3, '4': result4, '5': result5})


#print(df.loc[df['target'] == 'dialogue_2019_4a5c4385216b2220faf8860da8101e1c3744c9ca'])
titles = pd.read_csv('C:/data-txt/hash_title_url.tsv', sep='\t')

def file2url (filename):
    line = titles.loc[titles['hash'] == filename]
    url = line['url'].values[0]
    return url

#print(file2url ('dialogue_2019_4a5c4385216b2220faf8860da8101e1c3744c9ca'))


#['dialogue_2000_d779021b5cfc493595763bb6c4c3574894e715a9']

for index, row in df.iterrows():
    for i in range (0, 5):
        try:
            if file2url(row[i]) == '-':
                continue
            else:
                row[i] = file2url(row[i])
        except IndexError:
            row[i] = row[i]


df.to_csv (path2)

