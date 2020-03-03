# Автоматически делаем поисковые запросы на основе
# tsv таблицы с названиями и языками статей
# и сохраняем выводы командной строки в файл json
import os
import subprocess

# Задайте значение переменных test_task_path - абсолютный путь к директории с кодом, mapping_path - абсолютный путь к маппингу статей
# corpus_embeddings_path - абсолютный путь к нужному пиклу, save_path - куда сохраним, top - мы договорились, что будет 5
# tsv_path
test_task_path = 'C:/Users/79850/Documents/Python/RUSNLP/git/hse_nis/test_task'
mapping_path = 'C:/data-txt/extracted_texts/mapping.json'
corpus_embeddings_path = 'C:/data-txt/extracted_texts/MUSE.pkl'
save_path = 'C:/data-txt/extracted_texts/MUSE_results.txt'
tsv_path = 'C:/data-txt/extracted_texts/20ru20en.tsv'
top = 5


titles = []
with open(tsv_path, 'r', encoding='utf-8') as articles:
    for line in articles.readlines():
        titles += [line.split('\t')[0]]
os.chdir(test_task_path)
results = []
for title in titles:
    search = 'python monolang_search.py --target_article_path={} --lang=cross --mapping_path={} --corpus_embeddings_path={}' \
             ' --top={}'.format(title, mapping_path, corpus_embeddings_path, 5)
    output = subprocess.check_output(search, encoding='utf-8')
    #print(output)
    results += [output]

with open(save_path, 'w', encoding= 'utf-8') as file:
    for result in results:
        file.write(result)
