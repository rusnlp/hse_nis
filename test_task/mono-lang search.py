'''
Считаем, что все тексты, которые ищем, априори добавлены в корпус, предобработаны, вектора для всего построены

Подаём название текста, язык, путь к папке с текстами, путь к маппингу, тип модели, которой векторизовали, можно подать кол-во ближайших статей
Получаем n ближайших записей в виде списка кортежей (заголовок, близость) -- напечатем рейтинг, если не сделали verbose=False

python "mono-lang search.py" кровь ru texts/ruwiki texts/titles_mapping.json simple
python "mono-lang search.py" blood en texts/enwiki texts/titles_mapping.json simple
python "mono-lang search.py" blood en texts/enwiki texts/titles_mapping.json simple --number=10
'''

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from json import load as jload
from pickle import load as pload
import argparse


# Для каждого текста в корпусе считаем косинусную близость к данному
def search_similar(target_vec, corpus_vecs): # подаём текст как вектор из корпуса и векторизованный корпус
    similars = {} # словарь {индекс текста в корпусе: близость к данному}
    for i, v in tqdm(enumerate(corpus_vecs)):
       # для индекса, вектора для каждого текста в корпусе
        # для cosine_similarity придётся изменить размерност векторов (dim,) -> (1, dim), т.е. вместо [0 0 0 0] получаем [[0 0 0 0]]
        sim = cosine_similarity(target_vec.reshape(1, target_vec.shape[0]), v.reshape(1, v.shape[0])) # вычисляем косинусную близость для данного вектора и вектора текста
        #print(sim)
        similars[i] = sim[0][0] # для индекса текста добавили его близость к данному в словарь
    #print(similars)
    return similars


# Ранжируем тексты по близости и принтим красивый списочек
def make_rating(similars, verbose, n): # принимаем словарь близостей
    sorted_simkeys = sorted(similars, key=similars.get, reverse=True) # сортируем словарь по значениям в порядке убывания: сортируем список кортежей (key, value) по value
    # similars: [i, j, ...]
    similars_list = [(texts_mapping[i2lang].get(str(simkey)), similars[simkey]) for simkey in sorted_simkeys]
    # [(i_title, sim), (j_title, sim), (...)]
    if verbose:
        print('\nТоп-{} ближайших статей к {}:'.format(n, target_article))
        for i, sim_item in enumerate(similars_list[1:n+1]): # на 0 индексе всегда будет сама статья, если она из корпуса
            print('{}. {} ({})'.format(i+1, sim_item[0], sim_item[1]))
    return similars_list[1:n+1] # если нужна будет одна статья, вернётся список с одним элементом

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Векторизация корпуса и сохранение его в pkl в папке с текстами')
    parser.add_argument('target_article_title', type=str, help='Язык, для которго разбираем, нужен для определения словаря в маппинге (ru/en)')
    parser.add_argument('lang', type=str, help='Заголовок статьи в формате txt (только назание, без формата), для которой ищем ближайшие')
    parser.add_argument('texts_path', type=str, help='Папка, в которой лежат тексты в формате txt')
    parser.add_argument('mapping_path', type=str, help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('model_type', type=str, help='Краткое имя модели векторизации, чтобы не путать файлы. Будет использовано как имя pkl')
    parser.add_argument('--number', type=int, default=1, help='Сколько близких статeй возвращать (default: 1)')
    #TODO: сделать вариант для возвращения всех
    parser.add_argument('--verbose', type=bool, default=True, help='Принтить ли рейтинг (default: True)')
    args = parser.parse_args()

    i2lang = 'i2{}'.format(args.lang)
    lang2i = '{}2i'.format(args.lang)
    vecs_path = '{}/{}.pkl'.format(args.texts_path, args.model_type)
    target_article_title = args.target_article_title.lower() # на всякий приведём к нижнему регистру
    target_article = '{}.txt'.format(target_article_title)
    texts_mapping = jload(open(args.mapping_path))
    corpus_vecs = pload(open(vecs_path, 'rb'))

    target_article_id = texts_mapping[lang2i].get(target_article)
    #print(target_article, target_article_id)
    target_article_vec = corpus_vecs[target_article_id]

    similars = search_similar(target_article_vec, corpus_vecs)
    #print(similars)
    make_rating(similars, n=args.number, verbose=args.verbose)
