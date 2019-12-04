from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from json import load as jload
from pickle import load as pload
import argparse

# Для каждого текста в корпусе получаем индекс ближайшего к данному
#TODO: мы весь корпус считаем дважды, надо в табличку
def search_one_sim(target_vec, corpus_vecs): # подаём текст как вектор и векторизованный корпус
    similars = {} # словарь {индекс текста в корпусе: близость к данному}
    for i, v in enumerate(corpus_vecs):
        sim = cosine_similarity(target_vec.reshape(1, target_vec.shape[0]), v.reshape(1, v.shape[0])) # вычисляем косинусную близость для данного вектора и вектора текста
        #print(sim)
        similars[i] = sim[0][0] # для индекса текста добавили его близость к данному в словарь
    sorted_simkeys = sorted(similars, key=similars.get, reverse=True)  # сортируем словарь по значениям в порядке убывания: сортируем список кортежей (key, value) по value
    return sorted_simkeys[1] # не 0, т.к. там он сам



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Векторизация корпуса и сохранение его в pkl в папке с текстами')
    parser.add_argument('lang', type=str, help='Заголовок статьи в формате txt (только назание, без формата), для которой ищем ближайшие')
    parser.add_argument('texts_path', type=str, help='Папка, в которой лежат тексты в формате txt')
    parser.add_argument('mapping_path', type=str, help='Файл маппинга заголовков в индексы и обратно в формате json')
    parser.add_argument('model_type', type=str, help='Краткое имя модели векторизации, чтобы не путать файлы. Будет использовано как имя pkl')
    args = parser.parse_args()

    i2lang = 'i2{}'.format(args.lang)
    lang2i = '{}2i'.format(args.lang)
    vecs_path = '{}/{}.pkl'.format(args.texts_path, args.model_type)
    texts_mapping = jload(open(args.mapping_path))
    corpus_vecs = pload(open(vecs_path, 'rb'))

    for i in tqdm(range(len(corpus_vecs))):
        target_vec = corpus_vecs[i]
        sim_id = search_one_sim(target_vec, corpus_vecs)
        print(sim_id)