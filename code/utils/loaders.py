'''
Загрузка всякой всячины (эмбеддинги, словари, проекции)
'''
import csv
import gensim
from gensim import models
import gzip
import logging
import numpy as np
import os
from tqdm import tqdm
import zipfile
from json import load as jload
from utils.preprocessing import clean_ext, get_dirs

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

w2v_path_binarity = {
    '.bin.gz': True,
    '.bin': True,
    '.txt.gz': False,
    '.txt': False,
    '.vec.gz': False,
    '.vec': False
}


def get_binarity(path):
    binary = 'NA'
    for ext in w2v_path_binarity:
        if path.endswith(ext):
            binary = w2v_path_binarity.get(ext)
            break
    return binary


def load_embeddings(embeddings_path):
    binary = get_binarity(embeddings_path)

    if binary != 'NA':
        model = models.KeyedVectors.load_word2vec_format(embeddings_path, binary=binary,
                                                         unicode_errors='replace')
    # ZIP archive from the NLPL vector repository:
    elif embeddings_path.endswith('.zip'):
        with zipfile.ZipFile(embeddings_path, "r") as archive:
            # Loading and showing the metadata of the model:
            # metafile = archive.open('meta.json')
            # metadata = json.loads(metafile.read())
            # for key in metadata:
            #    print(key, metadata[key])
            # print('============')

            stream = archive.open("model.bin")  # или model.txt, чтобы взглянуть на модель
            model = models.KeyedVectors.load_word2vec_format(
                stream, binary=True, unicode_errors='replace')
    else:
        # Native Gensim format?
        model = models.KeyedVectors.load(embeddings_path)
        # If you intend to train further: emb_model = models.Word2Vec.load(embeddings_file)

    model.init_sims(replace=True)  # Unit-normalizing the vectors (if they aren't already)
    return model


def save_w2v(vocab, output_path):
    """аналог save_word2vec_format для простого словаря, не сортируем по частотам"""
    binary = get_binarity(output_path)
    total_vec = len(vocab)
    vectors = np.array(list(vocab.values()))
    vector_size = vectors.shape[1]
    logging.info("storing {}x{} projection weights into {}".format(total_vec, vector_size, output_path))
    assert (len(vocab), vector_size) == vectors.shape
    with gensim.utils.open(output_path, 'wb') as fout:
        fout.write(gensim.utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # TODO: store in sorted order: most frequent words at the top
        for word, row in tqdm(vocab.items(), desc='Saving'):
            if binary:
                row = row.astype(np.float32)
                fout.write(gensim.utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(gensim.utils.to_utf8("{} {}\n".format(word, ' '.join(repr(val) for val in row))))


def load_projection(projection_path):
    projection = np.loadtxt(projection_path, delimiter=',')
    return projection


def load_mapping(mapping_path):
    """ключи маппинга делаем снова числами"""
    raw_mapping = jload(open(mapping_path))
    mapping = {}
    for dict_name in raw_mapping:
        map_dict = raw_mapping[dict_name]
        # print(map_dict)
        if dict_name.endswith('i'):  # маппинг названий в индексы
            mapping[dict_name] = {k: int(v) for k, v in map_dict.items()}
        elif dict_name.startswith('i'):  # маппинг индексов в названия
            mapping[dict_name] = {int(k): v for k, v in map_dict.items()}
    return mapping


def load_article_data(article_data_path):
    '''получаем словарь хеш: название, ссылка'''
    lines = open(article_data_path, encoding='utf-8').read().splitlines()
    article_data = {line.split('\t')[0]:
                    {'real_title': line.split('\t')[1], 'url': line.split('\t')[2]} for line in lines}
    # print(article_data)
    return article_data


def load_embedding_dict(model):
    return {word: model[word] for word in model.wv.vocab}


def load_vocab(path, clean_pos=False):
    raw = open(path, encoding='utf-8').read().lower().splitlines()
    if clean_pos:  # если прикреплены pos-теги
        raw = [line.split('_')[0] for line in raw]
    vocab = set(raw)
    # print(len(vocab))
    return vocab


# def send_to_archieve(archieve_path, model_path, remove_source=True):
#     with gzip.open(archieve_path, 'wb') as zipped_file:
#         logging.info('Saving vectors into archieve')
#         zipped_file.writelines(open(model_path, 'rb'))
#         logging.info('Vectors are saved into archieve')
#     if remove_source:
#         logging.info('Deleting source file {}'.format(model_path))
#         os.remove(model_path)
#     logging.info('Saved vectors {} to archive {}'.format(model_path, archieve_path))


# def save_text_vectors(vectors, output_path, remove_source=True):
#     '''Сохранение даже в бинарном виде через текстовый формат, очень долго'''
#     # генерим пути
#     if output_path.endswith('gz'):  # если путь -- архив
#         model_path = clean_ext(output_path)
#         archieve_path = output_path
#     else:
#         model_path = output_path
#         archieve_path = ''
#
#     binary = get_binarity(output_path)
#
#     if binary:  # если название бинарное, а нам нужно временное текстовое
#         text_model_path = clean_ext(model_path)+'.vec'
#         bin_model_path = output_path  # возможно, оно уже с архивом, gensim разберётся
#     else:
#         text_model_path = model_path
#         bin_model_path = ''
#
#     # print('''
#     # text_model_path: {}
#     # bin_model_path: {}
#     # archieve_path: {}
#     # '''.format(text_model_path, bin_model_path, archieve_path))
#
#     # генерим текстовый формат w2v
#     logging.info('Saving vectors in the text w2v format to {}'.format(text_model_path))
#     vec_str = '{} {}'.format(len(vectors), len(list(vectors.values())[0]))
#     for word, vec in tqdm(vectors.items(), desc='Formatting'):
#         vec_str += '\n{} {}'.format(word, ' '.join(str(v) for v in vec))
#     open(text_model_path, 'w', encoding='utf-8').write(vec_str)
#
#     if binary:  # конвертируем через gensim и сразу архивируем
#         logging.info('Converting text w2v format into binary to {}'.format(bin_model_path))
#         model = load_embeddings(text_model_path)
#         model.save_word2vec_format(bin_model_path, binary=True)
#         if remove_source:
#             logging.info('Deleting source file {}'.format(text_model_path))
#             os.remove(text_model_path)
#
#     else:  # если нужен текстовый формат, проверяем, нужно ли архивировать
#         if archieve_path:
#             send_to_archieve(archieve_path, text_model_path, remove_source)
#         else:
#             logging.info('Saved vectors to {}'.format(output_path))


def format_task_name(description):
    return 'TASK::{}'.format(description.replace(' ', '_'))


def load_task_terms(file_path, column_name):
    task_terms = {}
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter='\t')
        for row in csvreader:
            descript = row['description']
            task_name = format_task_name(descript)
            terms = row[column_name].split()
            # print(task_name, terms, url)
            task_terms[task_name] = terms
        return task_terms


def split_paths(joint_path, texts_paths):
    # делим пути по + или задаём столько пустых, сколько пришло папок с текстами
    # TODO: принимать один общий или выдавать ошибку
    if joint_path:
        paths = joint_path.split('+')
    else:
        paths = [''] * len(texts_paths)
    return paths


def create_dir(path):
    path_dirs = get_dirs(path)
    if path_dirs:
        try:
            os.makedirs(path_dirs)
        except FileExistsError:
            pass
