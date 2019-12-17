"""
python string_vectorizer.py "сюда подойдет любой длинный текст либо одно слово в формате строки" models//ru_syntagrus.udpipe  models/rusvect.bin
python string_vectorizer.py "just a random string" models//en_partut.udpipe models/enwiki.bin
"""


from preprocess import unify_sym, process, stop_pos
from ufal.udpipe import Model, Pipeline
from gensim import models
import numpy as np
import argparse


def string_vectorizer(keep_stops):
    parser = argparse.ArgumentParser(description='Превращаем строку в вектор')
    parser.add_argument('String_line', type=str,
                        help='type in a string line')
    parser.add_argument('Udpipe_model_path', type=str,
                        help='Udpipe model path')
    parser.add_argument('w2v_model_path', type=str,
                        help='Udpipe model path')
    args = parser.parse_args()
    string = args.String_line.lower()
    model = Model.load(args.Udpipe_model_path)
    process_pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    w2v = models.KeyedVectors.load_word2vec_format(args.w2v_model_path, binary=True, encoding='utf-8')
    res = unify_sym(string.strip())
    all_lemmas = process(process_pipeline, res, keep_pos=True, keep_punct=False)
    if not keep_stops:
        lemmas = [lem for lem in all_lemmas if lem.split('_')[-1] not in stop_pos]
    else:
        lemmas = all_lemmas
    words = [token for token in lemmas if token in w2v]
    t_vecs = np.zeros((len(words), w2v.vector_size))
    for i, token in enumerate(words):
        t_vecs[i, :] = w2v.get_vector(token)
    t_vec = np.sum(t_vecs, axis=0)
    t_vec = np.divide(t_vec, len(words))
    #print(t_vec)
    return t_vec


if __name__ == "__main__":
    string_vectorizer(False)
