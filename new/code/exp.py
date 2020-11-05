# from ufal.udpipe import Model, Pipeline
# import os
# import numpy as np
# from tqdm import tqdm
# from utils.preprocessing import process_line, get_text, clean_lemma, punctuation
# from utils.loaders import load_embeddings
# from gensim import models
#
# lang = 'en'
# input_path = '../texts_wiki/'+lang
# output_path = '../texts_wiki/'+lang+'_conllu'
# try:
#     os.mkdir(output_path)
# except OSError:
#     pass
# udpipe_path = '../models/{}.udpipe'.format(lang)
# embed_path = '../models/{}_muse.vec'.format(lang)
#
# udpipe_model = Model.load(udpipe_path)
# pipeline = Pipeline(udpipe_model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
#
# def make_conllu(input_path, name, output_path):
#     input_name = '{}/{}'.format(input_path, name)
#     output_name = '{}/{}conllu'.format(output_path, name[:-3])
#
#     text = open(input_name, encoding='utf-8').read().lower().strip()
#     processed = pipeline.process(text)
#     # print([processed])
#     # print(type(processed))
#     open(output_name, 'w', encoding='utf-8').write(processed)
#
#
# for name in tqdm(os.listdir(input_path)):
#     make_conllu(input_path, name, output_path)
#
# # embed_model = load_embeddings(embed_path)
#
#
# # def save_corpus_vectors(vectors):  # TODO: в класс векторайзера?
# #     # TODO: подгружать эту модель в словарь, чтобы можно было его менять
# #     vec_str = '{} {}'.format(len(vectors), len(list(vectors.values())[0]))
# #     for word, vec in vectors.items():
# #         vec_str += '\n{} {}'.format(word, ' '.join(str(v) for v in vec))  #TODO: так можно? так сделать везде?
# #     open('models/{}_corp.vec'.format(lang), 'w', encoding='utf-8').write(vec_str)
# #
# #
# # vectors = {}
# for name in tqdm(os.listdir(output_path)):
#     input_name = '{}/{}'.format(output_path, name)
#     text = open(input_name, encoding='utf-8').read().strip()
#     preprocessed = get_text(text, lemmatize=1, keep_pos=0, keep_punct=0, keep_stops=0)
#     # print(preprocessed)
# #     vector = []
# #     for word in preprocessed:
# #         if word in embed_model:
# #             vector.append(embed_model.get_vector(word))
# #     vector = np.mean(vector, axis=0)
# #     # print(vector.shape)
# #     vectors[name] = vector  # TODO: названия с conllu
# # save_corpus_vectors(vectors)
# # # print(vectors)
#
#
# # model = load_embeddings('texts/common_lem_muse.vec')
# # print(model.vocab.keys())
# # print(model.get_vector('аксон.conllu'))
# # print(model.most_similar('аксон.conllu', topn=5))
# # sims = model.most_similar('аксон.conllu', topn=len(model.vocab))  # TODO: нет встроенного способа взять все? False не работает
# # print(sims)
# # print(type(sims))
# # print(len(sims))
# #
# # a = model.get_vector('аксон.conllu')
# # b = {t: s for t, s in model.similar_by_vector(a, topn=5)}
# # print(b)
# # print(b.pop('аксон.conllu'))
# # print(b)

print('khlkjb.hb'.split('.')[:-1])