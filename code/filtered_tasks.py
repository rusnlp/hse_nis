import argparse
from utils.loaders import load_vocab, load_task_terms


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Какие слова сохранились и потерялись при векторизации задач')
#     parser.add_argument('--vocab_path', type=str, required=True,
#                         help='Путь к словарю модели')
#     parser.add_argument('--task_path', type=str, required=True,
#                         help='Путь к файлу с задачами и их словами')
#     parser.add_argument('--task_lost_path', type=str, required=True,
#                         help='Путьо которому будет сохранена проекция (можно перечислить через +)')
#     return parser.parse_args()

vocab_path = '../words/cross_muse_orig_vocab.txt'
# vocab_path = '../words/cross_muse_ext_vocab.txt'
model_vocab = load_vocab(vocab_path)

task_path = '../words/nlpub.tsv'
task_cols = ['terms_short', 'terms_long']

task_lost_path = '../filtered_words/orig/vocabs/task_lost_saved.tsv'
# task_lost_path = '../filtered_words/ext/vocabs/task_lost_saved.tsv'

col_lost_strs = ['{}\t{}\t{}\t{}\t{}'.format('column', 'task', 'lost_proc', 'saved', 'lost')]
for task_col in task_cols:
    task_terms = load_task_terms(task_path, task_col)
    lost_strs = []
    for task in task_terms:
        terms = set(task_terms[task])
        lost = terms - model_vocab
        lost_proc = (len(lost)/len(terms)) * 100
        saved = terms - lost
        # print(lost_proc, lost)
        lost_strs.append('{}\t{}\t{}\t{}\t{}'.format(task_col, task, round(lost_proc, 1),
                                               ', '.join(saved), ', '.join(lost)))
    col_lost_strs.append('\n'.join(lost_strs))

open(task_lost_path, 'w', encoding='utf-8').write('\n\n'.join(col_lost_strs))






# col_lost_strs = ['{}\t{}\t{}\t{}\t{}'.format('column', 'task', 'lost_proc', 'saved', 'lost')]
# from utils.preprocessing import get_corpus
# from collections import Counter
# en_path = '../texts_conf/texts/en_conllu'
# ru_path = '../texts_conf/texts/ru_conllu'
#
# en_corpus = []
# for text in get_corpus(en_path, lemmatize=0).values():
#     en_corpus.extend(text)
#
# ru_corpus = []
# for text in get_corpus(ru_path, lemmatize=0).values():
#     ru_corpus.extend(text)


# en_counter = Counter(en_corpus)
#
# open('en_freqs.tsv', 'w', encoding='utf-8').\
#     write('\n'.join(['{}\t{}'.format(word, en_counter[word]) for word in sorted(en_counter, key=en_counter.get, reverse=True)]))


# ru_counter = Counter(ru_corpus)
#
# open('ru_freqs.tsv', 'w', encoding='utf-8').\
#     write('\n'.join(['{}\t{}'.format(word, ru_counter[word]) for word in sorted(ru_counter, key=ru_counter.get, reverse=True)]))


# ru_freqs = {line.split('\t')[0]: line.split('\t')[1] for line in open('ru_freqs.tsv', encoding='utf-8').read().split('\n')}
# en_freqs = {line.split('\t')[0]: line.split('\t')[1] for line in open('en_freqs.tsv', encoding='utf-8').read().split('\n')}
# print(en_freqs)

# terms_short = load_task_terms(task_path, task_cols[0]).values()
# terms_long = load_task_terms(task_path, task_cols[1]).values()
#
# terms_freqs = {}
# en_terms_freqs = {}
# ru_terms_freqs = {}
# for terms in terms_short:
#     for term in terms:
#         ru = int(ru_freqs.get(term, '0'))
#         en = int(en_freqs.get(term, '0'))
#         if ru > en:
#             ru_terms_freqs[term] = ru
#         else:
#             en_terms_freqs[term] = en
#
# print(ru_terms_freqs)
# print(en_terms_freqs)
#
# open('../words/ru_term_freqs.tsv', 'w', encoding='utf-8').write('\n'.join(
# ['{}\t{}'.format(word, ru_terms_freqs[word]) for word in sorted(ru_terms_freqs, key=ru_terms_freqs.get, reverse=True)]
# ))
# open('../words/en_term_freqs.tsv', 'w', encoding='utf-8').write('\n'.join(
#     ['{}\t{}'.format(word, en_terms_freqs[word]) for word in sorted(en_terms_freqs, key=en_terms_freqs.get, reverse=True)]
# ))
# # print(terms_freqs)
