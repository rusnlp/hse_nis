from utils.loaders import load_vocab, load_task_terms

vocab_path = '../words/cross_muse_orig_vocab.txt'
# vocab_path = '../words/cross_muse_ext_vocab.txt'
model_vocab = load_vocab(vocab_path)

task_path = '../words/nlpub.tsv'
task_cols = ['terms_short', 'terms_long']

task_lost_path = '../filtered_words/orig/vocabs/task_lost_saved.tsv'
# task_lost_path = '../filtered_words/ext/vocabs/task_lost_saved.tsv'

col_lost_strs = []
for task_col in task_cols:
    task_terms = load_task_terms(task_path, task_col)
    lost_strs = ['{}\t{}\t{}\t{}'.format(task_col, 'lost_proc', 'saved', 'lost')]
    for task in task_terms:
        terms = set(task_terms[task])
        lost = terms - model_vocab
        lost_proc = (len(lost)/len(terms)) * 100
        saved = terms - lost
        # print(lost_proc, lost)
        lost_strs.append('{}\t{}\t{}\t{}'.format(task, round(lost_proc, 1),
                                               ', '.join(saved), ', '.join(lost)))
    col_lost_strs.append('\n'.join(lost_strs))

open(task_lost_path, 'w', encoding='utf-8').write('\n\n'.join(col_lost_strs))





