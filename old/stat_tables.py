from collections import Counter
from json import load
from statistics import mean, median
from os import mkdir

#TODO: Анализировать не множества потерянных слов, а списки?

stats_path = 'filtered_words/'
en_simple_stats_path = stats_path + 'en_simple_stats.json'
ru_simple_stats_path = stats_path + 'ru_simple_stats.json'
ru_trans_stats_path = stats_path + 'ru_trans_stats.json'
en_muse_stats_path = stats_path + 'en_muse_stats.json'
ru_muse_stats_path = stats_path + 'ru_muse_stats.json'

vocab_path = 'filtered_words/vocabs/'
try:
    mkdir(vocab_path)
except OSError:
    pass

en_simple_lv_path = vocab_path + 'en_simple_lost.tsv'
ru_simple_lv_path = vocab_path + 'ru_simple_lost.tsv'
ru_trans_lv_path = vocab_path + 'ru_trans_lost.tsv'
en_muse_lv_path = vocab_path + 'en_muse_lost.tsv'
ru_muse_lv_path = vocab_path + 'ru_muse_lost.tsv'

en_simple_sv_path = vocab_path + 'en_simple_saved.tsv'
ru_simple_sv_path = vocab_path + 'ru_simple_saved.tsv'
ru_trans_sv_path = vocab_path + 'ru_trans_saved.tsv'
en_muse_sv_path = vocab_path + 'en_muse_saved.tsv'
ru_muse_sv_path = vocab_path + 'ru_muse_saved.tsv'

def get_stats(val_list):
    return mean(val_list), median(val_list), max(val_list), min(val_list)


def group_stats(stats_path, savedvocab_path, lostvocab_path, total=False):
    # print(stats_path)
    stats = load(open(stats_path, encoding='utf-8'))
    # print(stats.keys())

    saved_vocab = []
    lost_vocab = []
    lost_lens = []
    lost_vocabsizes = []
    losttok_procs = []
    lostvocab_procs = []

    for file in stats:
        stat = stats[file]
        # print(stat.keys())

        saved_vocab.extend(stat['vec_saved_vocab'])
        lost_vocab.extend(stat['vec_lost_vocab'])

        if total:
            lost_lens.append(int(stat['total_lost_len']))
            lost_vocabsizes.append(int(stat['total_lost_vocabsize']))
            losttok_procs.append(float(stat['total_losttok_proc']))
            lostvocab_procs.append(float(stat['total_lostvocab_proc']))
        else:
            lost_lens.append(int(stat['vec_lost_len']))
            lost_vocabsizes.append(int(stat['vec_lost_vocabsize']))
            losttok_procs.append(float(stat['vec_losttok_proc']))
            lostvocab_procs.append(float(stat['vec_lostvocab_proc']))

    saved_vocab_freqs = Counter(saved_vocab)
    saved_vocab_sort = sorted(saved_vocab_freqs, key=saved_vocab_freqs.get, reverse=True)
    lost_vocab_freqs = Counter(lost_vocab)
    lost_vocab_sort = sorted(lost_vocab_freqs, key=lost_vocab_freqs.get, reverse=True)

    open(savedvocab_path, 'w', encoding='utf-8').write('\n'.join(['{}\t{}'.format(k, saved_vocab_freqs[k]) for k in saved_vocab_sort]))
    open(lostvocab_path, 'w', encoding='utf-8').write('\n'.join(['{}\t{}'.format(k, lost_vocab_freqs[k]) for k in lost_vocab_sort]))

    # print(saved_vocab_freqs)
    # print(lost_vocab_freqs)
    mean_lost_lens, med_lost_lens, max_lost_lens, min_lost_lens = get_stats(lost_lens)
    mean_losttok_procs, med_losttok_procs, max_losttok_procs, min_losttok_procs = get_stats(losttok_procs)
    mean_lost_vocabsizes, med_lost_vocabsizes, max_lost_vocabsizes, min_lost_vocabsizes = get_stats(lost_vocabsizes)
    mean_lostvocab_procs, med_lostvocab_procs, max_lostvocab_procs, min_lostvocab_procs = get_stats(lostvocab_procs)

    stat_res = '''{}
\t\tLENS (%)\t\tVOCAB (%)
MEAN\t{:.2f} ({:.2f})\t{:.2f} ({:.2f})
MED\t\t{:.2f} ({:.2f})\t{:.2f} ({:.2f})
MAX\t\t{:.2f} ({:.2f})\t{:.2f} ({:.2f})
MIN\t\t{:.2f} ({:.2f})\t{:.2f} ({:.2f})
'''.format(stats_path,
        mean_lost_lens, mean_losttok_procs, mean_lost_vocabsizes, mean_lostvocab_procs,
        med_lost_lens, med_losttok_procs, med_lost_vocabsizes, med_lostvocab_procs,
        max_lost_lens, max_losttok_procs, max_lost_vocabsizes, max_lostvocab_procs,
        min_lost_lens, min_losttok_procs, min_lost_vocabsizes, min_lostvocab_procs
    )

    return stat_res


en_simple_stats = group_stats(en_simple_stats_path, en_simple_sv_path, en_simple_lv_path)
ru_simple_stats = group_stats(ru_simple_stats_path, ru_simple_sv_path, ru_simple_lv_path)
ru_trans_stats = group_stats(ru_trans_stats_path, ru_trans_sv_path, ru_trans_lv_path, total=True)
en_muse_stats = group_stats(en_muse_stats_path, en_muse_sv_path, en_muse_lv_path)
ru_muse_stats = group_stats(ru_muse_stats_path, ru_muse_sv_path, ru_muse_lv_path)

open(stats_path+'group_stats.txt', 'w', encoding='utf-8').\
    write('\n'.join([en_simple_stats, ru_simple_stats, ru_trans_stats, en_muse_stats, ru_muse_stats]))