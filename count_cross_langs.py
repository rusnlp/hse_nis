'''Считаем количество результатов на другом языке в рекомендациях'''

from collections import Counter
from numpy import sum

trans_path = "texts_wiki\\search_results\\results_trans_50.txt"
proj_path = "texts_wiki\\search_results\\results_proj_50.txt"
muse_path = "texts_wiki\\search_results\\results_muse_50.txt"
vecmap_path = "texts_wiki\\search_results\\results_vecmap_50.txt"


def get_origs(path, type='conf'):
    raw = [line.split('\t') for line in open(path, encoding='utf-8').readlines() if line != '\n']
    # print(raw)
    if type == 'conf':
        n = 1020
    elif type == 'wiki':
        n = 2754
    ru_origs = raw[:n]
    en_origs = raw[n:]
    # print(ru_origs)
    # print(len(ru_origs), len(en_origs))
    return ru_origs, en_origs


def get_cross_langs(origs, other):
    lines = [line for line in origs if '.' not in line[0]]
    # print(len(lines))
    # print(lines)
    cross_langs = [line[0] for line in lines if line[2] == other]
    return Counter(cross_langs)


def count_cross_langs(path, type):
    ru_origs, en_origs = get_origs(path, type)

    ru_cross_langs = get_cross_langs(ru_origs, 'en')
    en_cross_langs = get_cross_langs(en_origs, 'ru')

    return ru_cross_langs, en_cross_langs


def print_idx(cross_langs, group=False):
    counts = {str(i): 0 for i in range(1, 51)}
    counts.update(cross_langs)
    # print(cross_langs)
    # print(counts)

    if group:
        group_counts = {}
        for i in range(1, 11):
            start = (i-1)*5+1
            end = i*5+1
            idx = [i for i in range(start, end)]
            print(idx)
            g_sum = sum([cross_langs[str(id)] for id in idx])
            # print(g_sum)
            group_counts['{}-{}'.format(start, end)] = g_sum
        print('\n'.join([str(val) for val in group_counts.values()]))
    else:
        print('\n'.join([str(val) for val in counts.values()]))


for path in [proj_path, trans_path, muse_path, vecmap_path]:
    print('\n', path)
    ru_cross_langs, en_cross_langs = count_cross_langs(path, type='wiki')
    print_idx(ru_cross_langs, group=True)
    print('-' * 50)
    print_idx(en_cross_langs, group=True)
    # print(ru_cross_langs, en_cross_langs, sep='\n')
    print('=' * 50)



