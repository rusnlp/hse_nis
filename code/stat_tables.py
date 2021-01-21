"""
Запускать на результатах filtered_texts.py
Подсчитываем статистики из списков отфильтрованных слов, группируем и собираем в таблицы
В таблицу записывается потерянное
Также создаются списки отфильтрованных и сохранённых слов с частотами по всем текстам

python stat_tables.py --stats_paths=../filtered_words/orig/en_muse_stats_tok.json+../filtered_words/orig/ru_muse_stats_tok.json --saved_paths=../filtered_words/orig/vocabs/en_muse_tok_saved.tsv+../filtered_words/orig/vocabs/ru_muse_tok_saved.tsv --lost_paths=../filtered_words/orig/vocabs/en_muse_tok_lost.tsv+../filtered_words/orig/vocabs/ru_muse_tok_lost.tsv --group_stats_path=../filtered_words/orig/lost_stats_tok.txt
python stat_tables.py --stats_paths=../filtered_words/orig/en_muse_stats_lem.json+../filtered_words/orig/ru_muse_stats_lem.json --saved_paths=../filtered_words/orig/vocabs/en_muse_lem_saved.tsv+../filtered_words/orig/vocabs/ru_muse_lem_saved.tsv --lost_paths=../filtered_words/orig/vocabs/en_muse_lem_lost.tsv+../filtered_words/orig/vocabs/ru_muse_lem_lost.tsv --group_stats_path=../filtered_words/orig/lost_stats_lem.txt
"""

import argparse
from collections import Counter
from json import load
from statistics import mean, median
from tqdm import tqdm
from utils.loaders import split_paths, create_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description='Анализируем, сколько слов (и какие) потерялось и сохранилось при векторизации')
    parser.add_argument('--stats_paths', type=str, required=True,
                        help='Путь к json с результатами анализа (можно перечислить через +)')
    parser.add_argument('--saved_paths', type=str, default='',
                        help='Путь к файлу txt с сохранёнными токенами (можно перечислить через +)')
    parser.add_argument('--lost_paths', type=str, default='',
                        help='Путь к файлу txt с потерянными токенами (можно перечислить через +)')
    parser.add_argument('--group_stats_path', type=str, required=True,
                        help='Путь к файлу txt со сгруппированными статистиками')

    return parser.parse_args()


#TODO: Анализировать не множества потерянных слов, а списки?


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

    if savedvocab_path:
        open(savedvocab_path, 'w', encoding='utf-8').write('\n'.join(['{}\t{}'.format(k, saved_vocab_freqs[k]) for k in saved_vocab_sort]))
    if lostvocab_path:
        open(lostvocab_path, 'w', encoding='utf-8').write('\n'.join(['{}\t{}'.format(k, lost_vocab_freqs[k]) for k in lost_vocab_sort]))

    # print(saved_vocab_freqs)
    # print(lost_vocab_freqs)
    mean_lost_lens, med_lost_lens, max_lost_lens, min_lost_lens = get_stats(lost_lens)
    mean_losttok_procs, med_losttok_procs, max_losttok_procs, min_losttok_procs = get_stats(losttok_procs)
    mean_lost_vocabsizes, med_lost_vocabsizes, max_lost_vocabsizes, min_lost_vocabsizes = get_stats(lost_vocabsizes)
    mean_lostvocab_procs, med_lostvocab_procs, max_lostvocab_procs, min_lostvocab_procs = get_stats(lostvocab_procs)

    stats_str = '''{}
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

    return stats_str


def main():
    args = parse_args()
    stats_paths = args.stats_paths.split('+')
    saved_paths = split_paths(args.saved_paths, stats_paths)
    lost_paths = split_paths(args.lost_paths, stats_paths)

    all_stats_strs = []
    for stats_path, saved_path, lost_path in tqdm(zip(stats_paths, saved_paths, lost_paths),
                                                  desc='Grouping data'):
        create_dir(saved_path)
        create_dir(lost_path)

        stats_str = group_stats(stats_path, saved_path, lost_path)
        all_stats_strs.append(stats_str)

    open(args.group_stats_path, 'w', encoding='utf-8').write('\n'.join(all_stats_strs))


if __name__ == '__main__':
    main()

