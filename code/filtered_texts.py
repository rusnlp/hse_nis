"""
Собираем слова, которые сохранились и отфильтровались по словарям моделей, считаем проценты
и запоминаем тексты, слова из которых не получилось проанализировать (скорее всего, их там нет)

python filtered_texts.py --texts_paths=../texts_conf/texts/en_conllu+../texts_conf/texts/ru_conllu --lemmatize=1 --vocab_path=../words/cross_muse_orig_vocab.txt --stats_paths=../filtered_words/orig/en_muse_stats_lem.json+../filtered_words/orig/ru_muse_stats_lem.json --not_analyzed_path=../filtered_words/orig/not_analyzed_lem.txt
python filtered_texts.py --texts_paths=../texts_conf/texts/en_conllu+../texts_conf/texts/ru_conllu --lemmatize=0 --vocab_path=../words/cross_muse_orig_vocab.txt --stats_paths=../filtered_words/orig/en_muse_stats_tok.json+../filtered_words/orig/ru_muse_stats_tok.json --not_analyzed_path=../filtered_words/orig/not_analyzed_tok.txt
"""

import argparse
from json import dump
from tqdm import tqdm
from utils.preprocessing import get_corpus
from utils.loaders import load_vocab, split_paths, create_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description='Анализируем, сколько слов (и какие) потерялось и сохранилось при векторизации')
    parser.add_argument('--texts_paths', type=str, required=True,
                        help='Путь к текстам в формате conllu (можно перечислить через +)')
    parser.add_argument('--lemmatize', type=int, required=True,
                        help='Брать ли леммы текстов (0|1)')
    parser.add_argument('--vocab_path', type=str, required=True,
                        help='Путь к файлу со словарём модели')
    parser.add_argument('--stats_paths', type=str, required=True,
                        help='Путь к json с результатами анализа (можно перечислить через +)')
    parser.add_argument('--not_analyzed_path', type=str, default='',
                        help='Путь к файлу со списком текстов, которые не удалось проанализирвать')

    return parser.parse_args()


def analyze_filtered(corpus, vocab):
    not_analyzed = []
    stats = {}

    for file in tqdm(corpus, desc='Analyzing corpus'):

        text = corpus[file]
        text_vocab = set(text)

        if len(text_vocab) <= 1:  # там, скорее всего, какой-то символ
            not_analyzed.append(file)
        else:

            file_stats = {'text_len': len(text),
                          'text_vocabsize': len(text_vocab)}
            # print(file_stats)

            # проверяем, сколько слов отфильтровалось:
            # - насколько текст стал короче
            # - насколько словарь стал меньше

            # не хотим искать пересечение множеств, чтобы подсчитать
            # потерю длины текста, а не уникальных токенов
            vectorized = []
            lost = []
            for tok in text:
                # print(tok)
                if tok.lower() in vocab:  # ПОС-ТЕГИ ТОЖЕ К НИЖНЕМУ НАДО!
                    # print(tok, True)
                    vectorized.append(tok)
                else:
                    lost.append(tok)

            # print(text)
            # print(vocab)
            # print(lost)

            file_stats['vec_saved_len'] = len(vectorized)
            file_stats['vec_saved_vocab'] = list(set(vectorized))
            file_stats['vec_lost_len'] = len(lost)
            file_stats['vec_lost_vocab'] = list(set(lost))
            file_stats['vec_lost_vocabsize'] = len(file_stats['vec_lost_vocab'])

            file_stats['vec_losttok_proc'] = file_stats['vec_lost_len'] / file_stats['text_len'] * 100
            file_stats['vec_lostvocab_proc'] = file_stats['vec_lost_vocabsize'] / file_stats['text_vocabsize'] * 100

            # print(file_stats.keys())

            stats[file] = file_stats

    print(list(stats.values())[0].keys(), '\n')

    return stats, not_analyzed


def main():
    args = parse_args()
    texts_paths = args.texts_paths.split('+')
    stats_paths = split_paths(args.stats_paths, texts_paths)

    vocab = load_vocab(args.vocab_path)
    not_analyzed_all = set()

    for texts_path, stats_path in zip(texts_paths, stats_paths):
        corpus = get_corpus(texts_path, args.lemmatize)
        stats, not_analyzed = analyze_filtered(corpus, vocab)

        create_dir(stats_path)
        dump(stats, open(stats_path, 'w', encoding='utf-8'))

        not_analyzed_all.update(not_analyzed)

    print('Не проанализировано файлов: {}'.format(len(not_analyzed_all)))
    if not_analyzed_all and args.not_analyzed_path:
        open(args.not_analyzed_path, 'w', encoding='utf-8').write('\n'.join(not_analyzed_all))


if __name__ == '__main__':
    main()