"""
Собираем слова, которые сохранились и отфильтровались по словарям моделей, считаем проценты
и запоминаем тексты, слова из которых не получилось проанализировать (скорее всего, их там нет)
"""

import os
from tqdm import tqdm
from json import dump
from utils.preprocessing import get_text, clean_ext


def load_vocab(path, clean=False):
    raw = open(path, encoding='utf-8').read().lower().splitlines()
    if clean:
        raw = [line.split('_')[0] for line in raw]
    vocab = set(raw)
    # print(len(vocab))
    return vocab


def get_corpus(texts_path, lemmatize, keep_pos=False, keep_punct=False, keep_stops=False, unite=True):
    """собираем тексты из conllu в словарь списков"""
    texts = {}
    for file in os.listdir(texts_path):
        text = open('{}/{}'.format(texts_path, file), encoding='utf-8').read().strip()
        preprocessed = get_text(text, lemmatize, keep_pos, keep_punct, keep_stops, unite)
        # print(preprocessed)
        texts[clean_ext(file)] = preprocessed

    return texts


def analyze_filtered(corpus, vocab, stats):
    not_analyzed = []

    for file in corpus:

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


def main(params):
    # print(params)
    texts_path = params['texts_path']
    vocab_path = params['vocab_path']
    stats_path = params['stats_path']
    lemmatize = params['lemmatize']

    vocab = load_vocab(vocab_path, clean=False)
    corpus = get_corpus(texts_path, lemmatize)

    stats = {}
    stats, not_analyzed = analyze_filtered(corpus, vocab, stats)
    dump(stats, open(stats_path, 'w', encoding='utf-8'))
    return not_analyzed


if __name__ == '__main__':
    lemmatize = False

    en_texts_path = '../texts_conf/texts/EN_CONLLU'
    ru_texts_path = '../texts_conf/texts/RU_CONLLU'
    model_vocab_path = '../words/cross_muse_vocab.txt'
    stats_dir = '../filtered_words/'
    en_tok_stats_path = stats_dir + 'en_muse_stats_tok.json'
    ru_tok_stats_path = stats_dir + 'ru_muse_stats_tok.json'
    en_lem_stats_path = stats_dir + 'en_muse_stats_lem.json'
    ru_lem_stats_path = stats_dir + 'ru_muse_stats_lem.json'
    try:
        os.mkdir(stats_dir)
    except OSError:
        pass

    params = {
        'en_muse_tok': {
            'texts_path': en_texts_path,
            'lemmatize': False,
            'vocab_path': model_vocab_path,
            'stats_path': en_tok_stats_path
        },
        'ru_muse_tok': {
            'texts_path': ru_texts_path,
            'lemmatize': False,
            'vocab_path': model_vocab_path,
            'stats_path': ru_tok_stats_path
        },
        'en_muse_lem': {
            'texts_path': en_texts_path,
            'lemmatize': True,
            'vocab_path': model_vocab_path,
            'stats_path': en_lem_stats_path
        },
        'ru_muse_lem': {
            'texts_path': ru_texts_path,
            'lemmatize': True,
            'vocab_path': model_vocab_path,
            'stats_path': ru_lem_stats_path
        }
    }

    not_analyzed_all = []
    for param_dict in tqdm(params):
        print(param_dict, params[param_dict])
        not_analyzed_one = main(params[param_dict])
        not_analyzed_all.extend(not_analyzed_one)
    not_analyzed_all = set(not_analyzed_all)
    print('Не проанализировано файлов: {}'.format(len(not_analyzed_all)))
    open(stats_dir+'not_analyzed.txt', 'w', encoding='utf-8').write('\n'.join(not_analyzed_all))

