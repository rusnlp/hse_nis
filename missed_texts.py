from os import mkdir
from json import load, dump
from tqdm import tqdm
from test_task.utils.loaders import load_bidict


def load_vocab(path, clean=False):
    raw = open(path, encoding='utf-8').read().lower().splitlines()
    if clean:
        raw = [line.split('_')[0] for line in raw]
    vocab = set(raw)
    # print(len(vocab))
    return vocab


stats = {}


def analyze_filtered(lemmatized, vocab, bidict=None):
    not_analyzed = []

    for file in list(lemmatized.keys()):

        text = lemmatized[file]
        text_vocab = set(text)

        if len(text_vocab) == 1:
            not_analyzed.append(file)
        else:

            file_stats = {'text_len': len(text),
                          'text_vocabsize': len(text_vocab)}
            # print(file_stats)

            if bidict:
                # print(True)
                # print(bidict)
                translated = []
                lost = []
                for tok in text:
                    # print(tok)
                    if tok in bidict:  # ПОС-ТЕГИ К НИЖНЕМУ не НАДО!
                        # print(tok, True)
                        translated.append(bidict[tok])
                    else:
                        lost.append(tok)

                file_stats['trans_saved_len'] = len(translated)
                file_stats['trans_saved_vocab'] = list(set(translated))
                file_stats['trans_lost_len'] = len(lost)
                file_stats['trans_lost_vocab'] = list(set(lost))
                file_stats['trans_lost_vocabsize'] = len(file_stats['trans_lost_vocab'])

                file_stats['trans_losttok_proc'] = file_stats['trans_lost_len'] / file_stats['text_len'] * 100
                file_stats['trans_lostvocab_proc'] = file_stats['trans_lost_vocabsize'] / file_stats['text_vocabsize'] * 100

                # временно делаем рокировку, сохраняя предыдущие значения
                text = translated
                # print(text)
                file_stats['orig_text_len'] = file_stats['text_len']
                file_stats['text_len'] = file_stats['trans_saved_len']
                file_stats['orig_text_vocabsize'] = file_stats['text_vocabsize']
                file_stats['text_vocabsize'] = len(file_stats['trans_saved_vocab'])

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

            if bidict:
                # print(True)
                # исправляем рокировку
                file_stats['trans_text_len'] = file_stats['text_len']
                file_stats['text_len'] = file_stats['orig_text_len']
                del(file_stats['orig_text_len'])

                file_stats['trans_text_vocabsize'] = file_stats['text_vocabsize']
                file_stats['text_vocabsize'] = file_stats['orig_text_vocabsize']
                del(file_stats['orig_text_vocabsize'])

                file_stats['total_lost_len'] = file_stats['trans_lost_len'] + file_stats['vec_lost_len']
                file_stats['total_lost_vocabsize'] = file_stats['trans_lost_vocabsize'] + file_stats['vec_lost_vocabsize']
                file_stats['total_losttok_proc'] = file_stats['total_lost_len'] / file_stats['text_len'] * 100
                file_stats['total_lostvocab_proc'] = file_stats['total_lost_vocabsize'] / file_stats['text_vocabsize'] * 100

            # print(file_stats.keys())

            stats[file] = file_stats

    print(stats[list(stats.keys())[0]].keys())
    print()

    return stats, not_analyzed


def main(*args):
    args = args[0]
    print(args)
    lemmatized_path = args['lemmatized_path']
    vocab_path = args['vocab_path']
    bidict_path = args['bidict_path']
    stats_path = args['stats_path']

    lemmatized = load(open(lemmatized_path, encoding='utf-8'))
    # print(lemmatized.keys())
    vocab = load_vocab(vocab_path, clean=False)
    # print(vocab)
    if bidict_path:
        print(True)
        bidict = load_bidict(bidict_path)
    else:
        bidict = {}
    # print(bidict)

    stats, not_analyzed = analyze_filtered(lemmatized, vocab, bidict)
    dump(stats, open(stats_path, 'w', encoding='utf-8'))
    return not_analyzed


if __name__ == '__main__':
    en_pos_lemmatized_path = 'texts_conf/texts/en_pos_lemmatized.json'
    ru_pos_lemmatized_path = 'texts_conf/texts/ru_pos_lemmatized.json'

    en_lemmatized_path = 'texts_conf/texts/en_lemmatized.json'
    ru_lemmatized_path = 'texts_conf/texts/ru_lemmatized.json'

    vocab_path = 'words/vocab/'

    en_simple_vocab_path = vocab_path + 'en_simple_vocab.txt'
    en_muse_vocab_path = vocab_path + 'en_muse_vocab.txt'

    ru_simple_vocab_path = vocab_path + 'ru_simple_vocab.txt'
    ru_muse_vocab_path = vocab_path + 'ru_muse_vocab.txt'

    bidict_path = 'words/ru-en_lem.txt'

    stats_path = 'filtered_words/'
    try:
        mkdir(stats_path)
    except OSError:
        pass

    en_simple_stats_path = stats_path + 'en_simple_stats.json'
    ru_simple_stats_path = stats_path + 'ru_simple_stats.json'
    ru_trans_stats_path = stats_path + 'ru_trans_stats.json'
    en_muse_stats_path = stats_path + 'en_muse_stats.json'
    ru_muse_stats_path = stats_path + 'ru_muse_stats.json'

    params = {
        'en_proj': {
            'lemmatized_path': en_pos_lemmatized_path,
            'vocab_path': en_simple_vocab_path,
            'bidict_path': '',
            'stats_path': en_simple_stats_path
        },
        'ru_proj': {
            'lemmatized_path': ru_pos_lemmatized_path,
            'vocab_path': ru_simple_vocab_path,
            'bidict_path': '',
            'stats_path': ru_simple_stats_path
        },

        # Для векмапа не имеет смысла, т.к. словари полностью совпадают

        'ru_trans': {
            'lemmatized_path': ru_pos_lemmatized_path,
            'vocab_path': en_simple_vocab_path,
            'bidict_path': bidict_path,
            'stats_path': ru_trans_stats_path
        },

        'en_muse': {
            'lemmatized_path': en_lemmatized_path,
            'vocab_path': en_muse_vocab_path,
            'bidict_path': '',
            'stats_path': en_muse_stats_path
        },
        'ru_muse': {
            'lemmatized_path': ru_lemmatized_path,
            'vocab_path': ru_muse_vocab_path,
            'bidict_path': '',
            'stats_path': ru_muse_stats_path
        }
    }

    # not_analyzed_all = []
    # for param_dict in tqdm(params):
    #     print(param_dict)
    #     not_analyzed = main(params[param_dict])
    #     not_analyzed_all.extend(not_analyzed)
    # not_analyzed_all = set(not_analyzed_all)
    # print(len(not_analyzed_all))
    # open(stats_path+'not_analyzed.txt', 'w', encoding='utf-8').write('\n'.join(not_analyzed_all))

    # main(params['en_proj'])
    # main(params['ru_proj'])
    # main(params['ru_trans'])
    # main(params['en_muse'])
    # main(params['ru_muse'])
