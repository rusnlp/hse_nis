"""
python train_mono_w2v.py --texts_paths=../texts_conf/texts/en_conllu+../texts_conf/texts/ru_conllu --model_paths=../models/en_w2v_tok.bin.gz+../models/ru_w2v_tok.bin.gz --lemmatize=0 --min_count=2 --max_final_vocab=10000
"""

import argparse
from gensim.models import Word2Vec
from utils.loaders import get_binarity, split_paths
from utils.preprocessing import get_corpus


def parse_args():
    parser = argparse.ArgumentParser(
        description='Обучение одноязычного word2vec на корпусе')
    parser.add_argument('--texts_paths', type=str, required=True,
                        help='Путь к текстам в формате conllu (можно перечислить через +)')
    parser.add_argument('--model_paths', type=str, required=True,
                        help='Путь к моделям w2v (можно перечислить через +)')
    parser.add_argument('--lemmatize', type=int, required=True,
                        help='Брать ли леммы текстов (0|1)')
    parser.add_argument('--keep_pos', type=int, default=0,
                        help='Возвращать ли слова, помеченные pos-тегами (0|1; default: 0)')
    parser.add_argument('--keep_punct', type=int, default=0,
                        help='Сохранять ли знаки препинания (0|1; default: 0)')
    parser.add_argument('--keep_stops', type=int, default=0,
                        help='Сохранять ли слова, получившие тег функциональной части речи '
                             '(0|1; default: 0)')
    parser.add_argument('--join_propn', type=int, default=0,
                        help='Склеивать ли именованные сущности (0|1; default: 0)')
    parser.add_argument('--join_token', type=str, default='::',
                        help='Как склеивать именованные сущности (default: ::)')
    parser.add_argument('--unite', type=int, default=1,
                        help='Убирать ли деление на предложения (0|1; default: 1)')
    parser.add_argument('--vec_size', type=int, default=300,
                        help='Размерность эмбеддингов (default: 300)')
    parser.add_argument('--window', type=int, default=5,
                        help='(default: 5)')
    parser.add_argument('--min_count', type=int, default=5,
                        help='(default: 5)')
    parser.add_argument('--max_final_vocab', type=int, default=None,
                        help='(default: max_final_vocab)')

    return parser.parse_args()


def train_w2v(model_path, corpus, vec_size, window, min_count, max_final_vocab, save=True):
    # print(corpus.values())
    w2v = Word2Vec(corpus.values(), size=vec_size, window=window,
                   min_count=min_count, max_final_vocab=max_final_vocab)
    if save:
        binary = get_binarity(model_path)
        if binary != 'NA':
            w2v.wv.save_word2vec_format(model_path, binary=binary)
        else:  # Native Gensim format?
            w2v.wv.save(model_path)
    return w2v


def main():
    args = parse_args()
    texts_paths = args.texts_paths.split('+')
    model_paths = split_paths(args.model_paths, texts_paths)

    for texts_path, model_path in zip(texts_paths, model_paths):
        corpus = get_corpus(texts_path, args.lemmatize, args.keep_pos, args.keep_punct,
                            args.keep_stops, args.join_propn, args.join_token, args.unite)
        _ = train_w2v(model_path, corpus, args.vec_size, args.window, args.min_count, args.max_final_vocab)


if __name__ == '__main__':
    main()
