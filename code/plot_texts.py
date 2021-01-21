"""
python plot_texts.py --model_path=../models/common_tok_muse_orig.bin.gz --method=pca+tsne --vis_path=../visual/tok_muse_orig --mapping_path=../texts_conf/mapping.json
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import pylab as plot
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.loaders import load_mapping, load_embeddings, create_dir
from monocorp_search import get_lang


def parse_args():
    parser = argparse.ArgumentParser(
        description='Визуализация векторизованных текстов и задач')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Путь к векторам, которые нужно визуализировать')
    parser.add_argument('--method', type=str, default='pca',
                        help='Метод понижения размерности (pca|tsne), можно перечислить через +')
    parser.add_argument('--vis_path', type=str, default='',
                        help='Путь к сохранённой картинке без расширения(!))')
    parser.add_argument('--mapping_path', type=str, required=True,
                        help='Путь к файлу маппинга (нужен для определения класса текста (язык, задача))')

    return parser.parse_args()


def visualize(words, matrix, classes, method, fname):
    if method == 'pca':
        embedding = PCA(n_components=2, random_state=42)
    elif method == 'tsne':
        perplexity = int(len(words) ** 0.5)  # We set perplexity to a square root of the words number
        embedding = TSNE(n_components=2, perplexity=perplexity, metric='cosine', n_iter=500, init='pca')

    y = embedding.fit_transform(matrix)

    class_set = [c for c in set(classes)]
    colors = plot.cm.rainbow(np.linspace(0, 1, len(class_set)))
    class2color = [colors[class_set.index(w)] for w in classes]

    xpositions = y[:, 0]
    ypositions = y[:, 1]

    # plot.clf()

    for word, x, y, color in tqdm(zip(words, xpositions, ypositions, class2color), desc='Plotting with {}'.format(method)):
        plot.scatter(x, y, 30, marker='.', color=color)
        lemma = word.replace('TASK::', '')
        mid = len(lemma) / 2
        mid *= 4  # TODO Should really think about how to adapt this variable to the real plot size
        # plot.annotate(lemma, xy=(x, y), size=5, color='black')

    plot.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plot.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    save_name = '{}_{}.png'.format(fname, method)
    plot.savefig(save_name, dpi=150,
                 bbox_inches='tight')
    print('Plot saved to {}'.format(save_name))

    # plot.show()
    # plot.close()
    # plot.clf()


def main():
    args = parse_args()

    methods = args.method.split('+')
    model = load_embeddings(args.model_path)
    mapping = load_mapping(args.mapping_path)

    vec_class = {
        'task': 0,
        'en': 1,
        'ru': 2
    }

    words = []
    matrix = []
    classes = []
    for word in model.vocab:
        if word.startswith('TASK::'):
            w_class = 'task'
        else:
            w_class = get_lang(word, mapping)
        classes.append(vec_class[w_class])
        words.append(word)
        matrix.append(model[word])
    matrix = np.array(matrix)

    create_dir(args.vis_path)

    for method in methods:
        visualize(words, matrix, classes, method, args.vis_path)


if __name__ == '__main__':
    main()