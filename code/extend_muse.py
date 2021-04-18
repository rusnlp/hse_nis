"""
python extend_muse.py --src_model_paths=../models/en_w2v_tok.bin.gz+../models/ru_w2v_tok.bin.gz --tar_model_path=../models/cross_muse_orig.bin.gz --proj_paths=../models/en_muse_tok.proj+../models/ru_muse_tok.proj --src_words_paths=../filtered_words/import_tok_en.txt+../filtered_words/import_tok_ru.txt --not_added_paths=../filtered_words/not_added_tok_en.txt+../filtered_words/not_added_tok_ru.txt --ext_model_path=../models/cross_muse_ext.bin.gz
"""
import argparse
import numpy as np
from tqdm import tqdm
from utils.loaders import load_embeddings, load_embedding_dict, load_projection, save_w2v, split_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description='Проецируем вектора из одноязычного w2v во muse')
    parser.add_argument('--src_model_paths', type=str, required=True,
                        help='Путь к модели, вектора которой будут проецироваться '
                             '(можно перечислить через +)')
    parser.add_argument('--tar_model_path', type=str, required=True,
                        help='Путь к модели, в которую будут проецироваться вектора')
    parser.add_argument('--proj_paths', type=str, required=True,
                        help='Путь к файлу c матрицей трансформации векторов из '
                             'исходного пространства в целевое (можно перечислить через +)')
    parser.add_argument('--src_words_paths', type=str, required=True,
                        help='Путь к файлу txt со словами, которые нужно спроецировать '
                             '(можно перечислить через +)')
    parser.add_argument('--ext_model_path', type=str, required=True,
                        help='Путь к пополненной модели')
    parser.add_argument('--not_added_paths', type=str, default='',
                        help='Путь к файлу со списком слов, которые не удалось спроецировать '
                             '(можно перечислить через +)')

    return parser.parse_args()


def project_vec(src_vec, projection):
    '''Проецируем вектор'''
    test = np.mat(src_vec)
    test = np.c_[1.0, test]  # Adding bias term
    predicted_vec = np.dot(projection, test.T)
    predicted_vec = np.squeeze(np.asarray(predicted_vec))
    predicted_vec = predicted_vec / np.linalg.norm(predicted_vec)
    return predicted_vec


# обновляем целевую модель векторами пространства одного языка
def upd_monolang(src_model_path, proj_path, src_words_path, not_added_path, base_dict):
    src_model = load_embeddings(src_model_path)
    src_words = open(src_words_path, encoding='utf-8').read().split('\n')
    proj = load_projection(proj_path)

    ext_dict = {}
    not_added = []
    for word in tqdm(src_words, desc='Projecting'):
        try:
            src_vec = src_model[word]
            ext_dict[word] = project_vec(src_vec, proj)
        except KeyError:  # слишком редкое слово, нет в одноязычной w2v модели
            not_added.append(word)
    if not_added and not_added_path:
        open(not_added_path, 'w', encoding='utf-8').write('\n'.join(not_added))
    ext_dict.update(base_dict)  # чтобы добавленные слова были первыми в словаре muse
    return ext_dict


def main():
    args = parse_args()

    src_model_paths = args.src_model_paths.split('+')
    proj_paths = split_paths(args.proj_paths, src_model_paths)
    src_words_paths = split_paths(args.src_words_paths, src_model_paths)
    not_added_paths = split_paths(args.not_added_paths, src_model_paths)

    tar_model = load_embeddings(args.tar_model_path)
    tar_dict = load_embedding_dict(tar_model)

    for src_model_path, proj_path, src_words_path, not_added_path in \
            zip(src_model_paths, proj_paths, src_words_paths, not_added_paths):
        tar_dict = upd_monolang(src_model_path, proj_path, src_words_path, not_added_path, tar_dict)

    save_w2v(tar_dict, args.ext_model_path)


if __name__ == '__main__':
    main()