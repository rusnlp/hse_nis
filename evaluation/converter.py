import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description='converter')
    parser.add_argument('--path', type=str,
                        required=True, help='Путь к исходному файлу с оценками')
    parser.add_argument('--save_as', type=str,
                        required=True, help='Куда сохранить конвертированный файл')
    parser.add_argument('--sep', type=str,
                        help='Тип сепаратора в csv-файле')
    parser.add_argument('--bin', type=str,
                        help='Перевод 5-балльной системы оценки в бинарную')
    return parser.parse_args()


def binary_values(df):
    df = df.apply({lambda x: 0 if (int(x) < 3) else 1})
    return df


def converter(input_path, output_path, sep, binar):
    if not sep:
        sep = ','
    df = pd.read_csv(input_path, sep=sep, header=None)
    df = df.drop([0]).drop([0, 1], axis=1)
    if binar:
        df = binary_values(df)
    df.to_csv(output_path, index=False, sep=',', header=None)


def main():
    args = parse_args()
    converter(args.path, args.save_as, args.sep, args.bin)


if __name__ == '__main__':
    main()
