import pandas as pd


paths = ['results_muse.txt', 'results_proj.txt', 'results_trans.txt', 'results_vecmap.txt']
for path in paths:
    with open(path, 'r', encoding='utf-8') as file:
        df = pd.read_csv(file, sep='\t', header=None)
        df = df.drop(df.columns[[2, 3]], axis=1)
        df[0] = df[0].apply(lambda x: str(x)[:2])
        df.to_csv('{}_id_and_urls.tsv'.format(path[:-4]), sep='\t', index=False, header=False)
