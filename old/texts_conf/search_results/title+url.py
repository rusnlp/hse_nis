import pandas as pd

titles = pd.read_csv('C:/data-txt/hash_title_url.tsv', sep='\t')
titles.columns = ['id', 'title', 'url']


def id2title (filename):
    line = titles.loc[titles['id'] == filename]
    title = line['title'].values[0]
    title = ' '.join(title.split(' ')[:3]) + '...'
    return title

paths = ['results_muse_5.txt', 'results_proj_5.txt', 'results_trans_5.txt', 'results_vecmap_5.txt']
for path in paths:
    with open(path, 'r', encoding='utf-8') as file:
        df = pd.read_csv(file, sep='\t', header=None)
        df[0] = df[0].apply(lambda x: str(x)[:2])
        df.columns = ['N', 'id', 'lang', 'similarity', 'url']
        df['title'] = df['id'].apply(lambda x: id2title(x))
        df = df.drop(['id', 'lang', 'similarity'], axis=1)
        df = df[['N', 'title', 'url']]
        df.to_csv('{}_gf.tsv'.format(path[:-4]), sep='\t', index=False, header=False)




#print(file2url ('dialogue_2019_4a5c4385216b2220faf8860da8101e1c3744c9ca'))

# df.to_csv('{}_id_and_urls.tsv'.format(path[:-4]), sep='\t', index=False, header=False)