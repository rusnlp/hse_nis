from json import loads
import time
from tqdm import tqdm
from json import load, dump
import os
from os import mkdir


for dir in ['ru-titles', 'en-titles', 'ruwiki', 'enwiki']:
    try:
        mkdir(dir)
    except OSError:
        pass


enall = 'enwiki-20191001.json'
enlet = '0123456789abcdefghijklmnopqrstuvwxyz'
ruall = 'ruwiki_12_2018.json'
rulet = '0123456789абвгдеёжзиклмнопрстуфхцчшщъыьэюя'


def download_subcorpus(lang, iter):
    if lang == 'ru':
        corpus = ruall
    elif lang == 'en':
        corpus = enall

    with open(corpus) as wikifile:
        titles = []
        for i in tqdm(range(iter*10000)): # 10000, 20000, 30000, 40000, 50000
            while i < (iter-1)*10000: #00000, 10000, 20000, 30000, 40000
                wikifile.readline()
                break
            else:
                #print(i)
                titles.append(loads(wikifile.readline()).get('title'))
    #print()
    print(titles)

    open('{}-titles/{}-{}.txt'.format(lang, lang, iter), 'w', encoding='utf-8').write('\n'.join(sorted(titles)))



def distribute(lang, iter):
    if lang == 'ru':
        corpus = ruall
        letters = rulet
    elif lang == 'en':
        corpus = enall
        letters = enlet

    with open('{}-titles/{}-{}.txt'.format(lang, lang, iter), encoding='utf-8') as f:
        titles = f.read().lower().split('\n')
        for t in tqdm(titles):
            if t[0] in letters:
                #i = letters.index(t[0])
                #print(i)
                open('{}-titles/{}.txt'.format(lang, t[0]), 'a', encoding='utf-8').write(t+'\n')


def get_articles(lang):
    raw_titles = open('titles.txt', encoding='utf-8').read().lower().split('\n')
    titles = [tuple(pair.split('\t')) for pair in raw_titles]
    print(titles)

    if lang == 'ru':
        lang_i = {title[0]:i for i, title in enumerate(titles)}
        langall = ruall
    elif lang == 'en':
        lang_i = {title[1]:i for i, title in enumerate(titles)}
        langall = enall

    print(lang_i)

    with open(langall) as wikifile:
        titles = []
        for i in tqdm(range(50000)):
            art = loads(wikifile.readline())
            #print(art.keys())
            title = art['title'].lower()
            text = art['section_texts']
            if title.lower() in lang_i:
                #print(True, title)
                #print(text)
                open('{}wiki/{}.txt'.format(lang, lang_i[title]), 'w', encoding='utf-8').write(' '.join(text))



def remap():
    raw = open('titles.txt', encoding='utf-8').readlines()
    titles = [(line.strip().split('\t')[0], line.strip().split('\t')[1]) for line in raw]
    rui_dict = {i:'{}.txt'.format(v[0]) for i, v in enumerate(titles)}
    rut_dict = {'{}.txt'.format(v[0]):i for i, v in enumerate(titles)}
    eni_dict = {i:'{}.txt'.format(v[1]) for i, v in enumerate(titles)}
    ent_dict = {'{}.txt'.format(v[1]):i for i, v in enumerate(titles)}
    titles_mapping = {}
    titles_mapping['ru2i'] = rut_dict
    titles_mapping['en2i'] = ent_dict
    titles_mapping['i2ru'] = rui_dict
    titles_mapping['i2en'] = eni_dict
    print(titles_mapping)
    dump(titles_mapping, open('titles_mapping.json', 'w'))


    titles_mapping = load(open('titles_mapping.json'))
    print(titles_mapping)

    for i, w in tqdm(titles_mapping['i2ru'].items()):
        prev_name = 'ruwiki/{}.txt'.format(i)
        new_name = 'ruwiki/{}'.format(w)
        #print('{}\t-->\t{}'.format(prev_name, new_name))
        os.rename(prev_name, new_name)

    for i, w in tqdm(titles_mapping['i2en'].items()):
        prev_name = 'enwiki/{}.txt'.format(i)
        new_name = 'enwiki/{}'.format(w)
        #print('{}\t-->\t{}'.format(prev_name, new_name))
        os.rename(prev_name, new_name)


for i in range(1,6):
    download_subcorpus('ru', i)
    
for i in range(1,6):
    download_subcorpus('en', i)


for i in range(1,6):
    distribute('ru', i)

for i in range(1,6):
    distribute('en', i)


get_articles('ru')
get_articles('en')

remap()
