"""
Вспомогательный модуль, основанный на
https://github.com/akutuzov/webvectors/blob/master/preprocessing/rus_preprocessing_udpipe.py
Функция process принимает на вход необработанный русский текст (одно предложение на строку или один абзац на строку).
Он токенизируется, лемматизируется и размечается по частям речи с использованием UDPipe.
На выход подаётся список лемм с частями речи ["зеленый_NOUN, трамвай_NOUN"]
или пустой список, если в строке нет токенов, пригодных для разбора.

Также есть множество функциональных чатей речи для фильтрации стоп-слов и функция унификации символов unicode.
"""

import re

stop_pos = {'AUX', 'NUM', 'DET', 'PRON', 'ADP', 'SCONJ', 'CCONJ', 'INTJ', 'PART', 'X'}


def x_replace(word):
    """
    :param word: токен (строка)
    :return: последовательность x той же длины, что и токен (строка)
    """
    newtoken = 'x' * len(word)
    return newtoken


def clean_token(token, misc):
    """
    :param token: токен (строка)
    :param misc: содержимое поля "MISC" в CONLLU (строка)
    :return: очищенный токен (строка)
    """
    out_token = token.strip().replace(' ', '')

    if token == 'Файл' and 'SpaceAfter=No' in misc:
        return None

    return out_token


def clean_lemma(lemma, pos):
    """
    :param lemma: лемма (строка)
    :param pos: часть речи (строка)
    :return: очищенная лемма (строка)
    """
    out_lemma = lemma.strip().replace(' ', '').replace('_', '').lower()

    if '|' in out_lemma or out_lemma.endswith('.jpg') or out_lemma.endswith('.png'):
        return None

    if pos != 'PUNCT':
        if out_lemma.startswith('«') or out_lemma.startswith('»'):
            out_lemma = ''.join(out_lemma[1:])

        if out_lemma.endswith('«') or out_lemma.endswith('»'):
            out_lemma = ''.join(out_lemma[:-1])

        if out_lemma.endswith('!') or out_lemma.endswith('?') or out_lemma.endswith(',') or out_lemma.endswith('.'):
            out_lemma = ''.join(out_lemma[:-1])

    return out_lemma


def list_replace(search, replacement, text):
    """
    :param search: последовательность юникодовых символов, которые надо заменить (строка)
    :param replacement: юникодовый символ, на который надо заменить (строка)
    :param text: где заменить (строка)
    :return: текст после замены (строка)
    """
    search = [el for el in search if el in text]

    for c in search:
        text = text.replace(c, replacement)

    return text


def unify_sym(text):
    """
    :param text: текст в юникоде (строка)
    :return: текст с унифицироваными символами (строка)
    """
    # ['«', '»', '‹', '›', '„', '‚', '“', '‟', '‘', '‛', '”', '’']  ->  ['"']
    text = list_replace \
        ('\u00AB\u00BB\u2039\u203A\u201E\u201A\u201C\u201F\u2018\u201B\u201D\u2019',
         '\u0022', text)

    # ['‒', '–', '—', '―', '‾', '̅', '¯']  ->  [' -- ']
    text = list_replace \
        ('\u2012\u2013\u2014\u2015\u203E\u0305\u00AF',
         '\u2003\u002D\u002D\u2003', text)

    # ['‐', '‑']  ->  ['-']
    text = list_replace('\u2010\u2011', '\u002D', text)

    # [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '​', ' ', ' ', '⁠', '　'] -> [' ']
    text = list_replace \
        ('\u2000\u2001\u2002\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u200B\u202F\u205F\u2060\u3000',
         '\u2002', text)

    # ['  '] -> [' ']
    text = re.sub('\u2003\u2003', '\u2003', text)

    text = re.sub('\t\t', '\t', text)

    # ['ˌ', '̇', '̣', '•', '‣', '⁃', '⁌', '⁍', '∙', '◦', '·', '×', '⋅', '∙', '⁢']  ->  ['.']
    text = list_replace \
        ('\u02CC\u0307\u0323\u2022\u2023\u2043\u204C\u204D\u2219\u25E6\u00B7\u00D7\u22C5\u2219\u2062',
         '.', text)

    # ['∗']  ->  ['*']
    text = list_replace('\u2217', '\u002A', text)

    # ['≁', '≋', 'ⸯ', '҃']  ->  ['∽']
    text = list_replace('\u2241\u224B\u2E2F\u0483', '\u223D', text)

    text = list_replace('…', '...', text)

    # с надстрочными знаками и т.п.
    text = list_replace('\u00C4', 'A', text)
    text = list_replace('\u00E4', 'a', text)
    text = list_replace('\u00CB', 'E', text)
    text = list_replace('\u00EB', 'e', text)
    text = list_replace('\u1E26', 'H', text)
    text = list_replace('\u1E27', 'h', text)
    text = list_replace('\u00CF', 'I', text)
    text = list_replace('\u00EF', 'i', text)
    text = list_replace('\u00D6', 'O', text)
    text = list_replace('\u00F6', 'o', text)
    text = list_replace('\u00DC', 'U', text)
    text = list_replace('\u00FC', 'u', text)
    text = list_replace('\u0178', 'Y', text)
    text = list_replace('\u00FF', 'y', text)
    text = list_replace('\u00DF', 's', text)
    text = list_replace('\u1E9E', 'S', text)

    # валютные знаки
    currencies = list \
        (
            '\u20BD\u0024\u00A3\u20A4\u20AC\u20AA\u2133\u20BE\u00A2\u058F\u0BF9\u20BC\u20A1\u20A0\u20B4\u20A7\u20B0'
            '\u20BF\u20A3\u060B\u0E3F\u20A9\u20B4\u20B2\u0192\u20AB\u00A5\u20AD\u20A1\u20BA\u20A6\u20B1\uFDFC\u17DB'
            '\u20B9\u20A8\u20B5\u09F3\u20B8\u20AE\u0192'
        )

    alphabet = list \
        (
            '\t\n\r '
            'абвгдеёзжийклмнопрстуфхцчшщьыъэюяАБВГДЕЁЗЖИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ'
            ',.[]{}()=+-−*&^%$#@!~;:§/\|"'
            '0123456789'
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ' + "'"
        )

    allowed = set(currencies + alphabet)

    cleaned_text = [sym for sym in text if sym in allowed]
    cleaned_text = ''.join(cleaned_text)

    return cleaned_text


def process(pipeline, text, keep_pos=1, keep_punct=0, keep_stops=1):
    """
    :param pipeline: пайплайн udpipe
    :param text: текст на разбор (строка)
    :param keep_pos: прикреплять ли к леммам частеречные теги (pseudo-boolean int)
    :param keep_punct: сохранять ли знаки препинания (pseudo-boolean int)
    :param keep_stops: сохранять ли токены, получившие тег функциональной части речи (pseudo-boolean int)
    :return: леммы текста (список строк)
    """

    entities = {'PROPN'}
    named = False
    memory = []
    mem_case = None
    mem_number = None
    tagged_toks = []

    # обрабатываем текст, получаем результат в формате conllu:
    processed = pipeline.process(text)

    # пропускаем строки со служебной информацией:
    content = [line for line in processed.split('\n') if not line.startswith('#')]

    # извлекаем из обработанного текста леммы, тэги и морфологические характеристики
    tagged = [w.split('\t') for w in content if w]

    for line_tags in tagged:

        if len(line_tags) != 10:
            continue

        (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = line_tags
        token = clean_token(token, misc)
        lemma = clean_lemma(lemma, pos)

        if not lemma or not token:
            continue

        if pos in entities:
            if '|' not in feats:
                tagged_toks.append('{}_{}'.format(lemma, pos))
                continue
            morph = {el.split('=')[0]: el.split('=')[1] for el in feats.split('|')}

            if 'Case' not in morph or 'Number' not in morph:
                tagged_toks.append('{}_{}'.format(lemma, pos))
                continue

            if not named:
                named = True
                mem_case = morph['Case']
                mem_number = morph['Number']

            if morph['Case'] == mem_case and morph['Number'] == mem_number:
                memory.append(lemma)
                if 'SpacesAfter=\\n' in misc or 'SpacesAfter=\s\\n' in misc:
                    named = False
                    past_lemma = '::'.join(memory)
                    memory = []
                    tagged_toks.append('{}_PROPN'.format(past_lemma))

            else:
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                tagged_toks.append('{}_PROPN'.format(past_lemma))
                tagged_toks.append('{}_{}'.format(lemma, pos))

        else:
            if not named:
                if pos == 'NUM' and token.isdigit():
                    lemma = x_replace(token)
                tagged_toks.append('{}_{}'.format(lemma, pos))

            else:
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                tagged_toks.append('{}_PROPN'.format(past_lemma))
                tagged_toks.append('{}_{}'.format(lemma, pos))

    if not keep_punct:
        tagged_toks = [word for word in tagged_toks if word.split('_')[1] != 'PUNCT']

    if not keep_stops:
        tagged_toks = [word for word in tagged_toks if word.split('_')[-1] not in stop_pos]

    if not keep_pos:
        tagged_toks = [word.split('_')[0] for word in tagged_toks]

    return tagged_toks
