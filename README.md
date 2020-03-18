# hse_nis

## texts_conf
Вектора для текстов на статьях с конференций

## texts_wiki
Тексты и вектора для текстов на википедии

## evaluation
Оценки аннотаторов и скрипт для подсчёта средней оценки и согласия

## models
* русская модель udpipe (ru.udpipe): [russian-syntagrus-ud-2.4-190531.udpipe](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2998/russian-syntagrus-ud-2.4-190531.udpipe?sequence=74&isAllowed=y)
* английская модель udpipe (en.udpipe): [english-partut-ud-2.4-190531.udpipe](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2998/english-partut-ud-2.4-190531.udpipe?sequence=24&isAllowed=y)
* русские эмбеддинги (ru.bin): [ruwikiruscorpora_upos_skipgram_300_2_2019](http://vectors.nlpl.eu/repository/20/182.zip)
* английские эмбеддинги (en.bin): [300-English Wikipedia Dump of February 2017](http://vectors.nlpl.eu/repository/20/3.zip)
* русские muse (ru_muse.vec): [wiki.multi.ru.vec](https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.ru.vec)
* английские muse (en_muse.vec): [wiki.multi.en.vec](https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec)
* русские vecmap: [ru_vecmap.vec](https://yadi.sk/d/EswfSoTkJKJqRA) обучены самостоятельно
* английские vecmap: [en_vecmap.vec](https://yadi.sk/d/gyyLzlOkeiAwXg) обучены самостоятельно


## test_task
Код для поиска ближайших статей 

## words
Двуязычные словари, обученые проекции, словари моделей и т.п.
* русско-английский словарь [ru_en.txt](https://dl.fbaipublicfiles.com/arrival/dictionaries/ru-en.txt) (не лемматизированный и без pos-тегов, требует предобработки)


# Запуск 
## Статьи википедии
#### Создание маппинга
```python create_mapping.py --texts_path=../texts_wiki/ruwiki --lang=ru --mapping_path=../texts_wiki/mapping.json```
```python create_mapping.py --texts_path=../texts_wiki/enwiki --lang=en --mapping_path=../texts_wiki/mapping.json```

#### Предобработка
* С pos-тегами:

```python preprocess_corpus.py --texts_path=../texts_wiki/ruwiki --udpipe_path=../models/ru.udpipe --lemmatized_path=../texts_wiki/ru_pos_lemmatized.json```
```python preprocess_corpus.py --texts_path=../texts_wiki/enwiki --udpipe_path=../models/en.udpipe --lemmatized_path=../texts_wiki/en_pos_lemmatized.json```

* без pos-тегов (для muse):

```python preprocess_corpus.py --texts_path=../texts_conf/texts/RU --udpipe_path=../models/ru.udpipe --lemmatized_path=../texts_conf/texts/ru_lemmatized.json --keep_pos=0```
```python preprocess_corpus.py --texts_path=../texts_conf/texts/EN --udpipe_path=../models/en.udpipe --lemmatized_path=../texts_conf/texts/en_lemmatized.json --keep_pos=0```

#### Обучение проекции

```python learn_projection.py --src_model_path=../models/ru.bin --tar_model_path=../models/en.bin --bidict_path=../words/ru-en_lem.txt --proj_path=../words/ru-en_proj.txt```

#### Векторизация
* vecmap

```python vectorization_pipeline.py --src_lemmatized_path=../texts_wiki/ru_pos_lemmatized.json --tar_lemmatized_path=../texts_wiki/en_pos_lemmatized.json --direction=ru-en --method=model --mapping_path=../texts_wiki/mapping.json --src_embeddings_path=../models/ru_vecmap.vec --tar_embeddings_path=../models/en_vecmap.vec --src_output_vectors_path=../texts_wiki/ru_vecmap.pkl --tar_output_vectors_path=../texts_wiki/en_vecmap.pkl --common_output_vectors_path=../texts_wiki/common_vecmap.pkl```

* muse

```python vectorization_pipeline.py --src_lemmatized_path=../texts_wiki/ru_lemmatized.json --tar_lemmatized_path=../texts_wiki/en_lemmatized.json --direction=ru-en --method=model --mapping_path=../texts_wiki/mapping.json --src_embeddings_path=../models/ru_muse.vec --tar_embeddings_path=../models/en_muse.vec --src_output_vectors_path=../texts_wiki/ru_muse.pkl --tar_output_vectors_path=../texts_wiki/en_muse.pkl --common_output_vectors_path=../texts_wiki/common_muse.pkl```

* translation

```python vectorization_pipeline.py --src_lemmatized_path=../texts_wiki/ru_pos_lemmatized.json --tar_lemmatized_path=../texts_wiki/en_pos_lemmatized.json --direction=ru-en --method=translation --mapping_path=../texts_wiki/mapping.json --tar_embeddings_path=../models/en.bin --src_output_vectors_path=../texts_wiki/ru_trans.pkl --tar_output_vectors_path=../texts_wiki/en_trans.pkl --common_output_vectors_path=../texts_wiki/common_trans.pkl --bidict_path=../words/ru-en_lem.txt```

* projection

```python vectorization_pipeline.py --src_lemmatized_path=../texts_wiki/ru_pos_lemmatized.json --tar_lemmatized_path=../texts_wiki/en_pos_lemmatized.json --direction=ru-en --method=projection --mapping_path=../texts_wiki/mapping.json --src_embeddings_path=../models/ru.bin --tar_embeddings_path=../models/en.bin --src_output_vectors_path=../texts_wiki/ru_projection.pkl --tar_output_vectors_path=../texts_wiki/en_projection.pkl --common_output_vectors_path=../texts_wiki/common_projection.pkl --projection_path=../words/ru-en_proj.txt```


#### Поиск
* vecmap

```python monocorp_search.py --target_article_path=anemia.txt --lang=cross --mapping_path=../texts_wiki/mapping.json --corpus_vectors_path=../texts_wiki/common_vecmap.pkl --top=10```

* muse

```python monocorp_search.py --target_article_path=anemia.txt --lang=cross --mapping_path=../texts_wiki/mapping.json --corpus_vectors_path=../texts_wiki/common_muse.pkl --top=10```

* translation

```python monocorp_search.py --target_article_path=anemia.txt --lang=cross --mapping_path=../texts_wiki/mapping.json --corpus_vectors_path=../texts_wiki/common_trans.pkl --top=10```

* projection

```python monocorp_search.py --target_article_path=anemia.txt --lang=cross --mapping_path=../texts_wiki/mapping.json --corpus_vectors_path=../texts_wiki/common_projection.pkl --top=10```


#### Оценка
* vecmap 

```python evaluate_corpus.py --lang=cross --corpus_vectors_path=../texts_wiki/common_vecmap.pkl --mapping_path=../texts_wiki/mapping.json --golden_standard_path=../texts_wiki/standards/titles.txt```

* muse

```python evaluate_corpus.py --lang=cross --corpus_vectors_path=../texts_wiki/common_muse.pkl --mapping_path=../texts_wiki/mapping.json --golden_standard_path=../texts_wiki/standards/titles.txt```

* translation

```python evaluate_corpus.py --lang=cross --corpus_vectors_path=../texts_wiki/common_trans.pkl --mapping_path=../texts_wiki/mapping.json --golden_standard_path=../texts_wiki/standards/titles.txt```

* projection

```python evaluate_corpus.py --lang=cross --corpus_vectors_path=../texts_wiki/common_projection.pkl --mapping_path=../texts_wiki/mapping.json --golden_standard_path=../texts_wiki/standards/titles.txt```


#### Автоматический поиск для списка заголовков
* vecmap

```python automatic_search.py --titles_path=../texts_wiki/standards/titles.txt --mapping_path=../texts_wiki/mapping.json --corpus_vectors_path=../texts_wiki/common_vecmap.pkl --result_path=../texts_wiki/search_results/results_vecmap.txt --top=5```

* muse

```python automatic_search.py --titles_path=../texts_wiki/standards/titles.txt --mapping_path=../texts_wiki/mapping.json --corpus_vectors_path=../texts_wiki/common_muse.pkl --result_path=../texts_wiki/search_results/results_muse.txt --top=5```

* translation

```python automatic_search.py --titles_path=../texts_wiki/standards/titles.txt --mapping_path=../texts_wiki/mapping.json --corpus_vectors_path=../texts_wiki/common_trans.pkl --result_path=../texts_wiki/search_results/results_trans.txt --top=5```

* projection

```python automatic_search.py --titles_path=../texts_wiki/standards/titles.txt --mapping_path=../texts_wiki/mapping.json --corpus_vectors_path=../texts_wiki/common_projection.pkl --result_path=../texts_wiki/search_results/results_projection.txt --top=5```



## Статьи с конференций
#### Создание маппинга 
```python create_mapping.py --texts_path=../conf_evaluation/RU --lang=ru --mapping_path=../models_evaluation/mapping.json```
```python create_mapping.py --texts_path=../conf_evaluation/EN --lang=en --mapping_path=../models_evaluation/mapping.json```

#### Предобработка
* С pos-тегами:

```python preprocess_corpus.py --texts_path=../texts_conf/texts/RU --udpipe_path=../models/ru.udpipe --lemmatized_path=../texts_conf/texts/ru_pos_lemmatized.json```
```python preprocess_corpus.py --texts_path=../texts_conf/texts/EN --udpipe_path=../models/en.udpipe --lemmatized_path=../texts_conf/texts/en_pos_lemmatized.json```

* без pos-тегов (для muse):

```python preprocess_corpus.py --texts_path=../texts_conf/texts/RU --udpipe_path=../models/ru.udpipe --lemmatized_path=../texts_conf/texts/ru_lemmatized.json --keep_pos=0```
```python preprocess_corpus.py --texts_path=../texts_conf/texts/EN --udpipe_path=../models/en.udpipe --lemmatized_path=../texts_conf/texts/en_lemmatized.json --keep_pos=0```

#### Обучение проекции
```python learn_projection.py --src_model_path=../models/ru.bin --tar_model_path=../models/en.bin --bidict_path=../words/ru-en_lem.txt --proj_path=../words/ru-en_proj.txt```

#### Векторизация
* vecmap

```python vectorization_pipeline.py --src_lemmatized_path=../texts_conf/texts/ru_pos_lemmatized.json --tar_lemmatized_path=../texts_conf/texts/en_pos_lemmatized.json --direction=ru-en --method=model --mapping_path=../texts_conf/texts/mapping.json --src_embeddings_path=../models/ru_vecmap.vec --tar_embeddings_path=../models/en_vecmap.vec --src_output_vectors_path=../texts_conf/texts/ru_vecmap.pkl --tar_output_vectors_path=../texts_conf/texts/en_vecmap.pkl --common_output_vectors_path=../texts_conf/texts/common_vecmap.pkl```

* muse

```python vectorization_pipeline.py --src_lemmatized_path=../texts_conf/texts/ru_lemmatized.json --tar_lemmatized_path=../texts_conf/texts/en_lemmatized.json --direction=ru-en --method=model --mapping_path=../texts_conf/texts/mapping.json --src_embeddings_path=../models/ru_muse.vec --tar_embeddings_path=../models/en_muse.vec --src_output_vectors_path=../texts_conf/texts/ru_muse.pkl --tar_output_vectors_path=../texts_conf/texts/en_muse.pkl --common_output_vectors_path=../texts_conf/texts/common_muse.pkl```

* translation

```python vectorization_pipeline.py --src_lemmatized_path=../texts_conf/texts/ru_pos_lemmatized.json --tar_lemmatized_path=../texts_conf/texts/en_pos_lemmatized.json --direction=ru-en --method=translation --mapping_path=../texts_conf/texts/mapping.json --tar_embeddings_path=../models/en.bin --src_output_vectors_path=../texts_conf/texts/ru_trans.pkl --tar_output_vectors_path=../texts_conf/texts/en_trans.pkl --common_output_vectors_path=../texts_conf/texts/common_trans.pkl --bidict_path=../words/ru-en_lem.txt```

* projection

```python vectorization_pipeline.py --src_lemmatized_path=../texts_conf/texts/ru_pos_lemmatized.json --tar_lemmatized_path=../texts_conf/texts/en_pos_lemmatized.json --direction=ru-en --method=projection --mapping_path=../texts_conf/texts/mapping.json --src_embeddings_path=../models/ru.bin --tar_embeddings_path=../models/en.bin --src_output_vectors_path=../texts_conf/texts/ru_projection.pkl --tar_output_vectors_path=../texts_conf/texts/en_projection.pkl --common_output_vectors_path=../texts_conf/texts/common_projection.pkl --projection_path=../words/ru-en_proj.txt```


#### Поиск
* vecmap

```python monocorp_search.py --target_article_path=aist_2012_c6bc0383ea448fcb7e5f45ac85a1afb2d12505ef --lang=cross --mapping_path=../texts_conf/texts/mapping.json --corpus_vectors_path=../texts_conf/texts/common_vecmap.pkl --top=10```

* muse

```python monocorp_search.py --target_article_path=aist_2012_c6bc0383ea448fcb7e5f45ac85a1afb2d12505ef --lang=cross --mapping_path=../texts_conf/texts/mapping.json --corpus_vectors_path=../texts_conf/texts/common_muse.pkl --top=10```

* translation

```python monocorp_search.py --target_article_path=aist_2012_c6bc0383ea448fcb7e5f45ac85a1afb2d12505ef --lang=cross --mapping_path=../texts_conf/texts/mapping.json --corpus_vectors_path=../texts_conf/texts/common_trans.pkl --top=10```

* projection

```python monocorp_search.py --target_article_path=aist_2012_c6bc0383ea448fcb7e5f45ac85a1afb2d12505ef --lang=cross --mapping_path=../texts_conf/texts/mapping.json --corpus_vectors_path=../texts_conf/texts/common_projection.pkl --top=10```


#### Автоматический поиск для списка заголовков
* vecmap

```python automatic_search.py --titles_path=../texts_conf/20ru20en.tsv --mapping_path=../texts_conf/texts/mapping.json --corpus_vectors_path=../texts_conf/texts/common_vecmap.pkl --result_path=../texts_conf/search_results/results_vecmap.txt --top=5```

* muse

```python automatic_search.py --titles_path=../texts_conf/20ru20en.tsv --mapping_path=../texts_conf/texts/mapping.json --corpus_vectors_path=../texts_conf/texts/common_muse.pkl --result_path=../texts_conf/search_results/results_muse.txt --top=5```

* translation

```python automatic_search.py --titles_path=../texts_conf/20ru20en.tsv --mapping_path=../texts_conf/texts/mapping.json --corpus_vectors_path=../texts_conf/texts/common_trans.pkl --result_path=../texts_conf/search_results/results_trans.txt --top=5```

* projection

```python automatic_search.py --titles_path=../texts_conf/20ru20en.tsv --mapping_path=../texts_conf/texts/mapping.json --corpus_vectors_path=../texts_conf/texts/common_projection.pkl --result_path=../texts_conf/search_results/results_projection.txt --top=5```


#### Подсчёт средней оценки и коэффициента согласия аннотаторов (альфа Криппендорфа)
* vecmap

    ```-```
    
* muse
```python eval_score.py --path=converted_eval/muse.csv```

* translation

```python eval_score.py --path=converted_eval/baseline.csv```

* projection
    
    ```-```
    
 #### Конвертация оценок аннотаторов из пятибалльной шкалы в бинарную
* vecmap

    ```-```
    
* muse
```python converter.py --path=eval/muse_eval.csv --save_as=converted_eval/muse.csv --bin=True```

* translation

```python converter.py --path=eval/baseline_eval.csv --save_as=converted_eval/baseline.csv --bin=True```

* projection
    
    ```-```