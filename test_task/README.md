## test_task: Тренировочное задание 

* русская модель udpipe [russian-syntagrus-ud-2.4-190531.udpipe](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2998/russian-syntagrus-ud-2.4-190531.udpipe?sequence=74&isAllowed=y)
* английская модель udpipe [english-partut-ud-2.4-190531.udpipe](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2998/english-partut-ud-2.4-190531.udpipe?sequence=24&isAllowed=y)

* русские эмбеддинги: [ruwikiruscorpora_upos_skipgram_300_2_2019](http://vectors.nlpl.eu/repository/20/182.zip)
* английские эмбеддинги: [300-English Wikipedia Dump of February 2017](http://vectors.nlpl.eu/repository/20/3.zip)

### Одноязычный поиск

**preprocess:** вспомогательный модуль, основанный на [этом скрипте](https://github.com/akutuzov/webvectors/blob/master/preprocessing/rus_preprocessing_udpipe.py)

**preprocess_corpus:** лемматизация корпуса и сохранение его в json

Аргументы:
* обязательные

| Аргумент | Тип | Описание |
| :-------- | :---: | :---------|
|texts_path | str | Папка, в которой лежат тексты в формате txt
|udpipe_path | str | Путь к модели udpipe для обработки корпуса
|lemmatized_path | str | Путь к файлу json, в который будут сохраняться лемматизированные файлы. Если файл уже существует, он будет пополняться

* опциональные

| Аргумент | Тип | Default | Описание |
| :-------- | :---: | :-------: | :-------- |
| keep_pos | pseudo-boolean int | 1 | Возвращать ли леммы, помеченные pos-тегами
| keep_stops| pseudo-boolean int | 0 | Сохранять ли слова, получившие тег функциональной части речи
| keep_punct | pseudo-boolean int | 0 | Сохранять ли знаки препинания
| forced| pseudo-boolean int | 0 | Принудительно лемматизировать весь корпус заново

Примеры запуска:

* ```python preprocess_corpus.py --texts_path=texts/ruwiki --udpipe_path=models/ru.udpipe --lemmatized_path=texts/ruwiki/lemmatized.json```
* ```python preprocess_corpus.py --texts_path=texts/enwiki --udpipe_path=models/en.udpipe --lemmatized_path=texts/enwiki/lemmatized.json  --forced=1```



**vectorize_corpus:** векторизация корпуса и сохранение матрицы нормализованных векторов в pkl

Аргументы:
* обязательные

| Аргумент | Тип | Описание |
| :-------- | :---: | :---------|
| lang | str | Язык, для которго разбираем; нужен для определения словаря в маппинге
| lemmatized_path | str | Путь к файлу json с лемматизированными текстами
| mapping_path | str | Файл маппинга заголовков в индексы и обратно в формате json
| model_embeddings_path | str | Путь к модели для векторизации корпуса
| output_embeddings_path | str | Путь к файлу pkl, в котором будет лежать векторизованный корпус

* опциональные

| Аргумент | Тип | Default | Описание |
| :-------- | :---: | :-------: | :-------- |
| no_duplicates | pseudo-boolean int | 0 | Брать ли для каждого типа в тексте вектор только по одному разу
| forced | pseudo-boolean int | 0 | Принудительно векторизовать весь корпус заново

Примеры запуска:

* ```python vectorize_corpus.py --lang=ru --lemmatized_path=texts/ruwiki/lemmatized.json --mapping_path=texts/titles_mapping.json --model_embeddings_path=models/ru.bin --output_embeddings_path=texts/ruwiki/simple.pkl```
* ```python vectorize_corpus.py --lang=en --lemmatized_path=texts/enwiki/lemmatized.json --mapping_path=texts/titles_mapping.json --model_embeddings_path=models/en.bin --output_embeddings_path=texts/enwiki/simple.pkl```



**monolang_search:** ранжирование статей на основе косинусной близости векторов

Аргументы:
* обязательные

| Аргумент | Тип | Описание |
| :-------- | :---: | :---------|
| target_article_path | str | Путь к статье в формате txt, для которой ищем ближайшие. Если статья из корпуса, то только название без формата
| lang | str  | Язык, для которго разбираем; нужен для определения словаря в маппинге
| mapping_path | str | Файл маппинга заголовков в индексы и обратно в формате json
| corpus_embeddings_path | str | Путь к файлу pkl, в котором лежит векторизованный корпус

* опциональные

| Аргумент | Тип | Default | Описание |
| :-------- | :---: | :-------: | :-------- |
|top | int | 1 | Сколько близких статeй возвращать (-1 for all)
|verbose | pseudo-boolean int | 0 | Принтить ли рейтинг

* для внешних текстов

| Аргумент | Тип | Default | Описание |
| :-------- | :---: | :-------: | :-------- |
| included | pseudo-boolean int | 1 | Включена ли статья в корпус
| udpipe_path | str | '' | Папка, в которой лежит модель udpipe для обработки нового текста
| keep_pos | pseudo-boolean int | 1 | Возвращать ли леммы, помеченные pos-тегами
| keep_stops | pseudo-boolean int | 0 | Сохранять ли слова, получившие тег функциональной части речи
| keep_punct | pseudo-boolean int | 0 | Сохранять ли знаки препинания 
| model_embeddings_path | str | '' | Папка, в которой лежит модель для векторизации корпуса
| no_duplicates | pseudo-boolean int | 0 | Брать ли для каждого типа в тексте вектор только по одному разу

Примеры запуска:

* ```python monolang_search.py --target_article_path=кровь --lang=ru --mapping_path=texts/titles_mapping.json --corpus_embeddings_path=texts/ruwiki/simple.pkl --top=10```
* ```python monolang_search.py --target_article_path=blood --lang=en --mapping_path=texts/titles_mapping.json --corpus_embeddings_path=texts/enwiki/simple.pkl --top=10```

Примеры запуска для сторонней статьи:

* ```python monolang_search.py --target_article_path=texts/ruwiki/бензин.txt --lang=ru --mapping_path=texts/titles_mapping.json --corpus_embeddings_path=texts/ruwiki/simple.pkl --top=10 --included=0 --udpipe_path=models/ru.udpipe --model_embeddings_path=models/ru.bin```
* ```python monolang_search.py --target_article_path=texts/enwiki/gasoline.txt --lang=en --mapping_path=texts/titles_mapping.json --corpus_embeddings_path=texts/enwiki/simple.pkl --top=10 --included=0 --udpipe_path=models/en.udpipe --model_embeddings_path=models/en.bin```


**evaluate_corpus:** оценка качества поиска: проверка среди 1, 5, 10 ближайших

Аргументы:

* обязательные

| Аргумент | Тип | Описание |
| :-------- | :---: | :---------|
| lang | str | Язык, для которого разбираем; нужен для определения словаря в маппинге
| mapping_path | str | Файл маппинга заголовков в индексы и обратно в формате json
| corpus_embeddings_path | str | Путь к файлу pkl, в котором лежит векторизованный корпус
| golden_standard_path | str | Файл с парами наиболее близких статей

Примеры запуска:

* ```python evaluate_corpus.py --lang=ru --corpus_embeddings_path=texts/ruwiki/simple.pkl --mapping_path=texts/titles_mapping.json --golden_standard_path=texts/ru_similar_titles.txt```
* ```python evaluate_corpus.py --lang=en --corpus_embeddings_path=texts/enwiki/simple.pkl --mapping_path=texts/titles_mapping.json --golden_standard_path=texts/en_similar_titles.txt```



____________________________________________
### Кроссязыковой поиск