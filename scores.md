# Оценка качества поиска (recall)
Оценка производилась на 54 русских и 54 английских статьях википедии с параллельными заголовками по составленному вручную [золотому стандарту](https://github.com/rusnlp/hse_nis/tree/master/texts_wiki/standards). 

Готовые примеры запуска для каждого метода можно найти в [главном readme](https://github.com/rusnlp/hse_nis/blob/master/README.md)

### Одноязыковой поиск и разные модели эмбеддингов 

| Модель | Топ-1 | Топ-5 | Топ-10 |
| :-------- | :---: | :---: | ---:|
| [Ru_wiki](http://vectors.nlpl.eu/repository/20/182.zip)  |42.59 |75.93 | 87.04 |
| [Ru_ruscorpora](http://vectors.nlpl.eu/repository/20/180.zip) |35.19 |75.93 | 88.89 |
| [Ru_tayga](http://vectors.nlpl.eu/repository/20/185.zip)  |48.15|72.22 | 88.89 |
| [Ru_news](http://vectors.nlpl.eu/repository/20/184.zip)  |44.44 |72.22 | 87.04 |
| [En](http://vectors.nlpl.eu/repository/20/3.zip) | 46.30 |81.48 | 90.74 |


### Кросс-языковой поиск (ru -> en)

| Модель | Топ-1 | Топ-5 | Топ-10 | Ссылки
| :-------- | :---: | :---: | ---:| ---- |
| Vecmap | 38.89 |85.19 | 99.07 | [en](https://yadi.sk/d/gyyLzlOkeiAwXg) [ru](https://yadi.sk/d/EswfSoTkJKJqRA) |
| MUSE | 34.26 |90.74 | 100.00 | [en](https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec) [ru](https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.ru.vec) |
| Перевод (baseline) | 51.85 |87.96 | 95.37 | [en](http://vectors.nlpl.eu/repository/20/3.zip) [dict](https://github.com/rusnlp/hse_nis/blob/master/words/ru-en_lem.txt)|
| Проекция | 56.48 | 91.67 | 97.22  | [en](http://vectors.nlpl.eu/repository/20/3.zip) [ru](http://vectors.nlpl.eu/repository/20/182.zip) [dict](https://github.com/rusnlp/hse_nis/blob/master/words/ru-en_lem.txt)|
