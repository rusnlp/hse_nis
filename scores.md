# Оценка качества поиска

### Одноязыковой поиск и разные модели udpipe 

| Модель | Топ-1 | Топ-5 | Топ-10 |
| :-------- | :---: | :---: | ---:|
| **Monolang**               |
| [Ru_wiki](http://vectors.nlpl.eu/repository/20/182.zip)  |42.59 |75.93 | 87.04 |
| [Ru_ruscorpora](http://vectors.nlpl.eu/repository/20/180.zip) |35.19 |75.93 | 88.89 |
| [Ru_tayga](http://vectors.nlpl.eu/repository/20/185.zip)  |48.15|72.22 | 88.89 |
| [Ru_news](http://vectors.nlpl.eu/repository/20/184.zip)  |44.44 |72.22 | 87.04 |
| [En](http://vectors.nlpl.eu/repository/20/3.zip) | 46.30 |81.48 | 90.74 |


### Кросс-языковой поиск

| Модель | Топ-1 | Топ-5 | Топ-10 | Ссылки
| :-------- | :---: | :---: | ---:| ---- |
| Vecmap | 38.89 |85.19 | 99.07 | [en](https://yadi.sk/d/gyyLzlOkeiAwXg) [ru](https://yadi.sk/d/EswfSoTkJKJqRA) |
| MUSE | 34.26 |90.74 | 100.00 | [en](https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec) [ru](https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.ru.vec) |
| Перевод (baseline) | 51.85 |87.96 | 95.37 | [en](http://vectors.nlpl.eu/repository/20/3.zip) |
| Проекция | 56.48 | 91.67 | 97.22  | [en](http://vectors.nlpl.eu/repository/20/3.zip) [ru](http://vectors.nlpl.eu/repository/20/182.zip) |
| Laser|  |  |  | 
