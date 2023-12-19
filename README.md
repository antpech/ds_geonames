# ds_geonames
# Проект: Сопоставление гео названий с унифицированными именами geonames

**Цель:**

-Сопоставление произвольных гео названий с унифицированными именами geonames для внутреннего использования Карьерным центром


**Основные файлы**
citysearch.py - модуль с классом citysearch, используемый для поиска городов
City Similarity.ipynb - тетрадка Jupyter с описанием и примерами использования

**Дополнительные файлы, который необходимо загрузить**
allCountries.txt
alternateNamesV2.txt
countryInfo.txt
admin1CodesASCII.txt

Загрузить можно по ссылке: https://download.geonames.org/export/dump/

**Используемый стек технологий:**

- Класс для использования на backend: Pyhton
- База данных: Postgresql
- Модели: 
    - Count Vectorizer и 
    - Предъобученная модель Sbert - Labse (https://sbert.net/docs/pretrained_models.html)
    - Предъобученная модель Sbert - Labse-geonames (https://huggingface.co/dima-does-code/LaBSE-geonames-15K-MBML-5e-v1)
    - Поиск по косинусному сравнению


**Может потребоваться для запуска**

- pip install translate
- pip install SQLAlchemy
- pip install -U sentence-transformers

**Данные geonames:**

В проекте были использованы файлы:
- allCountries.txt - файл таблицы geoname. Содержит все данные сервиса geoname
- alternateNamesV2.txt - файл с альтернативными названиям для гео объектов
- admin1CodesASCII.txt - файл с кодами регионального деления
- countryInfo.txt - файл с информацией по странам

В проекте были использованы следующие фильтры:
- страны: ['RU', 'AM', 'AZ', 'BY', 'GE', 'KG', 'KZ', 'MD', 'TJ', 'UA', 'UZ']
- языки: ['ru', 'az', 'en', 'tr', 'uz', 'abbr', 'iata', 'icao', 'faac']
- класс объектов: ['P'] - населенные пункты
- кол-во населения: от 5000 чел.