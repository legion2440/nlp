# nlp

## О проекте
Практический репозиторий по базовой NLP-предобработке текста: от проверки окружения, lowercase/uppercase и удаления пунктуации до токенизации, stop words, stemming и построения Bag of Words представления для размеченных твитов.

В репозитории собраны упражнения `ex00`-`ex07`, которые последовательно показывают:
- проверку Python/NLP-окружения и зависимостей
- работу со строковыми преобразованиями в `pandas`
- удаление пунктуации из текста
- токенизацию текста через NLTK
- фильтрацию stop words
- stemming через `PorterStemmer`
- сборку общей функции preprocessing
- построение word count matrix через `CountVectorizer(max_features=500)`
- создание sparse DataFrame и добавление label из исходного датасета

## Структура
```text
nlp/
├── ex00/
│   └── check_env.py
├── ex01/
│   └── answer.py
├── ex02/
│   └── answer.py
├── ex03/
│   └── answer.py
├── ex04/
│   └── answer.py
├── ex05/
│   └── answer.py
├── ex06/
│   ├── __init__.py
│   └── answer.py
├── ex07/
│   └── answer.py
├── resources/
│   └── tweets_train.txt
├── nlp_utils.py
├── requirements.txt
└── README.md
```

## Зависимости
Основные зависимости:
- Python 3.9+
- jupyter
- nltk
- pandas
- scikit-learn
- tabulate

## Подготовка окружения
### Linux / WSL / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Git Bash на Windows
```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

## Данные
Для `ex07` используется файл:

```text
resources/tweets_train.txt
```

Файл должен лежать в `resources/` до запуска `ex07`.

NLTK-ресурсы проверяются и загружаются через `nlp_utils.py` при запуске упражнений, где используется NLTK.

## Запуск
### Проверка окружения
```bash
python ex00/check_env.py
```

### Exercises
```bash
python ex01/answer.py
python ex02/answer.py
python ex03/answer.py
python ex04/answer.py
python ex05/answer.py
python ex06/answer.py
python ex07/answer.py
```

## Что делает каждый exercise
### ex00
Проверяет:
- версию Python
- доступность библиотек `jupyter`, `nltk`, `pandas`, `sklearn`, `tabulate`
- доступность нужных NLTK-ресурсов

### ex01
Показывает базовые строковые преобразования в `pandas.Series`:
- перевод текста в lowercase
- перевод текста в uppercase

### ex02
Удаляет punctuation из заданной строки через `str.translate()` и `string.punctuation`.

### ex03
Показывает tokenization через NLTK:
- разбиение текста на предложения через `sent_tokenize`
- разбиение текста на слова и знаки через `word_tokenize`

### ex04
Удаляет stop words из текста:
- загружает английские stop words из NLTK
- токенизирует текст
- выводит список токенов без stop words

### ex05
Показывает stemming через NLTK:
- токенизирует текст
- применяет `PorterStemmer`
- выводит stemmed tokens

### ex06
Собирает общий preprocessing pipeline в функции `preprocess_text()`:
- lowercase
- removing punctuation
- tokenization
- stopword filtering
- stemming

В удаление пунктуации дополнительно включён Unicode ellipsis `…`, потому что он встречается в датасете твитов и иначе попадает в vocabulary как отдельный токен.

### ex07
Строит Bag of Words представление для `resources/tweets_train.txt`:
- читает labels и tweets из исходного файла
- применяет `preprocess_text()` из `ex06`
- строит `CountVectorizer(max_features=500)`
- создаёт sparse DataFrame через `pd.DataFrame.sparse.from_spmatrix(...)`
- выводит shape word count matrix
- выводит audit-релевантные срезы DataFrame
- показывает token counts для 4-го твита
- показывает top-15 самых частых токенов
- добавляет колонку `label` на основе исходных labels:
  - `positive` -> `1`
  - `neutral` -> `0`
  - `negative` -> `-1`

## Важные замечания
### Различия вывода между версиями библиотек
Точный вид некоторых выводов может отличаться между версиями `nltk`, `pandas` и `scikit-learn`.

Особенно это касается:
- форматирования sparse matrix
- форматирования `DataFrame.to_markdown()`
- порядка колонок в `CountVectorizer(max_features=500)`, если несколько токенов имеют одинаковую частоту около границы top-500

Это не обязательно означает ошибку в решении, если:
- shape word count matrix остаётся `(6588, 500)`
- данные берутся из `resources/tweets_train.txt`
- preprocessing выполняется в заданном порядке
- labels добавляются из исходного файла, а не хардкодятся
- token counts и top tokens считаются из реального DataFrame

### Про `ex07`
`CountVectorizer(max_features=500)` выбирает признаки по частоте в корпусе. Если на границе отбора есть несколько токенов с одинаковой частотой, точные позиции колонок могут немного отличаться между версиями библиотек.

Поэтому демонстрационный срез `iloc[:3, 400:403]` может показывать отличающиеся соседние токены, при сохранении корректной BoW-логики.

### Про `nlp_utils.py`
`nlp_utils.py` содержит общую вспомогательную логику:
- путь к датасету
- загрузку labels и tweets
- mapping labels в числовые значения
- проверку и загрузку NLTK-ресурсов

Это вспомогательный модуль для повторного использования общей логики, а не отдельное упражнение.

## TOC
- [О проекте](#о-проекте)
- [Структура](#структура)
- [Зависимости](#зависимости)
- [Подготовка окружения](#подготовка-окружения)
- [Данные](#данные)
- [Запуск](#запуск)
- [Что делает каждый exercise](#что-делает-каждый-exercise)
- [Важные замечания](#важные-замечания)

## Автор
- Nazar Yestayev (@nyestaye / @legion2440)
