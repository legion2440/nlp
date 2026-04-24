# NLP

Minimal script-based solution for the `nlp` Piscine AI project.

`## Structure

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
├── fetch_resources.sh
├── nltk_data/
├── requirements.txt
└── run_all.sh
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
bash fetch_resources.sh
```

## Run

```bash
bash run_all.sh
```
