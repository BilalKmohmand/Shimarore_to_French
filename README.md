# Shimaore ↔ French Translator (Flask)

## Setup

1. Put `shimaore_french_dataset.csv` in this folder (same level as `app.py`), or set `DATASET_PATH`.
2. Set your OpenAI key:

```bash
export OPENAI_API_KEY="..."
```

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000
