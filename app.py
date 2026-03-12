import os
import unicodedata

import pandas as pd
from flask import Flask, render_template, request
from openai import OpenAI

app = Flask(__name__)


def normalize(text: str) -> str:
    return (
        unicodedata.normalize("NFD", text.strip().lower())
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.iloc[:, :2]
    df.columns = ["shimaore", "french"]
    df["shimaore_norm"] = df["shimaore"].apply(normalize)
    df["french_norm"] = df["french"].apply(normalize)

    examples = "\n".join(
        f'Shimaore: {row["shimaore"]} -> French: {row["french"]}'
        for _, row in df.iterrows()
    )
    return df, examples


def exact_match(text: str, direction: str, df: pd.DataFrame):
    key = normalize(text)
    if direction == "Shimaore → French":
        row = df[df["shimaore_norm"] == key]
        if not row.empty:
            return row.iloc[0]["french"]
    else:
        row = df[df["french_norm"] == key]
        if not row.empty:
            return row.iloc[0]["shimaore"]
    return None


def translate_with_ai(text: str, direction: str, examples: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    instruction = (
        "Translate the following Shimaore sentence into French."
        if direction == "Shimaore → French"
        else "Translate the following French sentence into Shimaore."
    )

    prompt = f"""You are a translation assistant specializing in Shimaore and French.

Below is the COMPLETE translation dataset between Shimaore and French:

{examples}

IMPORTANT RULES:
1. First, check if the sentence exists EXACTLY in the dataset above.
   - If found: return that EXACT translation, nothing else.
2. If the sentence is NOT in the dataset:
   - {instruction}
   - Aim for natural meaning, preserve sentiment and structure.
3. Output ONLY the translated text. No arrows, no original sentence, no labels, no explanation. Just the translation.
"""

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"{prompt}\nSentence: {text}",
    )

    return response.output_text.strip()


DATASET_PATH = os.getenv("DATASET_PATH", "shimaore_french_dataset.csv")
try:
    DF, EXAMPLES = load_dataset(DATASET_PATH)
except FileNotFoundError:
    DF, EXAMPLES = None, ""


@app.get("/")
def index_get():
    return render_template(
        "index.html",
        direction="Shimaore → French",
        user_input="",
        result=None,
        is_exact=False,
        error=None,
    )


@app.post("/translate")
def translate_post():
    direction = request.form.get("direction", "Shimaore → French")
    user_input = (request.form.get("user_input") or "").strip()

    if not user_input:
        return render_template(
            "index.html",
            direction=direction,
            user_input=user_input,
            result=None,
            is_exact=False,
            error="Please enter some text to translate.",
        )

    if DF is None:
        return render_template(
            "index.html",
            direction=direction,
            user_input=user_input,
            result=None,
            is_exact=False,
            error="shimaore_french_dataset.csv not found. Place it in the project folder (or set DATASET_PATH).",
        )

    try:
        exact = exact_match(user_input, direction, DF)
        if exact:
            return render_template(
                "index.html",
                direction=direction,
                user_input=user_input,
                result=exact,
                is_exact=True,
                error=None,
            )

        result = translate_with_ai(user_input, direction, EXAMPLES)
        return render_template(
            "index.html",
            direction=direction,
            user_input=user_input,
            result=result,
            is_exact=False,
            error=None,
        )
    except Exception:
        return render_template(
            "index.html",
            direction=direction,
            user_input=user_input,
            result=None,
            is_exact=False,
            error="Something went wrong. Please try again in a moment.",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
