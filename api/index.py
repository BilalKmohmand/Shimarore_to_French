import os
import sys
import re
import heapq

# Add parent directory to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, request
import pandas as pd
import unicodedata
from openai import OpenAI

TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)


def normalize(text: str) -> str:
    text = (
        unicodedata.normalize("NFD", text.strip().lower())
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.iloc[:, :2]
    df.columns = ["shimaore", "french"]
    df["shimaore_norm"] = df["shimaore"].apply(normalize)
    df["french_norm"] = df["french"].apply(normalize)

    return df


def select_relevant_examples(user_text: str, direction: str, df: pd.DataFrame, max_examples: int = 25) -> str:
    tokens = normalize(user_text).split()
    if not tokens:
        return ""

    if direction == "Shimaore → French":
        source_norm_col = "shimaore_norm"
        source_col = "shimaore"
        target_col = "french"
    else:
        source_norm_col = "french_norm"
        source_col = "french"
        target_col = "shimaore"

    candidates = df[(df[source_col].astype(str).str.len() <= 140) & (df[target_col].astype(str).str.len() <= 140)]
    if candidates.empty:
        candidates = df

    scored = []
    for row in candidates.itertuples(index=False):
        try:
            source_norm = getattr(row, source_norm_col)
        except AttributeError:
            continue
        score = 0
        for t in tokens:
            if t and t in source_norm:
                score += 1
        if score:
            scored.append((score, getattr(row, source_col), getattr(row, target_col)))

    if not scored:
        sample = candidates.head(max_examples)
        return "\n".join(
            f"{source_col.title()}: {r[source_col]} -> {target_col.title()}: {r[target_col]}"
            for _, r in sample.iterrows()
        )

    top = heapq.nlargest(max_examples, scored, key=lambda x: x[0])
    return "\n".join(
        f"{source_col.title()}: {src} -> {target_col.title()}: {tgt}" for _, src, tgt in top
    )


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


def translate_with_ai(text: str, direction: str, df: pd.DataFrame) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    instruction = (
        "Translate the following Shimaore sentence into French."
        if direction == "Shimaore → French"
        else "Translate the following French sentence into Shimaore."
    )

    examples = select_relevant_examples(text, direction, df)

    prompt = f"""You are a translation assistant specializing in Shimaore and French.

Below are example translation pairs between Shimaore and French:

{examples}

IMPORTANT RULES:
1. {instruction}
2. Aim for natural meaning, preserve sentiment and structure.
3. Output ONLY the translated text. No arrows, no original sentence, no labels, no explanation. Just the translation.
"""

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a translation assistant specializing in Shimaore and French."},
            {"role": "user", "content": f"{prompt}\nSentence: {text}"}
        ],
    )

    return response.choices[0].message.content.strip()


# Load dataset at startup
DEFAULT_DATASET_PATH = os.path.join(PROJECT_ROOT, "shimaore_french_dataset.csv")
DATASET_PATH = os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH)
try:
    DF = load_dataset(DATASET_PATH)
except FileNotFoundError:
    DF = None


@app.route("/", methods=["GET"])
def index_get():
    return render_template(
        "index.html",
        direction="Shimaore → French",
        user_input="",
        result=None,
        is_exact=False,
        error=None,
    )


@app.route("/translate", methods=["POST"])
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

        result = translate_with_ai(user_input, direction, DF)
        return render_template(
            "index.html",
            direction=direction,
            user_input=user_input,
            result=result,
            is_exact=False,
            error=None,
        )
    except RuntimeError as e:
        message = str(e)
        if "OPENAI_API_KEY" in message:
            message = "OPENAI_API_KEY is not set on the server. Add it in Vercel → Project → Settings → Environment Variables, then redeploy."
        return render_template(
            "index.html",
            direction=direction,
            user_input=user_input,
            result=None,
            is_exact=False,
            error=message,
        )
    except Exception as e:
        app.logger.exception("Translation request failed")
        safe_details = (str(e) or e.__class__.__name__).strip()
        if len(safe_details) > 240:
            safe_details = safe_details[:240] + "…"
        return render_template(
            "index.html",
            direction=direction,
            user_input=user_input,
            result=None,
            is_exact=False,
            error=f"Something went wrong: {safe_details}",
        )
