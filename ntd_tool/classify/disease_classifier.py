# classify/disease_classifier.py
import pandas as pd
import joblib
import torch
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from typing import Literal
from pathlib import Path
import json
import os
import google.generativeai as genai
from google.generativeai import types


BASE_DIR = Path(__file__).resolve().parent.parent / "models"

DISEASE_LIST = [
    'Buruli Ulcer',
    'Chagas disease (American trypanosomiasis)',
    'Chromoblastomycosis',
    'Dracunculiasis (guinea-worm disease)',
    'Human African trypanosomiasis (HAT) - Sleeping sickness',
    'Leishmaniasis',
    'Leprosy',
    'Lymphatic filariasis (Elephantiasis)',
    'Mycetoma',
    'Noma',
    'Onchocerciasis',
    'Podoconiosis',
    'Scabies',
    'Schistosomiasis',
    'Snakebite envenoming',
    'Soil-transmitted helminths',
    'Trachoma',
    'Yaws',
    'Zika virus'
]


def run_svm_disease(df: pd.DataFrame, threshold: float = 0.5, model_path: str = None) -> pd.DataFrame:
    if model_path is None:
        model_path = BASE_DIR / "disease_svm.joblib"

    print("🔍 Loading SVM disease model…")
    vectorizer, clf, disease_cols = joblib.load(model_path)
    print("✅ Model loaded")

    texts = (df["title"].fillna("") + " " + df["abstract"].fillna(""))
    X = vectorizer.transform(texts)

    probs = [est.predict_proba(X)[:, 1] for est in clf.estimators_]
    pred_df = pd.DataFrame({f"svm_prob_{disease}": col for disease, col in zip(disease_cols, probs)})
    pred_df["disease_predictions"] = pred_df.apply(
        lambda row: [disease for disease in disease_cols if row[f"svm_prob_{disease}"] >= threshold], axis=1
    )

    return pd.concat([df.reset_index(drop=True), pred_df], axis=1)


def run_bert_disease(df: pd.DataFrame, threshold: float = 0.5, model_dir: str = None) -> pd.DataFrame:
    if model_dir is None:
        model_dir = BASE_DIR / "disease_bert"

    print("🔍 Loading BERT disease model…")
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print("✅ BERT model ready")

    disease_cols = DISEASE_LIST

    texts = (df["title"].fillna("") + " " + df["abstract"].fillna(""))
    results = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            results.append(probs)

    results_df = pd.DataFrame(results, columns=[f"bert_prob_{d}" for d in disease_cols])
    results_df["disease_predictions"] = results_df.apply(
        lambda row: [disease for i, disease in enumerate(disease_cols) if row[f"bert_prob_{disease_cols[i]}"] >= threshold], axis=1
    )

    return pd.concat([df.reset_index(drop=True), results_df], axis=1)

# ────────── GEMINI helpers (new API) ──────────
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json, time, re
from tenacity import retry, wait_random_exponential, stop_after_attempt

def _make_prompt(prompt_template: str, title: str, abstract: str) -> str:
    return f"{prompt_template}{title}\n\n{abstract}"


@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(5))
def _query_gemini(model, prompt: str) -> str:
    """Call Gemini and return plain text."""
    resp = model.generate_content(prompt)
    return resp.text.strip()

def _extract_diseases(resp_text: str) -> list[str]:
    m = re.search(r'\{.*\}', resp_text, re.S)
    if not m:
        return []
    try:
        blob = json.loads(m.group())
        return [d.strip() for d in blob.get("diseases", []) if isinstance(d, str)]
    except Exception:
        return []

def run_gemini_disease(
    df: pd.DataFrame,
    api_key: str,
    prompt: str,
    gemini_model: str = "gemini-2.5-flash",
    max_workers: int = 4,
    throttle_qps: float = 1.0,
) -> pd.DataFrame:
    """Row-by-row Gemini disease tagging using the *new* SDK."""
    if not api_key:
        raise ValueError("Gemini API-key missing")

    print(f"🔑 Using Gemini model: {gemini_model}")
    genai.configure(api_key=api_key)                    # ← NEW
    model = genai.GenerativeModel(gemini_model)         # ← NEW

    results = []
    def _worker(row):
        title = row.get("title", "") or ""
        abstract = row.get("abstract", "") or ""
        p = _make_prompt(prompt, title=title, abstract=abstract)
        text = _query_gemini(model, p)
        return _extract_diseases(text)

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        fut_to_idx = {exe.submit(_worker, r): i for i, r in df.iterrows()}
        for fut in tqdm(as_completed(fut_to_idx), total=len(df), desc="Gemini"):
            idx = fut_to_idx[fut]
            try:
                preds = fut.result()
            except Exception as e:
                preds = []
                print(f"❌ Gemini error on row {idx}: {e}")
            results.append((idx, preds))
            time.sleep(1.0 / throttle_qps)

    pred_series = pd.Series({idx: preds for idx, preds in results},
                            name="disease_predictions")
    df_out = df.copy()
    df_out["disease_predictions"] = pred_series
    return df_out.reset_index(drop=True)

def run_disease_model(df: pd.DataFrame, model: Literal["svm", "bert", "gemini"] = "svm", **kw) -> pd.DataFrame:
    if model == "svm":
        return run_svm_disease(df, threshold=kw.get("threshold", 0.5))
    elif model == "bert":
        return run_bert_disease(df, threshold=kw.get("threshold", 0.5))
    elif model == "gemini":
        return run_gemini_disease(
            df,
            api_key=kw["api_key"],
            prompt=kw["prompt"] or "",
            gemini_model=kw.get("gemini_model", "gemini-2.5-flash"),
            max_workers=kw.get("max_workers", 4),
            throttle_qps=kw.get("throttle_qps", 1.0),
        )
    else:
        raise ValueError("Unsupported model type.")
