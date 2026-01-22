# classify/inclusion_classifier.py
import pandas as pd
import joblib
import torch
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from typing import Literal
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "models"


def run_svm_inclusion(df: pd.DataFrame, model_path: str | None = None) -> pd.DataFrame:
    """
    Add an `inclusion_prob` column (0-1) and an `included` flag (bool) produced by the
    SVM inclusion model. All rows are returned untouched apart from these new columns.
    """
    if model_path is None:
        model_path = BASE_DIR / "inclusion_svm.joblib"

    print("🔍 Loading SVM inclusion model…")
    vectorizer, model = joblib.load(model_path)
    print("✅ Model loaded")

    texts = df["title"].fillna("") + " " + df["abstract"].fillna("")
    X = vectorizer.transform(texts)
    probs = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["inclusion_prob"] = probs
    out["included"] = probs >= 0.5
    return out


def run_bert_inclusion(df: pd.DataFrame, model_dir: str | None = None) -> pd.DataFrame:
    """
    Add an `inclusion_prob` column (0-1) and an `included` flag (bool) produced by the
    BERT inclusion model. All rows are returned untouched apart from these new columns.
    """
    if model_dir is None:
        model_dir = BASE_DIR / "inclusion_bert"

    print("🔍 Loading BERT inclusion model…")
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    model = BertForSequenceClassification.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print("✅ BERT ready")

    probs = []
    texts = df["title"].fillna("") + " " + df["abstract"].fillna("")

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            prob = torch.softmax(outputs.logits, dim=1)[0, 1].item()
            probs.append(prob)

    out = df.copy()
    out["inclusion_prob"] = probs
    out["included"] = out["inclusion_prob"] >= 0.5
    return out


def run_inclusion_model(
    df: pd.DataFrame, model: Literal["svm", "bert"] = "svm"
) -> pd.DataFrame:
    if model == "svm":
        return run_svm_inclusion(df)
    if model == "bert":
        return run_bert_inclusion(df)
    raise ValueError("Unsupported model type. Choose 'svm' or 'bert'.")
