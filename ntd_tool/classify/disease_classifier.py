# classify/disease_classifier.py
import pandas as pd
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Literal
from pathlib import Path

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


def run_disease_model(df: pd.DataFrame, model: Literal["svm", "bert"] = "svm", threshold: float = 0.5) -> pd.DataFrame:
    if model == "svm":
        return run_svm_disease(df, threshold=threshold)
    elif model == "bert":
        return run_bert_disease(df, threshold=threshold)
    else:
        raise ValueError("Unsupported model type. Choose 'svm' or 'bert'.")
