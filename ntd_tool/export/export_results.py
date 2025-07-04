# export/export_results.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def export_results(
    df: pd.DataFrame,
    used_models: list[str],
    output_path: str = None
) -> str:
    """
    Filters, sorts, and exports predictions to Excel.

    Parameters:
    - df: DataFrame with all predictions and metadata.
    - used_models: list of models used, e.g., ["bert", "svm"]
    - output_path: optional custom output path (default = timestamped file in `exports/`)

    Returns:
    - Full path to saved Excel file.
    """
    assert any(m in ["bert", "svm", "gemini"] for m in used_models), "No valid model specified."

    # ── 1. Inclusion logic: keep if any model says 1 ─────────────────────────
    keep_mask = np.zeros(len(df), dtype=bool)
    for model in used_models:
        col = f"inclusion_prediction_{model}"
        if col in df.columns:
            keep_mask |= (df[col] == 1)
    filtered_df = df[keep_mask].copy()

    if filtered_df.empty:
        print("⚠️ No entries passed the inclusion filter.")
        return None

    # ── 2. Sorting logic: BERT > SVM > Gemini ───────────────────────────────
    priority = [m for m in ["bert", "svm", "gemini"] if m in used_models]
    sort_col = next((f"inclusion_probability_{m}" for m in priority if f"inclusion_probability_{m}" in df.columns), None)
    if sort_col:
        filtered_df = filtered_df.sort_values(by=sort_col, ascending=False)

    # ── 3. Build links ──────────────────────────────────────────────────────
    def build_link(row):
        if pd.notna(row.get("doi")) and isinstance(row["doi"], str) and row["doi"].strip():
            return f"https://doi.org/{row['doi']}"
        elif pd.notna(row.get("title")):
            return f"https://scholar.google.com/scholar?q={row['title']}"
        else:
            return ""

    filtered_df["paper_link"] = filtered_df.apply(build_link, axis=1)

    # ── 4. Column selection ─────────────────────────────────────────────────
    base_cols = [
        "title", "abstract", "authors", "pub_year", "pub_month",
        "doi", "paper_link", "source"
    ]

    prob_cols = [f"inclusion_probability_{m}" for m in used_models if f"inclusion_probability_{m}" in df.columns]
    pred_cols = ["disease_predictions"]

    export_cols = base_cols + prob_cols + pred_cols
    final_df = filtered_df[export_cols]

    # ── 5. Output path ──────────────────────────────────────────────────────
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)

    if output_path is None:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_path = export_dir / f"ntd_results_{date_str}.xlsx"
    else:
        output_path = Path(output_path)

    # ── 6. Save to Excel ────────────────────────────────────────────────────
    final_df.to_excel(output_path, index=False)
    print(f"✅ Exported to {output_path}")
    return str(output_path)
