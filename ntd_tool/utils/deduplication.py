# utils/deduplication.py
import pandas as pd
import os
import re
from typing import List


def normalize_title(title: str) -> str:
    if not isinstance(title, str):
        return ""
    return re.sub(r"\s+", " ", title.lower().strip())


def load_and_merge_csvs(file_paths: List[str]) -> pd.DataFrame:
    dfs = []
    for path in file_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                source = os.path.basename(path).split("_")[0].lower()
                df["_source"] = source
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {path}: {e}")
    if not dfs:
        raise ValueError("No valid input files to merge.")
    return pd.concat(dfs, ignore_index=True)


def merge_and_deduplicate(inputs: List[str], output_path: str = "temp/merged_deduplicated.csv") -> pd.DataFrame:
    df = load_and_merge_csvs(inputs)
    print(f"🔍 Loaded {len(df)} total entries.")

    # Normalize title for potential future use
    df["_norm_title"] = df["title"].apply(normalize_title) if "title" in df.columns else ""

    # Simple deduplication using DOI if available, else title
    if "doi" in df.columns:
        df = df.drop_duplicates(subset="doi", keep="first")
    elif "title" in df.columns:
        df = df.drop_duplicates(subset="_norm_title", keep="first")
    else:
        df = df.drop_duplicates(keep="first")

    print(f"✅ Deduplicated to {len(df)} unique entries.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"📁 Saved merged dataset to {output_path}")
    return df
