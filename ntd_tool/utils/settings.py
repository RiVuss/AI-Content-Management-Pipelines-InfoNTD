# ntd_tool/utils/settings.py
"""
Very small JSON key–value store in the user's home directory.
Used for Gemini API key and prompt template.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path.home() / ".ntd_tool_settings.json"
_DEFAULTS = {
    "gemini_api_key": "",
    "gemini_prompt": '''You are an expert in tropical diseases.

Given the following medical research paper's title and abstract, identify which diseases are discussed.

You have the following list of known neglected tropical diseases (NTDs):

['Buruli Ulcer',
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
 'Zika virus']

Use the exact names from the list when applicable.
If the paper discusses diseases not in the list, include them too using their commonly accepted names.

Return your answer as a JSON object in the following format:
{"diseases": ["Disease 1", "Disease 2", ...]}

Here is the publication: 
'''
,
    "gemini_model": "gemini-2.0-flash",
}

def _load_all() -> dict[str, Any]:
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
            except Exception:
                obj = {}
    else:
        obj = {}
    return {**_DEFAULTS, **obj}

def load_setting(key: str) -> Any:
    return _load_all().get(key)

def save_setting(key: str, value: Any) -> None:
    data = _load_all()
    data[key] = value
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
