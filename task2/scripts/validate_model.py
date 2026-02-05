#!/usr/bin/env python3
import json
from pathlib import Path

import requests


SAMPLES = [
    "Egg, Onion",
    "Egg, Tomato, Garlic",
    "Potato, Onion, Pepper",
    "Rice, Egg, Peas",
    "Pasta, Garlic, Butter",
]


def main():
    api_url = "http://127.0.0.1:8000/chat"
    results = []
    for text in SAMPLES:
        resp = requests.post(api_url, json={"ingredients": text}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        results.append({"ingredients": text, "recipe": data.get("recipe", "")})

    out_path = Path("data/processed/validation_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
