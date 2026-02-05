#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/val.jsonl")
    parser.add_argument("--output", default="data/processed/val_preds.jsonl")
    parser.add_argument("--api", default="http://127.0.0.1:8000/chat")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"Missing input file: {in_path}")

    count = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if count >= args.limit:
                break
            item = json.loads(line)
            prompt = item.get("prompt", "")
            ingredients = prompt.split("Given these ingredients:", 1)[-1].strip()
            resp = requests.post(
                args.api, json={"ingredients": ingredients}, timeout=args.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            fout.write(
                json.dumps(
                    {
                        "ingredients": ingredients,
                        "response": data.get("recipe", ""),
                        "notes": data.get("notes", ""),
                    }
                )
                + "\n"
            )
            count += 1

    print(f"Wrote {count} predictions to {out_path}")


if __name__ == "__main__":
    main()
