#!/usr/bin/env python3
import json
from pathlib import Path


RAW_PATH = Path("data/raw/recipes.jsonl")
OUT_ALL_PATH = Path("data/processed/recipes_instruct.jsonl")
OUT_TRAIN_PATH = Path("data/processed/train.jsonl")
OUT_VAL_PATH = Path("data/processed/val.jsonl")


def normalize_ingredients(items):
    return ", ".join([i.strip().lower() for i in items if i.strip()])


def build_prompt(ingredients):
    return (
        "You are a helpful cooking assistant. "
        f"Given these ingredients: {ingredients}. "
        "Suggest a simple recipe and short steps."
    )


def build_response(title, instructions):
    steps = " ".join([f"Step {i+1}: {s}" for i, s in enumerate(instructions)])
    return f"Recipe: {title}. {steps}"


def main():
    if not RAW_PATH.exists():
        raise SystemExit(f"Missing {RAW_PATH}")

    OUT_ALL_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with RAW_PATH.open("r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            ingredients = normalize_ingredients(item.get("ingredients", []))
            prompt = build_prompt(ingredients)
            response = build_response(
                item.get("title", "Recipe"), item.get("instructions", [])
            )
            records.append({"prompt": prompt, "response": response})

    with OUT_ALL_PATH.open("w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r) + "\n")

    split_idx = max(1, int(len(records) * 0.9))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    with OUT_TRAIN_PATH.open("w", encoding="utf-8") as ftrain:
        for r in train_records:
            ftrain.write(json.dumps(r) + "\n")

    with OUT_VAL_PATH.open("w", encoding="utf-8") as fval:
        for r in val_records:
            fval.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} records to {OUT_ALL_PATH}")
    print(f"Train: {len(train_records)} -> {OUT_TRAIN_PATH}")
    print(f"Val: {len(val_records)} -> {OUT_VAL_PATH}")


if __name__ == "__main__":
    main()
