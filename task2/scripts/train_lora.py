#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_text(example):
    return f"{example['prompt']}\n\nResponse:\n{example['response']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--val", default="data/processed/val.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="models/lora-qwen2.5-0.5b")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    if not train_path.exists() or not val_path.exists():
        raise SystemExit("Missing train/val files. Run prepare_dataset.py first.")

    train_records = load_jsonl(train_path)
    val_records = load_jsonl(val_path)

    train_ds = Dataset.from_list(train_records).map(
        lambda ex: {"text": build_text(ex)}
    )
    val_ds = Dataset.from_list(val_records).map(lambda ex: {"text": build_text(ex)})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_len,
            padding="max_length",
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    trainer.train()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    print(f"Saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
