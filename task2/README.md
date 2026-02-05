# Task 2: Local LLM Integration & Chatbot

This project sets up a local recipe chatbot with fine-tuning, a FastAPI backend, and a CLI client.

**Stack**
1. FastAPI backend (`src/api/main.py`)
2. CLI client (`src/chatbot/cli.py`)
3. Local fine-tuning using LoRA (`scripts/train_lora.py`)

## Project Structure
- `data/raw/` raw recipe data
- `data/processed/` train/val splits and prompt/response data
- `models/` LoRA adapter output
- `scripts/` dataset generation + training
- `src/api/` FastAPI app
- `src/chatbot/` CLI app

## Setup (Linux)
1. Create venv
```bash
python3 -m venv task2-env
```

2. Install runtime deps
```bash
cd -> to the root of project location
pip install -r requirements.txt
```

3. Optional: install training deps( will only require for fine tuning)
```bash
pip install -r requirements-train.txt
```

## Setup (Windows)
1. Create venv
```bat
python -m venv task2-env
```

2. Install runtime deps
```bat
pip install -r requirements.txt
```

3. Optional: install training deps
```bat
pip install -r requirements-train.txt
```

## Environment Variables
Configured in `.env`:
- `API_URL` for CLI
- `HF_BASE_MODEL` and `HF_ADAPTER_PATH` for fine-tuned inference
- `OLLAMA_BASE_URL` and `USE_OLLAMA_FALLBACK`

## Data Preparation
Generate a synthetic recipe dataset and prepare prompt/response pairs:
```bash
python scripts/generate_synthetic_recipes.py
python scripts/prepare_dataset.py
```

Outputs:
- `data/processed/train.jsonl`
- `data/processed/val.jsonl`

## Fine-Tuning (LoRA)
```bash
python scripts/train_lora.py --epochs 2 --max_len 256
```

The LoRA adapter saves to:
- `models/lora-qwen2.5-0.5b`

## Run API
```bash
uvicorn src.api.main:app --reload
```

## Run CLI
```bash
python src/chatbot/cli.py
```

## Validation (Automatic Samples)
```bash
python scripts/validate_model.py
```
This writes results to `data/processed/validation_results.json`.

## Batch Inference (Evaluation)
```bash
python scripts/batch_infer.py --limit 20
```
This writes predictions to `data/processed/val_preds.jsonl`.

## Sample Input / Output
Input:
```
Egg, Onion
```

Output:
```
Recipe: Onion Egg Omelette. Step 1: Beat eggs with salt and pepper. Step 2: Saute chopped onion in a pan with oil until soft. Step 3: Pour eggs over onions and cook until set. Step 4: Fold and serve warm.
```

## Notes
- If `models/lora-qwen2.5-0.5b` exists, FastAPI uses the fine-tuned adapter.
- If not, it falls back to Ollama (if `USE_OLLAMA_FALLBACK=true`).
- API responses are cleaned to return only the recipe text (prompt text is removed).
