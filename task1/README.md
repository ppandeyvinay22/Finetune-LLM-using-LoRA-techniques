# Task 1: Get Matching Person Names

This project builds a name-matching system that finds the most similar names from a dataset when a user inputs a name.

## Quick Start
```bash
source .venv/bin/activate
python src/main.py "Geetha" 5
```

## Interactive Run
```bash
python src/main.py
```
You will be prompted for:
- Name to match
- Top N (how many matches to return)

## Structure
- `src/` application code
- `data/` sample names dataset

## How Matching Works
- **Algorithm:** RapidFuzz uses edit-distance based scoring (e.g., Levenshtein and token-based metrics).
- **Technique:** We use `WRatio`, which combines multiple string similarity strategies for robust matching.
- **Similarity Score:** A numeric score from `0` to `100` where higher is more similar.

## Input Validation
- Extra spaces are trimmed and multiple spaces are collapsed.
- Only alphabets and spaces are allowed. Inputs with characters like `^$!*(&#` are rejected.
* Spaces are preserved for matching, so `Shee la` and `Sheela` can score slightly differently.

## Why Not Cosine Similarity
- Cosine similarity is typically used on vector embeddings for **semantic** similarity.
- This task is about **string similarity** (character/typo differences) for names, not semantic meaning.
- Edit-distance methods are more direct and accurate for short names.

## Output
- Best match with similarity score
- Ranked list of matches
