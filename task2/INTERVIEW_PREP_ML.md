# Interview Prep: Data + ML Deep Dive (Task 2)

## 1) Current System Snapshot (What You Built)
- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Fine-tuning: PEFT LoRA on causal LM objective
- Data source: synthetic recipes generated from templates (`500` records)
- Train/val split: deterministic first `90/10` slice (not shuffled)
- Inference: FastAPI endpoint with prompt-based generation + post-processing
- Fallback: Ollama generation when local adapter unavailable

## 2) Model Architecture: How to Explain It in Interview
- Yes, this is a **decoder-only Transformer** setup for generation.
- It uses **causal (masked) self-attention** so each token only attends to previous tokens.
- Your fine-tuning approach (LoRA) inserts low-rank adapters into attention/MLP projection layers while freezing most base weights.
- In your code, LoRA targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.

How to phrase it crisply:
- "I used an instruction-tuned decoder-only LLM and adapted it with LoRA for domain style/control rather than full-model fine-tuning, to keep compute and memory low."

## 3) What Could Be Better (Data + ML)

### Data Pipeline
- Replace synthetic-only data with mixed data (real + synthetic) and clear provenance labels.
- Add data quality checks: duplicates, leakage between train/val, ingredient normalization drift, instruction completeness.
- Shuffle before split; ideally stratify by recipe complexity/ingredient count.
- Add hard examples: sparse ingredients, conflicting ingredients, long-tail cuisines.
- Store dataset versions (hashes, schema version, generation parameters).

### Training Technique
- Use instruction format closer to inference format (reduce train-infer mismatch).
- Mask prompt tokens in loss (train mostly on response tokens) to improve output quality.
- Add small hyperparameter sweeps: LoRA rank, alpha, lr, warmup, sequence length.
- Add early stopping and checkpoint selection by validation metrics.

### Evaluation
- Move beyond manual samples:
- Structural metrics: title presence, step count validity, duplicate step ratio.
- Content metrics: ingredient coverage, actionability score, contradiction detection.
- Pairwise eval: base vs LoRA vs QLoRA outputs judged by rubric.
- Add failure buckets and regression set for repeated checks.

### Inference/Post-processing
- Enforce schema output (JSON with `title`, `steps[]`) then render final text.
- Add robust parsing and fallback repair when schema invalid.
- Add constrained decoding/regex guardrails for step formatting.
- Track latency, token count, and generation failures in logs.

## 4) LoRA vs QLoRA vs Other Options

### LoRA (current)
- Pros: simple, stable, low trainable params, fast iteration.
- Cons: still loads base model in higher precision unless quantized separately.

### QLoRA
- Idea: quantize base model (typically 4-bit) and train LoRA adapters on top.
- Benefit: much lower VRAM/RAM; often similar downstream quality for many tasks.
- Tradeoff: slightly more engineering complexity; quantization config sensitivity.

### DoRA / RS-LoRA
- Variants improving adapter expressiveness or stability in some settings.
- Useful when LoRA plateaus and full fine-tune is too expensive.

### Full Fine-Tuning
- Best adaptation capacity but expensive and harder to serve.
- Usually overkill for this project scope unless data is large/high quality.

## 5) High-Quality Interview Questions (Practice Set)

### A. Data-Centric Questions
1. Why is synthetic-only training risky, and what failure modes did you observe?
2. How would you measure dataset diversity beyond record count?
3. How did you prevent near-duplicates and train/val leakage?
4. If user input ingredients are noisy, where do you normalize: pre-tokenization or prompt layer?
5. How would you design an active-learning loop from production logs?

### B. Training Questions
1. Why LoRA rank `8` and alpha `16`? What would indicate rank is too low/high?
2. Why `max_len=256`? What signals truncation is hurting quality?
3. How do gradient accumulation and batch size trade off on CPU training?
4. Why train on prompt+response together vs response-only loss?
5. What is your checkpoint selection rule and why?

### C. Architecture Questions
1. Explain causal masking and why encoder-decoder was not used here.
2. Which modules did you adapt with LoRA, and why include MLP projections?
3. What are the implications of using a 0.5B model for instruction reliability?
4. How would you reduce hallucinated recipe steps at decoding time?
5. What architecture-level change would you test next: larger base model, MoE, or retrieval augmentation?

### D. Evaluation Questions
1. What does a "good" recipe response mean formally (measurable criteria)?
2. How would you evaluate factuality when recipes are creative?
3. Which automatic metrics are misleading for this task and why?
4. How do you build a regression suite to catch formatting drift?
5. How would you compare base model vs fine-tuned model fairly?

### E. Product/Serving Questions
1. Why fallback to Ollama, and how do you ensure output consistency across backends?
2. How do you bound inference latency and timeout risk?
3. What would you log for observability without leaking PII?
4. How do you handle prompt injection-like instructions from users?
5. If traffic 10x increases, what first 3 scaling actions do you take?

### F. Failure Analysis Questions
1. A recipe has repeated steps; where do you debug first: prompt, decoding, or parser?
2. Output has wrong step count; how do you design a deterministic repair pass?
3. Model ignores key ingredient; how do you detect and penalize this in training/eval?
4. Validation looks good but production bad; what distribution-shift checks do you run?
5. How do you categorize failures into data, model, decoding, and post-processing buckets?

## 6) Strong "Improvements" You Can Say in Interview
- "I would shift to schema-constrained generation and parse/repair before final render."
- "I would introduce data versioning + leakage checks + hard-case benchmark sets."
- "I would run LoRA vs QLoRA ablation with equal eval rubric and latency/cost reporting."
- "I would align train prompt format with inference prompt to reduce instruction mismatch."
- "I would add automated structural metrics in CI so formatting regressions fail fast."

## 7) Short Rapid-Fire Round (One-Liners)
1. Why decoder-only?  
2. Why adapter tuning over full fine-tuning?  
3. Biggest data risk in your setup?  
4. Best quick win for quality?  
5. How do you quantify "meaningful steps"?  
6. If you had 2x budget, where would you spend it first?  
7. How would you productionize eval?  
8. Why might QLoRA outperform/underperform LoRA here?  
9. What metric would you put on dashboard day-1?  
10. What would make you retrain from scratch?
