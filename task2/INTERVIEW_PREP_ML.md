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

## 8) Answer Bank (1-2 Strong Answers Per Question)

### A. Data-Centric Questions
1. Why is synthetic-only training risky, and what failure modes did you observe?  
Answer Option 1: Synthetic data tends to collapse language diversity, so the model learns template style instead of robust reasoning. In this project, that can show up as repeated generic steps and weak endings.  
Answer Option 2: Synthetic generators encode their own bias/errors. If generation templates include artifacts, those become “ground truth” and the fine-tuned model amplifies them in production.

2. How would you measure dataset diversity beyond record count?  
Answer Option 1: Track unique n-grams for actions and ingredient combinations, plus entropy of recipe titles and step verbs. Low entropy is a red flag for repetitive templates.  
Answer Option 2: Use embedding clustering and report cluster balance. If most samples fall into a few dense clusters, coverage is narrow even with many rows.

3. How did you prevent near-duplicates and train/val leakage?  
Answer Option 1: Canonicalize ingredients + normalize steps, hash examples, and remove exact/near duplicates before split. Then split by grouped hash rather than raw row order.  
Answer Option 2: Do similarity-based dedup (e.g., Jaccard/embedding threshold) and verify overlap report between train/val; any high-similarity pair across splits is leakage risk.

4. If user input ingredients are noisy, where do you normalize: pre-tokenization or prompt layer?  
Answer Option 1: Normalize before prompt construction so both training and inference share one canonical representation. That reduces prompt variance and improves model consistency.  
Answer Option 2: Keep raw input for traceability but pass normalized form to generation. This gives auditability while preserving model stability.

5. How would you design an active-learning loop from production logs?  
Answer Option 1: Capture failures by rule checks (bad step count, repeated steps, ingredient miss), sample those cases for human correction, and retrain periodically.  
Answer Option 2: Add user feedback signals (thumbs up/down), prioritize high-disagreement or low-confidence outputs, and build a hard-case replay set for each release.

### B. Training Questions
1. Why LoRA rank `8` and alpha `16`? What would indicate rank is too low/high?  
Answer Option 1: Rank 8/alpha 16 is a compute-efficient baseline for small-domain adaptation. Too-low rank shows underfitting (generic outputs), too-high shows unstable or overfit style copying.  
Answer Option 2: I start with literature/common defaults, then tune by validation quality and repetition metrics. Rank is too high if gains plateau while variance and memory cost rise.

2. Why `max_len=256`? What signals truncation is hurting quality?  
Answer Option 1: 256 keeps CPU fine-tuning practical for this task size. If truncation hurts, you’ll see cut-off responses, missing later steps, and dropped instruction suffixes.  
Answer Option 2: I’d verify by logging truncation rate during tokenization. If a meaningful fraction clips, increase length or shorten prompt format.

3. How do gradient accumulation and batch size trade off on CPU training?  
Answer Option 1: Small per-device batch with accumulation simulates larger effective batch under memory limits. Tradeoff is slower wall-clock and sometimes noisier optimization dynamics.  
Answer Option 2: On CPU, accumulation is often required for stability. I tune it with learning rate because effective batch size and LR should be co-calibrated.

4. Why train on prompt+response together vs response-only loss?  
Answer Option 1: Prompt+response is simpler to implement but may waste capacity predicting prompt text. Response-only masking usually improves instruction-following quality.  
Answer Option 2: For fast prototype, full-sequence loss is acceptable; for production quality, I’d mask prompt tokens to focus gradient where behavior matters.

5. What is your checkpoint selection rule and why?  
Answer Option 1: Select checkpoint by validation task metrics (format compliance + ingredient coverage + repetition penalty), not only loss. Loss can improve while UX worsens.  
Answer Option 2: Use Pareto choice: highest structure compliance under latency budget. This ties model selection directly to product constraints.

### C. Architecture Questions
1. Explain causal masking and why encoder-decoder was not used here.  
Answer Option 1: Causal masking ensures each token attends only to prior tokens, enabling left-to-right generation. Decoder-only was chosen for simplicity and compatibility with instruction LLM fine-tuning.  
Answer Option 2: Encoder-decoder can be stronger for strict input-output mapping, but decoder-only has easier local serving ecosystem and lower integration complexity here.

2. Which modules did you adapt with LoRA, and why include MLP projections?  
Answer Option 1: I adapted attention and MLP projections (`q/k/v/o` plus `up/down/gate`). Attention controls token interaction; MLP layers help domain style and response composition.  
Answer Option 2: Restricting only attention is cheaper, but adding MLP projections often improves controllability on generation tasks with structured output.

3. What are the implications of using a 0.5B model for instruction reliability?  
Answer Option 1: 0.5B is cost-efficient and fast but less robust for strict formatting and long-context consistency. You need stronger post-processing guardrails.  
Answer Option 2: Small models are good for local prototype latency, but quality ceiling is lower. If error rate matters, scale model or add constrained decoding.

4. How would you reduce hallucinated recipe steps at decoding time?  
Answer Option 1: Lower temperature/top-p, add stop sequences, and enforce schema extraction with repair pass. This reduces creative drift.  
Answer Option 2: Add ingredient-coverage checks and reject/re-generate if key ingredients are missing or unsupported actions appear.

5. What architecture-level change would you test next: larger base model, MoE, or retrieval augmentation?  
Answer Option 1: Larger base model first; it is lowest integration risk and typically gives immediate quality gains for instruction adherence.  
Answer Option 2: Retrieval augmentation if factual grounding becomes important (nutrition, cuisine constraints). MoE is usually overkill for this scope.

### D. Evaluation Questions
1. What does a "good" recipe response mean formally (measurable criteria)?  
Answer Option 1: Valid format (single title + numbered steps), no duplicates, 4-8 actionable steps, and ingredient alignment score above threshold.  
Answer Option 2: Combine structural pass/fail with human preference scores for usefulness/taste plausibility; optimize both.

2. How would you evaluate factuality when recipes are creative?  
Answer Option 1: Evaluate feasibility instead of strict factuality: step order validity, ingredient-action compatibility, and food-safety constraints.  
Answer Option 2: Use rubric-based expert review for sampled outputs; objective checks catch syntax, human checks catch culinary realism.

3. Which automatic metrics are misleading for this task and why?  
Answer Option 1: BLEU/ROUGE can reward lexical overlap with references and miss whether steps are actually useful or coherent.  
Answer Option 2: Perplexity/loss alone may improve while output structure worsens; task-specific structural metrics are more diagnostic.

4. How do you build a regression suite to catch formatting drift?  
Answer Option 1: Freeze a fixed prompt set with expected schema constraints and run on every model/prompt change in CI.  
Answer Option 2: Include adversarial prompts (ambiguous ingredients, long lists, junk text) and assert deterministic post-processing behavior.

5. How would you compare base model vs fine-tuned model fairly?  
Answer Option 1: Same prompt template, same decoding params, same test set, blind side-by-side judging with fixed rubric.  
Answer Option 2: Report both quality and latency/resource usage so the tradeoff is explicit, not just output preference.

### E. Product/Serving Questions
1. Why fallback to Ollama, and how do you ensure output consistency across backends?  
Answer Option 1: Fallback improves availability when adapter load fails. Consistency comes from shared prompt contract + shared parser/repair logic.  
Answer Option 2: Keep backend-specific generation settings versioned and test both in the same eval suite.

2. How do you bound inference latency and timeout risk?  
Answer Option 1: Cap max tokens, tune decoding, and enforce API timeout with clear fallback path.  
Answer Option 2: Add cold-start warmup and cache frequent ingredient patterns to reduce p95 latency.

3. What would you log for observability without leaking PII?  
Answer Option 1: Log request IDs, latency, model version, decoding params, parser error codes, and hashed/normalized ingredient stats.  
Answer Option 2: Avoid raw user text retention by default; store sampled redacted prompts only for debugging with strict retention policy.

4. How do you handle prompt injection-like instructions from users?  
Answer Option 1: Treat user content as data only, keep system prompt authority, and strip unsupported meta-instructions pre-generation.  
Answer Option 2: Post-validate output schema and safety rules; reject/regenerate when instructions attempt out-of-scope behavior.

5. If traffic 10x increases, what first 3 scaling actions do you take?  
Answer Option 1: Add request queue + worker pooling, enable horizontal API scaling, and tune generation length to control compute.  
Answer Option 2: Introduce caching for repeated ingredient sets, add circuit breaker for backend failures, and monitor p95/p99 latency dashboards.

### F. Failure Analysis Questions
1. A recipe has repeated steps; where do you debug first: prompt, decoding, or parser?  
Answer Option 1: Start with parser because it’s deterministic and quickest to verify. If parser is clean, inspect decoding temperature and repetition controls.  
Answer Option 2: If repeats are semantic rather than exact text, inspect training data duplication and template diversity.

2. Output has wrong step count; how do you design a deterministic repair pass?  
Answer Option 1: Parse numbered steps, dedupe, remove malformed/filler steps, then renumber to allowed range.  
Answer Option 2: If repaired output falls below minimum quality threshold, trigger one controlled regeneration with stricter decoding.

3. Model ignores key ingredient; how do you detect and penalize this in training/eval?  
Answer Option 1: Add ingredient coverage metric (required ingredient mentions/usage) and treat low coverage as failure.  
Answer Option 2: Add contrastive training examples where omission is explicitly marked incorrect to teach dependency.

4. Validation looks good but production bad; what distribution-shift checks do you run?  
Answer Option 1: Compare train/val vs production distributions: ingredient counts, rare ingredients, typo rates, language style.  
Answer Option 2: Build shadow evaluation from real logs and track drift over time with alert thresholds.

5. How do you categorize failures into data, model, decoding, and post-processing buckets?  
Answer Option 1: Define taxonomy with deterministic tags (e.g., parse_fail, repetition, ingredient_miss, hallucination) and map each to owner layer.  
Answer Option 2: Run ablation debugging: same prompt with different decoding and parser variants; differences localize the faulty stage.

### G. Rapid-Fire One-Liners
1. Why decoder-only?  
Answer Option 1: Simpler serving/training path for generative chat tasks.  
Answer Option 2: Strong ecosystem support and fast iteration with PEFT adapters.

2. Why adapter tuning over full fine-tuning?  
Answer Option 1: Much cheaper compute/memory with good domain adaptation.  
Answer Option 2: Easier versioning and rollback of small adapter artifacts.

3. Biggest data risk in your setup?  
Answer Option 1: Synthetic template bias causing repetitive outputs.  
Answer Option 2: Non-random split causing optimistic validation.

4. Best quick win for quality?  
Answer Option 1: Response-only loss masking + better evaluation gates.  
Answer Option 2: Stronger post-processing schema enforcement.

5. How do you quantify "meaningful steps"?  
Answer Option 1: Actionable verb presence, non-duplicate, ingredient-linked, no filler/tips.  
Answer Option 2: Pass/fail rules plus human preference sampling.

6. If you had 2x budget, where would you spend it first?  
Answer Option 1: Higher quality mixed real dataset and labeling.  
Answer Option 2: Model size bump plus systematic hyperparameter sweep.

7. How would you productionize eval?  
Answer Option 1: CI regression prompts with structural assertions.  
Answer Option 2: Online canary eval + weekly human review panel.

8. Why might QLoRA outperform/underperform LoRA here?  
Answer Option 1: Outperform operationally by enabling larger effective models in same memory.  
Answer Option 2: Underperform if quantization config is suboptimal for this narrow task/data scale.

9. What metric would you put on dashboard day-1?  
Answer Option 1: Valid-response rate (schema/format pass).  
Answer Option 2: Repetition/error rate tied to user-visible failures.

10. What would make you retrain from scratch?  
Answer Option 1: Major domain shift + persistent failures not fixable by adapters.  
Answer Option 2: New high-quality dataset where base model prior is misaligned with product goals.
