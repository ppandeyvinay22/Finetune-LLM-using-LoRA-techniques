import os
from pathlib import Path
import re

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

load_dotenv()

app = FastAPI(title="Recipe Chatbot API")

HF_BASE_MODEL = os.getenv("HF_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
HF_ADAPTER_PATH = os.getenv("HF_ADAPTER_PATH", "models/lora-qwen2.5-0.5b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
USE_OLLAMA_FALLBACK = os.getenv("USE_OLLAMA_FALLBACK", "true").lower() == "true"


class LocalModel:
    def __init__(self) -> None:
        self._loaded = False
        self._tokenizer = None
        self._model = None

    def can_load(self) -> bool:
        return Path(HF_ADAPTER_PATH).exists()

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "Transformers/PEFT not installed. Install requirements-train.txt."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            HF_BASE_MODEL,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        model = PeftModel.from_pretrained(base, HF_ADAPTER_PATH)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._loaded = True

    def generate(self, prompt: str) -> str:
        if not self._loaded:
            self.load()
        tokenizer = self._tokenizer
        model = self._model
        import torch

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Decode only newly generated tokens (avoid echoing the prompt).
        gen_tokens = output[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return text


local_model = LocalModel()


class ChatRequest(BaseModel):
    ingredients: str


class ChatResponse(BaseModel):
    recipe: str
    notes: str


def extract_recipe(text: str) -> str:
    if not text:
        return ""
    marker = "Response:"
    if marker in text:
        text = text.split(marker, 1)[-1].strip()
    first = text.find("Recipe:")
    if first != -1:
        text = text[first:]
    # Cut off if the model drifts into other chat-like sections.
    for stop in ["Human:", "User:", "Story:", "###"]:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx].strip()
    # Normalize repeated prefix tokens before deciding if it emitted multiple blocks.
    text = re.sub(r"(?:\bRecipe:\s*){2,}", "Recipe: ", text, flags=re.IGNORECASE).strip()
    # If it repeats "Recipe:" keep only the first block.
    first = text.find("Recipe:")
    if first != -1:
        second = text.find("Recipe:", first + 1)
        if second != -1:
            text = text[:second].strip()
    title, body = _extract_title_and_body(text)
    steps = _extract_meaningful_steps(body)
    if not steps:
        return text.strip()
    return _format_recipe(title, steps)


def _extract_title_and_body(text: str) -> tuple[str, str]:
    text = re.sub(r"(?:\bRecipe:\s*){2,}", "Recipe: ", text, flags=re.IGNORECASE).strip()
    if "Recipe:" not in text:
        return "Recipe", text.strip()
    after = text.split("Recipe:", 1)[1].strip()
    step_match = re.search(r"\bStep\s*\d+\s*[:.)-]", after, flags=re.IGNORECASE)
    if step_match:
        title = after[: step_match.start()].strip(" .:-")
        body = after[step_match.start() :].strip()
        return (title or "Recipe"), body
    title = after.split(".", 1)[0].strip(" .:-")
    return (title or "Recipe"), after


def _extract_meaningful_steps(body: str) -> list[str]:
    numbered = re.findall(
        r"(?:Step\s*)?\d+\s*[:.)-]\s*(.+?)(?=(?:\s+(?:Step\s*)?\d+\s*[:.)-]\s*)|$)",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidates = numbered if numbered else re.split(r"[.!?]\s+", body)
    cleaned: list[str] = []
    seen = set()
    for c in candidates:
        step = " ".join(c.replace("\n", " ").split()).strip(" .,:;-")
        step = re.split(r"\boptional\b|\btips?\b|note:", step, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .,:;-")
        words = step.split()
        if len(words) < 2 or len(words) > 22:
            continue
        if re.fullmatch(r"step\s*\d+\.?", step, flags=re.IGNORECASE):
            continue
        lowered = step.lower()
        if lowered in seen:
            continue
        if any(tok in lowered for tok in ("enjoy", "serve and enjoy")):
            continue
        if lowered in {
            "serve warm",
            "serve immediately",
            "adjust seasoning as needed",
            "adjust salt and pepper as needed",
        }:
            continue
        seen.add(lowered)
        cleaned.append(step)
    if not cleaned:
        return []
    # Keep adaptive complexity but cap long noisy endings.
    if len(cleaned) < 4:
        return cleaned
    return cleaned[:6]


def _format_recipe(title: str, steps: list[str]) -> str:
    rendered_steps = " ".join(f"Step {i+1}: {s}." for i, s in enumerate(steps))
    title = re.sub(r"^(?:Recipe:\s*)+", "", title, flags=re.IGNORECASE).strip(" .:-")
    if not title or title.lower() == "recipe":
        title = "Simple Home-Style Recipe"
    return f"Recipe: {title}. {rendered_steps}".strip()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    prompt = (
        "You are a helpful cooking assistant. "
        f"Given these ingredients: {req.ingredients}. "
        "Return exactly one recipe with a clear title and a concise step list. "
        "Use 4 to 8 steps depending on complexity; keep only meaningful cooking actions. "
        "Do not include optional steps, tips, or serving notes. "
        "Do not output placeholder text like 'Step 2' as a step description. "
        "Do not repeat the word 'Recipe' more than once. "
        "Return only the recipe title and steps."
    )

    if local_model.can_load():
        try:
            text = local_model.generate(prompt)
            return ChatResponse(recipe=extract_recipe(text), notes="local fine-tuned model")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    if USE_OLLAMA_FALLBACK:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": "qwen2.5:0.5b",
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return ChatResponse(
            recipe=extract_recipe(data.get("response", "")),
            notes="ollama fallback",
        )

    recipe = f"Basic omelette with {req.ingredients}".strip()
    return ChatResponse(recipe=recipe, notes="stub response")
