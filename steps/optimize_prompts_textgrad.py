"""
optimize_prompts_textgrad.py — TextGrad + Gemini edition
=========================================================
Alternatywna implementacja optymalizacji promptu domenowego
używająca TextGrad zamiast DSPy/SIMBA.

TextGrad nie wspiera Gemini natywnie — używamy custom engine
który owija LiteLLM. Dzięki temu zachowujemy provider-agnostic
layer który mamy w reszcie pipeline'u.

Branch: experiment
Cel: porównanie podejść — TextGrad vs SIMBA

TextGrad vs SIMBA — kluczowe różnice:
  SIMBA (DSPy):
    - Stochastic mini-batch ascent
    - Optymalizuje przez próbkowanie trudnych przykładów
    - Strukturyzowane I/O przez DSPy Signature
    - Lepszy przy małych datasetach (15-30 przykładów)
    - Tańszy — mniej wywołań LLM per krok

  TextGrad:
    - Automatyczna "dyferencjacja" przez tekst
    - Backpropagation przez textual feedback (LLM jako gradient)
    - Elastyczniejszy — optymalizuje dowolną zmienną tekstową
    - Lepszy gdy masz złożony pipeline i chcesz propagować
      feedback z końca łańcucha wstecz do wcześniejszych komponentów

Dla naszego use case'u (jeden prompt, mały dataset):
  SIMBA jest prawdopodobnie lepszy.
  TextGrad jest tu przede wszystkim dla nauki i porównania.

Użycie:
  export GEMINI_API_KEY="twój_klucz"   # lub LLM_PROVIDER=ollama / mlx (patrz .env.example)
  uv run steps/optimize_prompts_textgrad.py

Wynik:
  artifacts/prompts/extractor_optimized_textgrad.json
  Porównanie ze SIMBA/GEPA: uv run compare-optimizers
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "textgrad",
#     "litellm>=1.56",
#     "python-dotenv>=1.0",
# ]
# ///

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import litellm
import textgrad as tg
from textgrad.engine import EngineLM
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.extraction_metric import feedback_text, parse_extraction, score_against_expected

load_dotenv()

ROOT         = Path(__file__).parent.parent
TRAINSET_DIR = ROOT / "artifacts" / "trainset"
PROMPTS_DIR  = ROOT / "artifacts" / "prompts"

# Provider-agnostic model config, matching optimize_prompts.py/generate_digests.py.
_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
if _PROVIDER == "ollama":
    _DEFAULT_MODEL = "ollama/qwen3.6:35b-a3b-q4_K_M"
    _LM_KWARGS     = {"api_base": "http://localhost:11434"}
elif _PROVIDER == "mlx":
    _DEFAULT_MODEL = "openai/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
    _LM_KWARGS     = {
        "api_base": "http://localhost:8888/v1",
        "api_key":  os.environ.get("LLM_API_KEY", "fake"),
    }
else:
    _DEFAULT_MODEL = "gemini/gemini-2.5-flash"
    _LM_KWARGS     = {}
FORWARD_MODEL = os.environ.get("MODEL") or _DEFAULT_MODEL   # tańszy — wykonuje zadanie

# Reflection/backward (gradient) model — defaults to a stronger tier of the
# same provider. ollama/mlx have no cheap/pro split locally, so they reuse
# FORWARD_MODEL unless REFLECTION_MODEL is set explicitly.
BACKWARD_MODEL = os.environ.get("REFLECTION_MODEL") or (
    "gemini/gemini-2.5-pro" if _PROVIDER == "gemini" else FORWARD_MODEL
)

NUM_EPOCHS  = 3
BATCH_SIZE  = 4

litellm.set_verbose = False


# ── Custom Gemini engine dla TextGrad ─────────────────────────────────────────
#
# TextGrad wymaga żeby engine implementował EngineLM z metodą __call__
# która przyjmuje prompt i zwraca string.
# Wewnętrznie używamy LiteLLM żeby zachować spójność z resztą pipeline'u.

class GeminiEngine(EngineLM):
    """
    Custom TextGrad engine dla Gemini przez LiteLLM.

    TextGrad wywołuje engine na dwa sposoby:
    1. Forward pass:  engine(user_prompt) → string
    2. Backward pass: engine(gradient_prompt) → string (textual gradient)

    LiteLLM obsługuje oba tak samo — to jest duże uproszczenie względem
    natywnych enginów TextGrad które mają osobną logikę dla każdego providera.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

    def __init__(self, model: str, system_prompt: str = None, **litellm_kwargs):
        self.model          = model
        self.system_prompt  = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.litellm_kwargs = litellm_kwargs

    def generate(
        self,
        content:       Union[str, list],
        system_prompt: str  = None,
        **kwargs,
    ) -> str:
        """
        Główna metoda wywoływana przez TextGrad.
        content może być stringiem lub listą (multimodal) — obsługujemy oba.
        """
        if isinstance(content, list):
            # TextGrad czasem przekazuje listę (np. dla multimodal)
            # Łączymy tekstowe elementy
            text_parts = [
                c if isinstance(c, str) else str(c)
                for c in content
            ]
            prompt = "\n".join(text_parts)
        else:
            prompt = content

        sys = system_prompt or self.system_prompt

        try:
            response = litellm.completion(
                model    = self.model,
                messages = [
                    {"role": "system", "content": sys},
                    {"role": "user",   "content": prompt},
                ],
                temperature = 0.3,
                max_tokens  = 1024,
                **self.litellm_kwargs,
            )
            return response.choices[0].message.content

        except Exception as e:
            raise RuntimeError(f"GeminiEngine call failed: {e}") from e

    def __call__(
        self,
        content:       Union[str, list],
        system_prompt: str = None,
        **kwargs,
    ) -> str:
        """TextGrad wywołuje engine przez __call__ lub generate — obsługujemy oba."""
        return self.generate(content, system_prompt=system_prompt, **kwargs)


# ── Initial prompt ────────────────────────────────────────────────────────────

INITIAL_PROMPT = """
Extract structured factual knowledge from an AI news item.
Be precise. Use N/A for any field not supported by the source.
Never hallucinate — only extract what is explicitly stated.

For each item, produce:
SOURCE_TYPE: arxiv | blog | hackernews | other
MATURITY: research | beta | production
BENCHMARK: known benchmark name or N/A
RESULT: numeric benchmark result or N/A
REPLACES: what this replaces or improves, or N/A
GITHUB_URL: GitHub repo URL if mentioned, or N/A
LICENSE: MIT | Apache | proprietary | unclear
MATURITY_EVIDENCE: one sentence from source justifying maturity classification
""".strip()

# ── Evaluation — shared with SIMBA/GEPA (lib/extraction_metric.py) ────────────
#
# Uses the exact same rubric as optimize_prompts.py's extraction_metric and
# optimize_prompts_gepa.py's metric, so all three optimizers' scores are
# directly comparable (see compare_optimizers.py).


def evaluate_extraction(predicted_text: str, example: dict) -> dict:
    """
    Ocenia jakość ekstraktu i generuje textual feedback dla TextGrad.

    Kluczowe: TextGrad backpropaguje tekstowy opis błędów — im bardziej
    precyzyjny i actionable, tym lepszy "gradient" dla optymalizatora.
    """
    pred   = parse_extraction(predicted_text)
    source = example["input"]["content"]
    result = score_against_expected(pred, example["output"], source)

    return {
        "score":    result["score"],
        "feedback": feedback_text(result),
        "issues":   result["problems"],
    }


# ── TextGrad optimization loop ────────────────────────────────────────────────

def run_textgrad(
    trainset: list[dict],
    devset:   list[dict],
) -> tuple[str, dict]:
    """
    Uruchamia TextGrad z custom Gemini engines.

    Architektura:
      forward_engine  → wykonuje ekstrakcję (Flash — tańszy)
      backward_engine → generuje textual gradient (Pro — mocniejszy)

    TextGrad flow per przykład:
      1. system_prompt (Variable, requires_grad=True)
      2. forward_engine(system_prompt + input) → prediction
      3. evaluate(prediction) → textual feedback
      4. feedback.backward() → gradient propaguje do system_prompt
      5. optimizer.step() → aktualizuje system_prompt
    """
    forward_engine  = GeminiEngine(FORWARD_MODEL, **_LM_KWARGS)
    backward_engine = GeminiEngine(BACKWARD_MODEL, **_LM_KWARGS)

    # Ustaw backward engine globalnie dla TextGrad
    tg.set_backward_engine(backward_engine)

    # Prompt jako Variable — TextGrad będzie go modyfikować
    system_prompt = tg.Variable(
        INITIAL_PROMPT,
        requires_grad    = True,
        role_description = (
            "System prompt for an AI news knowledge extractor. "
            "Instructs the model how to extract structured factual "
            "information accurately without hallucination."
        ),
    )

    model     = tg.BlackboxLLM(forward_engine, system_prompt=system_prompt)
    optimizer = tg.TGD(parameters=[system_prompt])

    # Baseline
    print(f"  Baseline (dev n=5)...")
    baseline_scores = []
    for ex in devset[:5]:
        inp = (
            f"Title: {ex['input']['title']}\n"
            f"Source: {ex['input']['source']}\n"
            f"Content: {ex['input']['content'][:500]}"
        )
        try:
            pred   = model(tg.Variable(inp, requires_grad=False))
            result = evaluate_extraction(pred.value, ex)
            baseline_scores.append(result["score"])
        except Exception as e:
            print(f"    ✗ {e}")

    baseline_avg = (
        sum(baseline_scores) / len(baseline_scores)
        if baseline_scores else 0.0
    )
    print(f"  Baseline score: {baseline_avg:.3f}")

    import random
    random.seed(42)

    epoch_log = []

    for epoch in range(NUM_EPOCHS):
        print(f"\n  Epoch {epoch + 1}/{NUM_EPOCHS}")
        epoch_scores = []

        indices = list(range(len(trainset)))
        random.shuffle(indices)
        batches = [
            indices[i:i + BATCH_SIZE]
            for i in range(0, len(indices), BATCH_SIZE)
        ]

        for b_num, b_idx in enumerate(batches):
            batch        = [trainset[i] for i in b_idx]
            batch_scores = []

            for ex in batch:
                inp = (
                    f"Title: {ex['input']['title']}\n"
                    f"Source: {ex['input']['source']}\n"
                    f"Content: {ex['input']['content'][:500]}"
                )
                input_var = tg.Variable(
                    inp,
                    requires_grad    = False,
                    role_description = "AI news item to extract from",
                )

                try:
                    prediction = model(input_var)
                    result     = evaluate_extraction(prediction.value, ex)
                    batch_scores.append(result["score"])

                    # Loss Variable — TextGrad propaguje ten tekst wstecz
                    loss = tg.Variable(
                        result["feedback"],
                        requires_grad    = True,
                        role_description = (
                            "Evaluation feedback describing extraction errors. "
                            "Backpropagate this into the system prompt to fix them."
                        ),
                    )
                    loss.backward()

                except Exception as e:
                    print(f"      ✗ {e}")
                    continue

            if batch_scores:
                optimizer.step()
                optimizer.zero_grad()

                avg = sum(batch_scores) / len(batch_scores)
                epoch_scores.extend(batch_scores)
                print(
                    f"    Batch {b_num + 1}/{len(batches)}"
                    f"  score={avg:.3f}"
                )

        avg_epoch = (
            sum(epoch_scores) / len(epoch_scores)
            if epoch_scores else 0.0
        )
        epoch_log.append({
            "epoch":           epoch + 1,
            "avg_train_score": round(avg_epoch, 3),
            "prompt_chars":    len(system_prompt.value),
        })
        print(f"  Epoch {epoch + 1} avg: {avg_epoch:.3f}")

    # Final eval
    print(f"\n  Final eval (dev n={len(devset)})...")
    final_scores = []
    for ex in devset:
        inp = (
            f"Title: {ex['input']['title']}\n"
            f"Source: {ex['input']['source']}\n"
            f"Content: {ex['input']['content'][:500]}"
        )
        try:
            pred   = model(tg.Variable(inp, requires_grad=False))
            result = evaluate_extraction(pred.value, ex)
            final_scores.append(result["score"])
        except Exception as e:
            print(f"    ✗ {e}")

    final_avg = (
        sum(final_scores) / len(final_scores)
        if final_scores else 0.0
    )

    log = {
        "optimizer":       "TextGrad TGD + Gemini",
        "forward_model":   FORWARD_MODEL,
        "backward_model":  BACKWARD_MODEL,
        "num_epochs":      NUM_EPOCHS,
        "batch_size":      BATCH_SIZE,
        "trainset_size":   len(trainset),
        "devset_size":     len(devset),
        "baseline_score":  round(baseline_avg, 3),
        "optimized_score": round(final_avg, 3),
        "improvement":     round(final_avg - baseline_avg, 3),
        "epoch_log":       epoch_log,
    }

    return system_prompt.value, log


# ── Save ──────────────────────────────────────────────────────────────────────

def save(optimized_prompt: str, log: dict) -> None:
    PROMPTS_DIR.mkdir(exist_ok=True)

    result = {
        "generated_at":           datetime.now(timezone.utc).isoformat(),
        "optimizer":              "TextGrad + Gemini (LiteLLM)",
        "forward_model":          FORWARD_MODEL,
        "backward_model":         BACKWARD_MODEL,
        "optimizer_model":        None,             # TextGrad has no separate cheap compile model
        "reflection_model":       BACKWARD_MODEL,   # normalized key, see optimize_prompts.py/optimize_prompts_gepa.py
        "initial_prompt":         INITIAL_PROMPT,
        "optimized_instructions": optimized_prompt,
        "few_shot_demos":         [],
        "optimization_log":       log,
    }

    path = PROMPTS_DIR / "extractor_optimized_textgrad.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  ✓  {path}")
    print(f"  Compare against SIMBA/GEPA: uv run compare-optimizers")


# ── Load trainset ─────────────────────────────────────────────────────────────

def load_trainset() -> tuple[list[dict], list[dict]]:
    import random
    path = TRAINSET_DIR / "trainset_domain.json"
    if not path.exists():
        raise FileNotFoundError(
            "Brak artifacts/trainset/trainset_domain.json\n"
            "Uruchom najpierw: uv run gen-trainset"
        )
    raw      = json.loads(path.read_text())
    approved = [e for e in raw if e.get("approved", False)]
    if len(approved) < 8:
        raise ValueError(f"Za mało approved przykładów: {len(approved)} (min 8)")

    random.seed(42)
    random.shuffle(approved)
    split = max(1, int(len(approved) * 0.7))
    return approved[:split], approved[split:]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if _PROVIDER in ("ollama", "mlx"):
        api_key = os.environ.get("LLM_API_KEY", "fake")
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set\n"
            "  export GEMINI_API_KEY='your_key'\n"
            "  or add it to .env\n"
            "  To run locally without a key: LLM_PROVIDER=ollama or LLM_PROVIDER=mlx"
        )

    out = PROMPTS_DIR / "extractor_optimized_textgrad.json"
    if out.exists():
        d = json.loads(out.read_text())
        s = d.get("optimization_log", {}).get("optimized_score", "?")
        print(f"\n⏭  już istnieje (score={s}) — usuń żeby uruchomić ponownie")
        return

    print(f"\nTextGrad Prompt Optimizer")
    print(f"Forward:  {FORWARD_MODEL}  (execution)")
    print(f"Backward: {BACKWARD_MODEL}  (gradient generation)")
    print(f"Epochs: {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}")
    print(f"{'═' * 60}")

    trainset, devset = load_trainset()
    print(f"  Train: {len(trainset)} | Dev: {len(devset)}\n")

    optimized_prompt, log = run_textgrad(trainset, devset)
    save(optimized_prompt, log)

    print(f"\n{'═' * 60}")
    print(f"  Score: {log['baseline_score']} → {log['optimized_score']} "
          f"({log['improvement']:+.3f})")
    print(f"  Compare against SIMBA/GEPA: uv run compare-optimizers")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
