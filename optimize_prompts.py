"""
Prompt Optimizer — SIMBA edition
==================================
Uruchamia SIMBA na zatwierdzonym trainset_domain.json
i optymalizuje domenowy "AI Knowledge Extractor" prompt.

SIMBA (Stochastic Mini-Batch Ascent) jest lepszy od MIPROv2
przy małych datasetach (15-30 przykładów) — iteruje lokalnie
zamiast szukać globalnego optimum przez Bayesian Optimization.

Wynik:
  prompts/extractor_optimized.json   ← zoptymalizowany prompt
  prompts/optimization_log.json      ← historia optymalizacji
  prompts/before_after.md            ← porównanie przed/po (do blogu)

Użycie:
  python optimize_prompts.py

generate_digests.py automatycznie załaduje zoptymalizowany prompt
jeśli plik istnieje.
"""

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import dspy
from dotenv import load_dotenv

load_dotenv()

TRAINSET_DIR = Path("trainset")
PROMPTS_DIR  = Path("prompts")

# ── DSPy setup ────────────────────────────────────────────────────────────────

# SIMBA używa tego modelu do oceny i generowania instrukcji
# Flash wystarczy — oszczędza koszt względem Pro
OPTIMIZER_MODEL = "gemini/gemini-2.5-flash"
# Ten model będzie używany przez zoptymalizowany prompt w produkcji
STUDENT_MODEL   = "gemini/gemini-2.5-pro"


# ── DSPy Signature — co program ma robić ─────────────────────────────────────

class AINewsExtractor(dspy.Signature):
    """
    Extract structured factual knowledge from an AI news item.
    Be precise. Use N/A for any field not supported by the source.
    Never hallucinate — only extract what is explicitly stated.
    """
    title:       str = dspy.InputField(desc="Title of the AI news item")
    source:      str = dspy.InputField(desc="Source name and type (arXiv, blog, HN)")
    date:        str = dspy.InputField(desc="Publication date")
    content:     str = dspy.InputField(desc="Full content or abstract of the item")

    source_type:        str = dspy.OutputField(desc="One of: arxiv | blog | hackernews | other")
    maturity:           str = dspy.OutputField(desc="One of: research | beta | production")
    benchmark:          str = dspy.OutputField(desc="Known benchmark name or N/A")
    result:             str = dspy.OutputField(desc="Numeric benchmark result or N/A")
    replaces:           str = dspy.OutputField(desc="What this replaces or improves, or N/A")
    github_url:         str = dspy.OutputField(desc="GitHub repo URL if mentioned, or N/A")
    license:            str = dspy.OutputField(desc="One of: MIT | Apache | proprietary | unclear")
    maturity_evidence:  str = dspy.OutputField(
        desc="One sentence quoting or paraphrasing source text that justifies maturity classification"
    )


# ── DSPy Program ──────────────────────────────────────────────────────────────

class ExtractorProgram(dspy.Module):
    def __init__(self):
        self.extractor = dspy.ChainOfThought(AINewsExtractor)

    def forward(self, title, source, date, content):
        return self.extractor(
            title=title, source=source, date=date, content=content
        )


# ── Metric — metryka którą SIMBA optymalizuje ─────────────────────────────────

KNOWN_BENCHMARKS = {
    "MMLU", "HumanEval", "GSM8K", "SWE-bench", "HELM",
    "BIG-Bench", "OSWorld", "WebArena", "GPQA", "LiveCodeBench", "N/A",
}
MATURITY_VALUES = {"research", "beta", "production"}
SOURCE_TYPES    = {"arxiv", "blog", "hackernews", "other"}
LICENSE_VALUES  = {"MIT", "Apache", "proprietary", "unclear"}


def extraction_metric(example: dspy.Example, prediction, trace=None) -> float:
    """
    Metryka merytorycznej jakości ekstraktu.
    Identyczna logika jak score_extract() w generate_trainset.py
    — spójność jest kluczowa żeby SIMBA optymalizowała właściwą rzecz.

    Zwraca float 0.0–1.0.
    """
    score  = 0.0
    checks = 0

    # 1. Maturity poprawny i zgodny z expected
    checks += 1
    pred_maturity = getattr(prediction, "maturity", "").strip().lower()
    exp_maturity  = example.maturity.strip().lower()
    if pred_maturity in MATURITY_VALUES:
        score += 0.5                          # poprawna wartość
        if pred_maturity == exp_maturity:
            score += 0.5                      # zgodna z expected
    checks += 1   # liczymy jako dwa sub-checks

    # 2. Benchmark z dozwolonej listy
    checks += 1
    pred_bench = getattr(prediction, "benchmark", "N/A").strip()
    if pred_bench in KNOWN_BENCHMARKS:
        score += 1
        # Bonus jeśli zgodny z expected i nie jest N/A
        if pred_bench == example.benchmark and pred_bench != "N/A":
            score += 0.5
            checks += 1

    # 3. Result — N/A lub numeryczny
    checks += 1
    import re
    pred_result = getattr(prediction, "result", "N/A").strip()
    exp_result  = example.result.strip()
    if pred_result == "N/A" or re.search(r"\d+[\.,]\d+", pred_result):
        score += 1
    # Bonus za dokładną zgodność z expected
    if pred_result == exp_result and exp_result != "N/A":
        score += 0.5
        checks += 1

    # 4. Source type poprawny
    checks += 1
    pred_src = getattr(prediction, "source_type", "").strip().lower()
    if pred_src in SOURCE_TYPES:
        score += 1

    # 5. License z dozwolonej listy
    checks += 1
    pred_lic = getattr(prediction, "license", "").strip()
    if pred_lic in LICENSE_VALUES:
        score += 1

    # 6. Maturity evidence — nie generyczna, min 20 znaków
    checks += 1
    evidence = getattr(prediction, "maturity_evidence", "").strip()
    generic  = ["not mentioned", "no information", "unclear from source", "n/a"]
    if len(evidence) >= 20 and not any(g in evidence.lower() for g in generic):
        score += 1

    # 7. GitHub URL — anty-halucynacja
    checks += 1
    pred_gh = getattr(prediction, "github_url", "N/A").strip()
    exp_gh  = example.github_url.strip()
    if pred_gh == "N/A":
        score += 1   # brak URL jest zawsze bezpieczny
    elif "github.com" in example.content.lower():
        score += 1   # URL w source — OK
    # else: możliwa halucynacja, brak punktu

    return round(score / max(checks, 1), 3)


# ── Load trainset ─────────────────────────────────────────────────────────────

def load_trainset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Ładuje approved przykłady z trainset_domain.json.
    Dzieli na train (70%) i dev (30%) dla SIMBA.
    """
    path = TRAINSET_DIR / "trainset_domain.json"
    if not path.exists():
        raise FileNotFoundError(
            "Brak trainset/trainset_domain.json\n"
            "Uruchom najpierw: python generate_trainset.py"
        )

    raw      = json.loads(path.read_text())
    approved = [e for e in raw if e.get("approved", False)]

    if len(approved) < 10:
        raise ValueError(
            f"Za mało approved przykładów: {len(approved)} (minimum 10).\n"
            "Sprawdź trainset/review_report.md i zatwierdź więcej przykładów."
        )

    examples = []
    for e in approved:
        inp = e["input"]
        out = e["output"]
        examples.append(
            dspy.Example(
                # Inputs
                title   = inp["title"],
                source  = inp["source"],
                date    = inp.get("date", ""),
                content = inp["content"],
                # Expected outputs
                source_type       = out.get("SOURCE_TYPE", "other").lower(),
                maturity          = out.get("MATURITY", "research").lower(),
                benchmark         = out.get("BENCHMARK", "N/A"),
                result            = out.get("RESULT", "N/A"),
                replaces          = out.get("REPLACES", "N/A"),
                github_url        = out.get("GITHUB_URL", "N/A"),
                license           = out.get("LICENSE", "unclear"),
                maturity_evidence = out.get("MATURITY_EVIDENCE", ""),
            ).with_inputs("title", "source", "date", "content")
        )

    # Shuffle i podziel
    import random
    random.seed(42)
    random.shuffle(examples)

    split    = max(1, int(len(examples) * 0.7))
    trainset = examples[:split]
    devset   = examples[split:]

    print(f"  Approved examples: {len(approved)}")
    print(f"  Train: {len(trainset)} | Dev: {len(devset)}")

    return trainset, devset


# ── Run SIMBA ─────────────────────────────────────────────────────────────────

def run_simba(
    program:  dspy.Module,
    trainset: list[dspy.Example],
    devset:   list[dspy.Example],
) -> tuple[dspy.Module, dict]:
    """
    Uruchamia SIMBA optimizer.

    SIMBA vs MIPROv2 przy małych datasetach:
    - Nie potrzebuje dużego validation setu
    - Iteruje lokalnie — tańszy i szybszy
    - Generuje self-reflective rules zamiast przeszukiwać przestrzeń
    """
    optimizer = dspy.SIMBA(
        metric    = extraction_metric,
        bsize     = min(8, len(trainset)),   # mini-batch size
        num_steps = 12,                      # liczba iteracji
    )

    print(f"\n  Uruchamiam SIMBA...")
    print(f"  Mini-batch size: {min(8, len(trainset))}")
    print(f"  Steps: 12")
    print(f"  Model: {OPTIMIZER_MODEL}")
    print()

    # Baseline score przed optymalizacją
    baseline_scores = [
        extraction_metric(ex, program(**ex.inputs()))
        for ex in devset[:5]
    ]
    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    print(f"  Baseline score (dev, n=5): {baseline_avg:.3f}")

    optimized = optimizer.compile(
        program,
        trainset = trainset,
    )

    # Score po optymalizacji
    optimized_scores = [
        extraction_metric(ex, optimized(**ex.inputs()))
        for ex in devset
    ]
    optimized_avg = sum(optimized_scores) / len(optimized_scores)
    print(f"  Optimized score (dev, n={len(devset)}): {optimized_avg:.3f}")
    print(f"  Improvement: {optimized_avg - baseline_avg:+.3f}")

    log = {
        "baseline_score":  round(baseline_avg, 3),
        "optimized_score": round(optimized_avg, 3),
        "improvement":     round(optimized_avg - baseline_avg, 3),
        "trainset_size":   len(trainset),
        "devset_size":     len(devset),
    }

    return optimized, log


# ── Save results ──────────────────────────────────────────────────────────────

def save_optimized_prompt(
    original:  dspy.Module,
    optimized: dspy.Module,
    log:       dict,
) -> None:
    """
    Zapisuje zoptymalizowany prompt w formacie który generate_digests.py
    może załadować zamiast hardcoded system promptu.
    """
    PROMPTS_DIR.mkdir(exist_ok=True)

    # Wyciągnij zoptymalizowane instrukcje z DSPy programu
    opt_instructions = ""
    opt_demos        = []

    if hasattr(optimized, "extractor"):
        predictor = optimized.extractor
        if hasattr(predictor, "signature") and predictor.signature.instructions:
            opt_instructions = predictor.signature.instructions
        if hasattr(predictor, "demos") and predictor.demos:
            for demo in predictor.demos[:3]:
                demo_dict = {}
                for field in ["title", "source", "date", "content",
                               "maturity", "benchmark", "result",
                               "source_type", "license", "maturity_evidence"]:
                    if hasattr(demo, field):
                        demo_dict[field] = getattr(demo, field)
                if demo_dict:
                    opt_demos.append(demo_dict)

    # Zapisz w formacie który generate_digests.py załaduje
    result = {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "optimizer":           "SIMBA",
        "student_model":       STUDENT_MODEL,
        "optimizer_model":     OPTIMIZER_MODEL,
        "optimized_instructions": opt_instructions,
        "few_shot_demos":      opt_demos,
        "optimization_log":    log,
    }

    prompt_path = PROMPTS_DIR / "extractor_optimized.json"
    prompt_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  ✓  {prompt_path}")

    # Log
    log_path = PROMPTS_DIR / "optimization_log.json"
    log_path.write_text(json.dumps(
        {**log, "generated_at": result["generated_at"]},
        indent=2
    ))
    print(f"  ✓  {log_path}")

    # Before/after dla blogu
    write_before_after(opt_instructions, opt_demos, log)


def write_before_after(
    optimized_instructions: str,
    demos:                  list[dict],
    log:                    dict,
) -> None:
    """Generuje czytelne porównanie przed/po optymalizacji do sekcji blogu."""

    original_instructions = textwrap.dedent("""
        Extract structured factual knowledge from an AI news item.
        Be precise. Use N/A for any field not supported by the source.
        Never hallucinate — only extract what is explicitly stated.
    """).strip()

    lines = [
        "# SIMBA Optimization — Before / After",
        "",
        "## Optimization Results",
        "",
        f"- Baseline score:  {log['baseline_score']}",
        f"- Optimized score: {log['optimized_score']}",
        f"- Improvement:     {log['improvement']:+.3f}",
        f"- Trainset size:   {log['trainset_size']} examples",
        "",
        "---",
        "",
        "## Before — Original Instructions",
        "",
        "```",
        original_instructions,
        "```",
        "",
        "## After — SIMBA-Optimized Instructions",
        "",
        "```",
        optimized_instructions or "(no change — baseline already optimal)",
        "```",
        "",
    ]

    if demos:
        lines += [
            "## Few-Shot Examples Added by SIMBA",
            "",
            f"SIMBA added {len(demos)} few-shot example(s) to the prompt.",
            "",
        ]
        for i, demo in enumerate(demos, 1):
            lines += [
                f"### Example {i}",
                f"- Title: {demo.get('title', '?')[:60]}",
                f"- Maturity: {demo.get('maturity', '?')}",
                f"- Benchmark: {demo.get('benchmark', '?')} → {demo.get('result', '?')}",
                "",
            ]

    lines += [
        "---",
        "",
        "## What This Means",
        "",
        (
            "The optimized instructions are what `generate_digests.py` will use "
            "instead of the hand-written system prompt. SIMBA found these by "
            "iterating over mini-batches of training examples and generating "
            "self-reflective improvement rules — no manual prompt engineering required."
        ),
        "",
        (
            "ACE will then personalize on top of this optimized extractor: "
            "Alex sees a REPLICATOR CARD, Jordan sees a DECISION BRIEF. "
            "The domain knowledge (maturity, benchmarks, what it replaces) "
            "comes from SIMBA. The persona lens comes from ACE."
        ),
    ]

    path = PROMPTS_DIR / "before_after.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓  {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Sprawdź czy prompt już istnieje
    out_path = PROMPTS_DIR / "extractor_optimized.json"
    if out_path.exists():
        data = json.loads(out_path.read_text())
        score = data.get("optimization_log", {}).get("optimized_score", "?")
        print(f"\n⏭  extractor_optimized.json już istnieje (score={score})")
        print(f"   Usuń plik żeby uruchomić ponownie.")
        return

    print(f"\nPrompt Optimizer — SIMBA")
    print(f"{'═' * 60}")

    # Skonfiguruj DSPy z dwoma modelami:
    # - student: model który będzie używany w produkcji
    # - optimizer: model który generuje instrukcje (może być słabszy/tańszy)
    import os
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Brak GEMINI_API_KEY\n"
            "  export GEMINI_API_KEY='twój_klucz'"
        )

    student_lm   = dspy.LM(STUDENT_MODEL,   api_key=api_key)
    optimizer_lm = dspy.LM(OPTIMIZER_MODEL, api_key=api_key)
    dspy.configure(lm=student_lm)

    # Załaduj trainset
    print(f"\nŁadowanie trainset...")
    trainset, devset = load_trainset()

    # Inicjalizuj program
    program = ExtractorProgram()

    # Uruchom SIMBA
    optimized, log = run_simba(program, trainset, devset)

    # Zapisz wyniki
    print(f"\nZapisywanie wyników...")
    save_optimized_prompt(program, optimized, log)

    print(f"\n{'═' * 60}")
    print(f"  Optymalizacja zakończona!")
    print(f"  Score: {log['baseline_score']} → {log['optimized_score']} "
          f"({log['improvement']:+.3f})")
    print(f"\n  Następny krok: python generate_digests.py")
    print(f"  (automatycznie załaduje zoptymalizowany prompt)")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
