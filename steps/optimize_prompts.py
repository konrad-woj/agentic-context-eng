"""
Prompt Optimizer — SIMBA edition
==================================
Runs SIMBA on the approved trainset_domain.json and optimizes
the domain-level "AI Knowledge Extractor" prompt.

SIMBA (Stochastic Mini-Batch Ascent) outperforms MIPROv2 on small
datasets (15–30 examples) — it iterates locally instead of searching
for a global optimum via Bayesian Optimization.

Output:
  prompts/extractor_optimized.json   ← optimized prompt
  prompts/optimization_log.json      ← optimization history
  prompts/before_after.md            ← before/after comparison (for blog)

Usage:
  uv run optimize

generate_digests.py automatically loads the optimized prompt
if the file exists; otherwise it falls back to the hardcoded prompt.
"""

import json
import os
import random
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import dspy
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.extraction_metric import dspy_example_to_dict, dspy_pred_to_dict, score_against_expected

load_dotenv()

ROOT         = Path(__file__).parent.parent
TRAINSET_DIR = ROOT / "artifacts" / "trainset"
PROMPTS_DIR  = ROOT / "artifacts" / "prompts"

# ── DSPy setup ────────────────────────────────────────────────────────────────

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
_MODEL          = os.environ.get("MODEL") or _DEFAULT_MODEL
OPTIMIZER_MODEL = _MODEL
STUDENT_MODEL   = _MODEL


# ── DSPy Signature ────────────────────────────────────────────────────────────

class AINewsExtractor(dspy.Signature):
    """
    Extract structured factual knowledge from an AI news item.
    Be precise. Use N/A for any field not supported by the source.
    Never hallucinate — only extract what is explicitly stated.
    """
    title:   str = dspy.InputField(desc="Title of the AI news item")
    source:  str = dspy.InputField(desc="Source name and type (arXiv, blog, HN)")
    date:    str = dspy.InputField(desc="Publication date")
    content: str = dspy.InputField(desc="Full content or abstract of the item")

    source_type:       str = dspy.OutputField(desc="One of: arxiv | blog | hackernews | other")
    maturity:          str = dspy.OutputField(desc="One of: research | beta | production")
    benchmark:         str = dspy.OutputField(desc="Known benchmark name or N/A")
    result:            str = dspy.OutputField(desc="Numeric benchmark result or N/A")
    replaces:          str = dspy.OutputField(desc="What this replaces or improves, or N/A")
    github_url:        str = dspy.OutputField(desc="GitHub repo URL if mentioned, or N/A")
    license:           str = dspy.OutputField(desc="One of: MIT | Apache | proprietary | unclear")
    maturity_evidence: str = dspy.OutputField(
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


# ── Metric — what SIMBA optimizes ─────────────────────────────────────────────
#
# Field constants and scoring logic live in lib/extraction_metric.py, shared
# with optimize_prompts_gepa.py and optimize_prompts_textgrad.py so all three
# optimizers are judged by the exact same rubric and their results are
# directly comparable (see compare_optimizers.py).


def extraction_metric(example: dspy.Example, prediction, trace=None) -> float:
    """Factual quality metric for the extraction. See lib/extraction_metric.py."""
    pred = dspy_pred_to_dict(prediction)
    exp  = dspy_example_to_dict(example)
    return score_against_expected(pred, exp, example.content)["score"]


# ── Load trainset ─────────────────────────────────────────────────────────────

def load_trainset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Loads approved examples from trainset_domain.json.
    Splits into train (70%) and dev (30%) for SIMBA.
    """
    path = TRAINSET_DIR / "trainset_domain.json"
    if not path.exists():
        raise FileNotFoundError(
            "trainset/trainset_domain.json not found\n"
            "Run first: uv run gen-trainset"
        )

    raw      = json.loads(path.read_text())
    approved = [e for e in raw if e.get("approved", False)]

    if len(approved) < 10:
        raise ValueError(
            f"Too few approved examples: {len(approved)} (minimum 10).\n"
            "Check trainset/review_report.md and approve more examples."
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
    program:      dspy.Module,
    trainset:     list[dspy.Example],
    devset:       list[dspy.Example],
    optimizer_lm: dspy.LM,
) -> tuple[dspy.Module, dict]:
    """
    Runs the SIMBA optimizer.

    SIMBA vs MIPROv2 on small datasets:
    - Does not require a large validation set
    - Iterates locally — cheaper and faster
    - Generates self-reflective rules instead of searching the space globally

    optimizer_lm is passed as prompt_model to SIMBA, so it is used only
    during compilation. The student LM (configured globally via dspy.configure)
    is used for inference after optimization.
    """
    optimizer = dspy.SIMBA(
        metric       = extraction_metric,
        bsize        = min(8, len(trainset)),   # mini-batch size
        max_steps    = 12,                      # correct param name (not num_steps)
        prompt_model = optimizer_lm,            # cheaper Flash model for compilation
    )

    print(f"\n  Running SIMBA...")
    print(f"  Mini-batch size: {min(8, len(trainset))}")
    print(f"  Steps: 12")
    print(f"  Optimizer model: {OPTIMIZER_MODEL}")
    print(f"  Student model:   {STUDENT_MODEL}")
    print()

    # Baseline score before optimization
    baseline_scores = [
        extraction_metric(ex, program(**ex.inputs()))
        for ex in devset[:5]
    ]
    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    print(f"  Baseline score (dev, n=5): {baseline_avg:.3f}")

    optimized = optimizer.compile(
        program,
        trainset=trainset,
    )

    # Score after optimization (uses student LM, configured globally)
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
    Saves the optimized prompt in a format that generate_digests.py
    can load in place of the hardcoded system prompt.
    """
    PROMPTS_DIR.mkdir(exist_ok=True)

    # Extract optimized instructions from the DSPy program
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

    result = {
        "generated_at":           datetime.now(timezone.utc).isoformat(),
        "optimizer":              "SIMBA",
        "student_model":          STUDENT_MODEL,
        "optimizer_model":        OPTIMIZER_MODEL,
        "reflection_model":       None,   # SIMBA has no separate reflection/gradient model
        "optimized_instructions": opt_instructions,
        "few_shot_demos":         opt_demos,
        "optimization_log":       log,
    }

    prompt_path = PROMPTS_DIR / "extractor_optimized.json"
    prompt_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  ✓  {prompt_path}")

    log_path = PROMPTS_DIR / "optimization_log.json"
    log_path.write_text(json.dumps(
        {**log, "generated_at": result["generated_at"]},
        indent=2
    ))
    print(f"  ✓  {log_path}")

    write_before_after(opt_instructions, opt_demos, log)


def write_before_after(
    optimized_instructions: str,
    demos:                  list[dict],
    log:                    dict,
) -> None:
    """Generates a readable before/after comparison for the blog post."""

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
            "ACE then personalizes on top of this optimized extractor: "
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
    out_path = PROMPTS_DIR / "extractor_optimized.json"
    if out_path.exists():
        data  = json.loads(out_path.read_text())
        score = data.get("optimization_log", {}).get("optimized_score", "?")
        print(f"\n⏭  extractor_optimized.json already exists (score={score})")
        print(f"   Delete the file to re-run.")
        return

    print(f"\nPrompt Optimizer — SIMBA")
    print(f"{'═' * 60}")

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

    # Student LM — used for inference after optimization
    student_lm   = dspy.LM(STUDENT_MODEL,   api_key=api_key, **_LM_KWARGS)
    # Optimizer LM — cheaper model used only during SIMBA compilation
    optimizer_lm = dspy.LM(OPTIMIZER_MODEL, api_key=api_key, **_LM_KWARGS)
    dspy.configure(lm=student_lm)

    print(f"\nLoading trainset...")
    trainset, devset = load_trainset()

    program = ExtractorProgram()

    optimized, log = run_simba(program, trainset, devset, optimizer_lm)

    print(f"\nSaving results...")
    save_optimized_prompt(program, optimized, log)

    print(f"\n{'═' * 60}")
    print(f"  Optimization complete!")
    print(f"  Score: {log['baseline_score']} → {log['optimized_score']} "
          f"({log['improvement']:+.3f})")
    print(f"\n  Next step: uv run gen-digests")
    print(f"  (automatically loads the optimized prompt)")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
