"""
Prompt Optimizer — GEPA edition
=================================
Runs DSPy's GEPA (reflective, evolutionary prompt optimizer) on the approved
trainset_domain.json and optimizes the same domain-level "AI Knowledge
Extractor" that optimize_prompts.py (SIMBA) and optimize_prompts_textgrad.py
(TextGrad) optimize — same Signature, same trainset, same metric, so all
three are directly comparable (see compare_optimizers.py).

GEPA vs SIMBA:
  SIMBA samples mini-batches of hard examples and generates self-reflective
  improvement rules locally. GEPA instead runs a genetic/Pareto search over
  candidate instructions, using a (typically stronger) reflection_lm to
  propose mutations from textual feedback on each candidate's failures.
  DSPy's docs report GEPA outperforming MIPROv2 by >10% in some evaluations —
  treat that as a vendor claim, not independently verified here.

Requires dspy>=3.0 (already the pinned floor; `gepa` ships as a transitive
dependency of the dspy package, no separate install needed).

Output:
  artifacts/prompts/extractor_optimized_gepa.json   ← optimized prompt

Usage:
  uv run optimize-gepa

generate_digests.py loads this instead of SIMBA's output when
PROMPT_SOURCE=gepa is set (see steps/generate_digests.py).
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import dspy
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.extraction_metric import dspy_example_to_dict, dspy_pred_to_dict, feedback_text, score_against_expected
from steps.optimize_prompts import (
    _LM_KWARGS, _PROVIDER, ExtractorProgram, PROMPTS_DIR, STUDENT_MODEL, load_trainset,
)

load_dotenv()

# Reflection model — defaults to a stronger tier of the same provider.
# ollama/mlx have no cheap/pro split locally, so they reuse STUDENT_MODEL
# unless REFLECTION_MODEL is set explicitly.
REFLECTION_MODEL = os.environ.get("REFLECTION_MODEL") or (
    "gemini/gemini-2.5-pro" if _PROVIDER == "gemini" else STUDENT_MODEL
)

# Cost/quality dial — "light" is the cheapest, matching SIMBA's cheap/fast
# positioning for this small-dataset use case. Bump to "medium"/"heavy" via
# GEPA_AUTO for a stronger (pricier) search. Uses `or` so a blank GEPA_AUTO=
# in .env falls back correctly instead of resolving to "".
GEPA_AUTO = os.environ.get("GEPA_AUTO") or "light"


# ── Metric — GEPA's 5-arg feedback contract ───────────────────────────────────
#
# GEPA requires exactly (gold, pred, trace, pred_name, pred_trace) and can
# return a float or a dspy.Prediction(score=..., feedback=...) — the richer
# form improves GEPA's reflective mutation, same idea as TextGrad's textual
# gradient.

def extraction_metric_gepa(gold, pred, trace=None, pred_name=None, pred_trace=None):
    p = dspy_pred_to_dict(pred)
    e = dspy_example_to_dict(gold)
    result = score_against_expected(p, e, gold.content)
    return dspy.Prediction(score=result["score"], feedback=feedback_text(result))


# ── Run GEPA ──────────────────────────────────────────────────────────────────

def run_gepa(
    program:       dspy.Module,
    trainset:      list[dspy.Example],
    devset:        list[dspy.Example],
    reflection_lm: dspy.LM,
) -> tuple[dspy.Module, dict]:
    """
    Runs the GEPA optimizer.

    valset=devset (rather than leaving it unset) avoids GEPA's
    "reusing trainset as valset" overfitting warning and gives it the same
    held-out split SIMBA/TextGrad use for their own baseline/optimized scores.
    """
    optimizer = dspy.GEPA(
        metric=extraction_metric_gepa,
        auto=GEPA_AUTO,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
        track_stats=True,
    )

    print(f"\n  Running GEPA (auto={GEPA_AUTO})...")
    print(f"  Reflection model: {REFLECTION_MODEL}")
    print()

    # Baseline score before optimization (same devset[:5] sampling SIMBA/TextGrad use)
    baseline_scores = [
        extraction_metric_gepa(ex, program(**ex.inputs())).score
        for ex in devset[:5]
    ]
    baseline_avg = sum(baseline_scores) / len(baseline_scores)
    print(f"  Baseline score (dev, n=5): {baseline_avg:.3f}")

    optimized = optimizer.compile(program, trainset=trainset, valset=devset)

    # Score after optimization
    optimized_scores = [
        extraction_metric_gepa(ex, optimized(**ex.inputs())).score
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
        "auto":            GEPA_AUTO,
    }

    return optimized, log


# ── Save results ──────────────────────────────────────────────────────────────

def save_optimized_prompt(optimized: dspy.Module, log: dict) -> None:
    """
    Saves the optimized prompt in the same schema as optimize_prompts.py
    (SIMBA) and optimize_prompts_textgrad.py (TextGrad) so
    compare_optimizers.py and generate_digests.py can treat all three
    uniformly.
    """
    PROMPTS_DIR.mkdir(exist_ok=True)

    opt_instructions = ""
    opt_demos        = []

    if hasattr(optimized, "extractor"):
        predictor = optimized.extractor
        if hasattr(predictor, "signature") and predictor.signature.instructions:
            opt_instructions = predictor.signature.instructions
        # GEPA is instruction-only (reflective mutation) — it typically
        # doesn't add few-shot demos, but check anyway in case a future
        # dspy version changes that.
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
        "optimizer":              "GEPA",
        "student_model":          STUDENT_MODEL,
        "optimizer_model":        None,               # GEPA has no separate cheap compile model
        "reflection_model":       REFLECTION_MODEL,    # normalized key, see optimize_prompts.py/optimize_prompts_textgrad.py
        "optimized_instructions": opt_instructions,
        "few_shot_demos":         opt_demos,
        "optimization_log":       log,
    }

    prompt_path = PROMPTS_DIR / "extractor_optimized_gepa.json"
    prompt_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  ✓  {prompt_path}")
    print(f"  Compare against SIMBA/TextGrad: uv run compare-optimizers")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_path = PROMPTS_DIR / "extractor_optimized_gepa.json"
    if out_path.exists():
        data  = json.loads(out_path.read_text())
        score = data.get("optimization_log", {}).get("optimized_score", "?")
        print(f"\n⏭  extractor_optimized_gepa.json already exists (score={score})")
        print(f"   Delete the file to re-run.")
        return

    print(f"\nPrompt Optimizer — GEPA")
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

    student_lm    = dspy.LM(STUDENT_MODEL, api_key=api_key, **_LM_KWARGS)
    reflection_lm = dspy.LM(REFLECTION_MODEL, api_key=api_key, **_LM_KWARGS)
    dspy.configure(lm=student_lm)

    print(f"\nLoading trainset...")
    trainset, devset = load_trainset()

    program = ExtractorProgram()

    optimized, log = run_gepa(program, trainset, devset, reflection_lm)

    print(f"\nSaving results...")
    save_optimized_prompt(optimized, log)

    print(f"\n{'═' * 60}")
    print(f"  Optimization complete!")
    print(f"  Score: {log['baseline_score']} → {log['optimized_score']} "
          f"({log['improvement']:+.3f})")
    print(f"\n  Next step: PROMPT_SOURCE=gepa uv run gen-digests")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
