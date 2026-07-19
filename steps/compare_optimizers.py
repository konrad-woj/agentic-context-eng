"""
Optimizer Comparison — SIMBA vs GEPA vs TextGrad
===================================================
Discovers every optimizer output in artifacts/prompts/ and builds one
side-by-side comparison table.

Default mode reads each optimizer's own recorded optimization_log (no API
calls, free, instant). This is valid apples-to-apples because all three
optimizers now share the exact same metric (lib/extraction_metric.py) and
the exact same trainset/devset split (same trainset_domain.json, same
random.seed(42) 70/30 split via steps/optimize_prompts.py::load_trainset,
reused directly by optimize_prompts_gepa.py and mirrored by
optimize_prompts_textgrad.py) — so "optimized_score" already means the same
thing across all three.

--rescore attempts a live re-score against the full devset with one shared
baseline pass, instead of trusting each optimizer's own (independently
computed, small-sample) baseline. This requires reconstructing a runnable
DSPy program from each artifact's saved instructions — more fragile, and not
implemented for the TextGrad artifact (which isn't a DSPy program in the
first place). Marked experimental; falls back to recorded-only per artifact
if reconstruction isn't possible.

Usage:
  uv run compare-optimizers
  uv run compare-optimizers --rescore

Output:
  artifacts/prompts/optimizer_comparison.md
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT        = Path(__file__).parent.parent
PROMPTS_DIR = ROOT / "artifacts" / "prompts"

# Maps an artifact's own "optimizer" field to a short display name.
_DISPLAY_NAMES = {
    "SIMBA":                     "SIMBA (DSPy)",
    "GEPA":                      "GEPA (DSPy)",
    "TextGrad + Gemini (LiteLLM)": "TextGrad",
}


def discover_artifacts() -> list[dict]:
    """Loads every extractor_optimized*.json found in artifacts/prompts/."""
    artifacts = []
    for path in sorted(PROMPTS_DIR.glob("extractor_optimized*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ✗  Skipping {path.name}: {e}")
            continue
        data["_path"] = path
        artifacts.append(data)
    return artifacts


def models_used(artifact: dict) -> str:
    student = artifact.get("student_model") or artifact.get("forward_model") or "?"
    opt     = artifact.get("optimizer_model")
    refl    = artifact.get("reflection_model") or artifact.get("backward_model")
    parts = [student]
    if opt:
        parts.append(f"compile:{opt}")
    if refl:
        parts.append(f"reflect:{refl}")
    return " / ".join(parts)


def recorded_row(artifact: dict) -> dict:
    log = artifact.get("optimization_log", {})
    return {
        "optimizer":  _DISPLAY_NAMES.get(artifact.get("optimizer"), artifact.get("optimizer", "?")),
        "models":     models_used(artifact),
        "baseline":   log.get("baseline_score", "?"),
        "optimized":  log.get("optimized_score", "?"),
        "delta":      log.get("improvement", "?"),
        "notes":      "self-reported (recorded-only)",
    }


# ── Optional --rescore mode ───────────────────────────────────────────────────

def try_rescore(artifact: dict, devset) -> dict | None:
    """
    Best-effort live re-score against the full devset with the artifact's
    optimized_instructions. Returns None (caller falls back to recorded_row)
    if the artifact can't be turned into a runnable program (e.g. TextGrad's
    plain-text output, or any dspy API mismatch) — this mode is explicitly
    experimental, not required for the default comparison to be valid.
    """
    optimizer_name = artifact.get("optimizer", "")
    instructions   = artifact.get("optimized_instructions", "")
    if not instructions:
        return None

    try:
        from steps.optimize_prompts import ExtractorProgram, extraction_metric
    except ImportError:
        return None

    if "TextGrad" in optimizer_name:
        # Not a DSPy program — would need a separate plain-text litellm replay
        # path. Not implemented; recorded-only is already valid for TextGrad
        # since it shares the same metric/devset as SIMBA/GEPA.
        return None

    try:
        program = ExtractorProgram()
        program.extractor.signature = program.extractor.signature.with_instructions(instructions)
        scores = [extraction_metric(ex, program(**ex.inputs())) for ex in devset]
        avg = sum(scores) / len(scores) if scores else 0.0
        return {"rescored_optimized": round(avg, 3), "rescored_n": len(devset)}
    except Exception as e:
        print(f"  ⚠  --rescore failed for {optimizer_name}: {e} — falling back to recorded values")
        return None


def rescore_shared_baseline(devset) -> float | None:
    """One shared baseline pass (unoptimized program) over the full devset."""
    try:
        from steps.optimize_prompts import ExtractorProgram, extraction_metric
    except ImportError:
        return None
    try:
        program = ExtractorProgram()
        scores = [extraction_metric(ex, program(**ex.inputs())) for ex in devset]
        return round(sum(scores) / len(scores), 3) if scores else None
    except Exception as e:
        print(f"  ⚠  Shared baseline re-score failed: {e}")
        return None


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(rows: list[dict]) -> Path:
    lines = [
        "# Optimizer Comparison — SIMBA vs GEPA vs TextGrad",
        "",
        "All three optimize the same domain-extraction Signature against the",
        "same trainset/devset split, scored by the same metric "
        "(lib/extraction_metric.py) — see EXPERIMENT.md for details.",
        "",
        "| Optimizer | Models (student / compile / reflect) | Baseline | Optimized | Δ | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['optimizer']} | {row['models']} | {row['baseline']} "
            f"| {row['optimized']} | {row['delta']} | {row['notes']} |"
        )
    lines += [
        "",
        "Baseline scores above are each computed independently per optimizer "
        "run (small live-LLM sample, `devset[:5]`) — expect minor variance "
        "from sampling noise, not a methodology difference. Optimized scores "
        "are directly comparable since all three share the same metric and "
        "trainset split.",
    ]

    path = PROMPTS_DIR / "optimizer_comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Compare optimizer results")
    parser.add_argument(
        "--rescore", action="store_true",
        help="Experimental: attempt a live re-score with a shared baseline "
             "(requires a real LLM call per devset example; falls back to "
             "recorded values per-artifact if reconstruction isn't possible).",
    )
    args = parser.parse_args()

    if not PROMPTS_DIR.exists():
        print(f"✗  {PROMPTS_DIR} does not exist — run an optimizer first "
              f"(uv run optimize / optimize-gepa / steps/optimize_prompts_textgrad.py)")
        return

    artifacts = discover_artifacts()
    if not artifacts:
        print(f"✗  No extractor_optimized*.json files found in {PROMPTS_DIR}")
        print(f"   Run at least one optimizer first.")
        return

    print(f"\nOptimizer Comparison")
    print(f"{'═' * 60}")
    print(f"Found {len(artifacts)} optimizer artifact(s):")
    for a in artifacts:
        print(f"  - {a['_path'].name}  ({a.get('optimizer', '?')})")

    rows = []
    shared_baseline = None
    if args.rescore:
        try:
            from steps.optimize_prompts import load_trainset
            _, devset = load_trainset()
            shared_baseline = rescore_shared_baseline(devset)
        except Exception as e:
            print(f"  ⚠  Could not load devset for --rescore: {e}")

    for artifact in artifacts:
        row = recorded_row(artifact)
        if args.rescore and shared_baseline is not None:
            rescored = try_rescore(artifact, devset)
            if rescored is not None:
                row["baseline"] = shared_baseline
                row["optimized"] = rescored["rescored_optimized"]
                row["delta"] = round(rescored["rescored_optimized"] - shared_baseline, 3)
                row["notes"] = f"re-scored (n={rescored['rescored_n']}, shared baseline)"
        rows.append(row)

    report_path = write_report(rows)

    print(f"\n{'═' * 60}")
    for row in rows:
        print(f"  {row['optimizer']:20} {row['baseline']:>6} → {row['optimized']:>6}  "
              f"({row['delta']})  [{row['notes']}]")
    print(f"\n  ✓  {report_path}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
