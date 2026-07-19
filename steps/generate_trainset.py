"""
Synthetic Trainset Generator for SIMBA — domain level
=======================================================
Generates labeled examples for SIMBA that evaluate the factual quality
of AI knowledge extraction — no persona split at this stage.

SIMBA optimizes the domain-level "AI Knowledge Extractor":
  - Correct maturity classification (research/beta/production)
  - Correct identification of benchmarks and numeric results
  - Correct identification of what the technology replaces
  - No hallucinations — only extract what is in the source

ACE personalizes on top of the SIMBA-optimized extractor (separate step).

Usage:
  uv run gen-trainset

Requires:
  data/snapshot_*.json (from uv run collect)

Output:
  trainset/trainset_domain.json   ← labeled examples for SIMBA
  trainset/review_report.md       ← report for manual review

Cost: ~$0.05 (Gemini Flash, ~30 calls)
"""

import os
import json
import random
import sys
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
import lib.extraction_metric as _metric

load_dotenv()

_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
if _PROVIDER == "ollama":
    _DEFAULT_MODEL = "ollama/qwen3.6:35b-a3b-q4_K_M"
    _LLM_KWARGS    = {"api_base": "http://localhost:11434"}
elif _PROVIDER == "mlx":
    _DEFAULT_MODEL = "openai/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
    _LLM_KWARGS    = {
        "api_base": "http://localhost:8888/v1",
        "api_key":  os.environ.get("LLM_API_KEY", "fake"),
    }
else:
    _DEFAULT_MODEL = "gemini/gemini-2.5-flash"
    _LLM_KWARGS    = {}
MODEL = os.environ.get("MODEL") or _DEFAULT_MODEL
ROOT         = Path(__file__).parent.parent
DATA_DIR     = ROOT / "artifacts" / "data"
TRAINSET_DIR = ROOT / "artifacts" / "trainset"

EXAMPLES_PER_SNAPSHOT = 10
random.seed(42)

# ── Domain extractor output format ───────────────────────────────────────────
#
# This is the format SIMBA will optimize.
# No persona-specific fields — this is pure domain knowledge about an AI item.
# ACE receives this output and personalizes the presentation.

EXTRACTOR_FORMAT = """
<EXTRACT>
TITLE: <title, max 80 chars>
SOURCE_TYPE: <arxiv | blog | hackernews | other>
MATURITY: <research | beta | production>
BENCHMARK: <name from: MMLU, HumanEval, GSM8K, SWE-bench, HELM, BIG-Bench, OSWorld, WebArena, GPQA, LiveCodeBench | N/A>
RESULT: <exact numeric result e.g. 87.3% | N/A>
REPLACES: <what this concretely replaces or improves | N/A>
GITHUB_URL: <full github.com/org/repo URL | N/A>
LICENSE: <MIT | Apache | proprietary | unclear>
MATURITY_EVIDENCE: <one sentence justifying the maturity classification>
</EXTRACT>
""".strip()

# ── Generation prompt ─────────────────────────────────────────────────────────

GENERATION_PROMPT = """
You are a domain expert in AI/ML who extracts structured factual knowledge
from AI news items. Your job is to extract only what is explicitly stated
in the source — no hallucinations, no inferences beyond what is written.

Rules:
- MATURITY classification:
    research  = paper/preprint, no public API, no production deployments mentioned
    beta      = public API/demo exists but labeled preview/beta, or <6 months old with limited adoption
    production = stable API, widely deployed, used by paying customers or major orgs
- BENCHMARK: use only benchmark names from the allowed list. If a different benchmark
  is mentioned, use N/A rather than inventing a match.
- RESULT: copy the exact number from the source. If range given, use the best result.
- REPLACES: the specific existing tool, method, or approach that this work makes obsolete
  or directly competes with. This is NOT the paper's own contribution — it is the thing
  being replaced. Wrong: "A new framework for federated learning". Right: "FedAvg".
  If no specific prior work is named as being replaced, use N/A.
- GITHUB_URL: only if explicitly mentioned in the source. Do not search or infer.
- LICENSE: must be one of MIT | Apache | proprietary | unclear. Use "unclear" if the
  source does not mention a license. Never use N/A.
- MATURITY_EVIDENCE: copy a short verbatim phrase from CONTENT that justifies your
  maturity classification. Do not quote the SOURCE or DATE fields — only CONTENT.
  Wrong: "SOURCE: arXiv [cs.LG]". Right: "deployed by 12 enterprise customers".

Here is the AI news item to extract:

---
TITLE: {title}
SOURCE: {source}
DATE: {date}
CONTENT:
{content}
---

Extract structured knowledge now. Use N/A for any field not supported by the source.

{format}
""".strip()

# ── Metric for SIMBA ──────────────────────────────────────────────────────────
#
# This is the self-consistency check used to auto-approve/flag synthetic
# trainset examples. Field constants and scoring logic live in
# lib/extraction_metric.py (shared with optimize_prompts.py,
# optimize_prompts_gepa.py, optimize_prompts_textgrad.py) so every script
# agrees on what counts as a valid MATURITY/BENCHMARK/LICENSE/etc. value.

KNOWN_BENCHMARKS = _metric.KNOWN_BENCHMARKS   # kept for backward-compat imports


def score_extract(extract: dict, source_content: str) -> dict:
    """Scores the factual quality of an extraction. See lib/extraction_metric.py."""
    return _metric.score_self_consistency(extract, source_content, require_all_fields=True)


# ── Item extraction from snapshots ────────────────────────────────────────────

def extract_items(snapshots: list[dict]) -> list[dict]:
    """
    Extracts items from all snapshots.
    Mix of arXiv + RSS + HN so SIMBA learns to handle different formats.
    """
    items = []

    for snap in snapshots:
        snap_id = snap["snapshot_id"]
        label   = snap["label"]

        # arXiv — best source for benchmarks and maturity=research
        for p in snap.get("arxiv", []):
            items.append({
                "snapshot_id": snap_id,
                "label":       label,
                "source_type": "arxiv",
                "source":      f"arXiv [{p['category']}]",
                "title":       p["title"],
                "date":        p["published"][:10],
                "content": (
                    f"Abstract: {p['abstract']}\n"
                    f"arXiv ID: {p['arxiv_id']}\n" +
                    (f"Code: {p['github_url']}\n" if p.get("github_url") else "")
                ),
            })

        # RSS blog posts — best for maturity=production/beta
        for r in snap.get("rss", []):
            items.append({
                "snapshot_id": snap_id,
                "label":       label,
                "source_type": "blog",
                "source":      r["source"],
                "title":       r["title"],
                "date":        r.get("published", "")[:10],
                "content":     r.get("summary", "")[:800],
            })

        # HN — good mix, community signal
        for h in snap.get("hn", []):
            items.append({
                "snapshot_id": snap_id,
                "label":       label,
                "source_type": "hackernews",
                "source":      f"HackerNews [{h.get('points', 0)} pts]",
                "title":       h["title"],
                "date":        h.get("published", "")[:10],
                "content": (
                    f"URL: {h['url']}\n"
                    f"Points: {h.get('points', 0)} | "
                    f"Comments: {h.get('num_comments', 0)}"
                ),
            })

    return items


def select_balanced(items: list[dict], n_per_snapshot: int) -> list[dict]:
    """
    Selects a balanced set of items:
    - Per snapshot: n items
    - Per source_type: roughly equal arXiv/blog/HN split
    """
    by_snap: dict[str, list] = {}
    for item in items:
        by_snap.setdefault(item["snapshot_id"], []).append(item)

    selected = []
    for snap_id, snap_items in by_snap.items():
        by_type: dict[str, list] = {}
        for item in snap_items:
            by_type.setdefault(item["source_type"], []).append(item)

        # ~40% arXiv, ~40% blog, ~20% HN
        targets = {
            "arxiv":       max(1, int(n_per_snapshot * 0.40)),
            "blog":        max(1, int(n_per_snapshot * 0.40)),
            "hackernews":  max(1, int(n_per_snapshot * 0.20)),
        }

        for src_type, target in targets.items():
            pool = by_type.get(src_type, [])
            selected.extend(
                random.sample(pool, min(target, len(pool)))
            )

    return selected


# ── Generate one example ──────────────────────────────────────────────────────

def generate_example(item: dict) -> dict | None:
    prompt = GENERATION_PROMPT.format(
        title   = item["title"],
        source  = item["source"],
        date    = item["date"],
        content = item["content"],
        format  = EXTRACTOR_FORMAT,
    )

    raw = ""
    for attempt in range(2):
        try:
            user_content = ("/no_think\n\n" + prompt) if _PROVIDER == "mlx" else prompt
            messages = [{"role": "user", "content": user_content}]
            response = litellm.completion(
                model       = MODEL,
                messages    = messages,
                temperature = 0.1,
                max_tokens  = 4096,
                caching     = False,
                **_LLM_KWARGS,
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            print(f"      ✗ API: {e}")
            return None

        if "<EXTRACT>" in raw and "</EXTRACT>" in raw:
            break
        if attempt == 0:
            print(f"      ↻ Format error — retrying")
            time.sleep(2)

    # Parse output
    if "<EXTRACT>" not in raw or "</EXTRACT>" not in raw:
        print(f"      ✗ Format error — raw: {repr(raw[:400])}")
        return None

    chunk  = raw.split("<EXTRACT>")[1].split("</EXTRACT>")[0].strip()
    output = {}
    for line in chunk.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            k = key.strip().upper()
            v = val.strip()
            if k and v:
                output[k] = v

    if len(output) < 4:
        print(f"      ✗ Too few fields ({len(output)})")
        return None

    # Score quality
    quality = score_extract(output, item["content"])

    return {
        # Input for SIMBA
        "input": {
            "title":       item["title"],
            "source":      item["source"],
            "source_type": item["source_type"],
            "date":        item["date"],
            "content":     item["content"],
        },
        # Expected output — domain knowledge
        "output":     output,
        "raw_output": raw,
        # Quality score — SIMBA metric
        "quality": quality,
        # Metadata
        "snapshot_id": item["snapshot_id"],
        "label":       item["label"],
        # Review flags
        "approved":    quality["score"] >= 0.7,   # auto-approve if score is high
        "review_note": (
            "; ".join(quality["problems"])
            if quality["problems"] else "auto-approved"
        ),
    }


# ── Review report ─────────────────────────────────────────────────────────────

def write_review_report(examples: list[dict]) -> None:
    """
    Generates a readable Markdown report for manual review.
    Shows examples sorted from worst to best.
    """
    lines = [
        "# Trainset Review Report",
        "",
        f"Total examples: {len(examples)}",
        f"Auto-approved: {sum(1 for e in examples if e['approved'])}",
        f"Needs review: {sum(1 for e in examples if not e['approved'])}",
        "",
        "---",
        "",
        "## Examples Needing Review (score < 0.7)",
        "",
    ]

    needs_review = sorted(
        [e for e in examples if not e["approved"]],
        key=lambda x: x["quality"]["score"],
    )

    for e in needs_review:
        lines += [
            f"### {e['input']['title'][:70]}",
            f"- Score: {e['quality']['score']}",
            f"- Source: {e['input']['source']}",
            f"- Problems: {e['review_note']}",
            f"- Snapshot: {e['label']}",
            "",
            "**Output:**",
            "```",
        ]
        for k, v in e["output"].items():
            lines.append(f"{k}: {v}")
        lines += ["```", ""]

    lines += [
        "---",
        "",
        "## Approved Examples (score ≥ 0.7)",
        "",
    ]

    approved = sorted(
        [e for e in examples if e["approved"]],
        key=lambda x: x["quality"]["score"],
        reverse=True,
    )

    for e in approved:
        lines += [
            f"### {e['input']['title'][:70]}",
            f"- Score: {e['quality']['score']}",
            f"- Source: {e['input']['source']}",
            f"- Maturity: {e['output'].get('MATURITY', '?')}",
            f"- Benchmark: {e['output'].get('BENCHMARK', '?')} "
              f"→ {e['output'].get('RESULT', '?')}",
            "",
        ]

    report_path = TRAINSET_DIR / "review_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓  Review report: {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    TRAINSET_DIR.mkdir(exist_ok=True)

    out_path = TRAINSET_DIR / "trainset_domain.json"
    existing: list[dict] = []
    done_snap_ids: set[str] = set()

    if out_path.exists():
        existing = json.loads(out_path.read_text())
        done_snap_ids = {e["snapshot_id"] for e in existing}
        print(f"\n⏭  Found {len(existing)} existing examples "
              f"({len(done_snap_ids)} snapshot(s) already done)")

    # Load snapshots
    snap_paths = sorted(DATA_DIR.glob("snapshot_*.json"))
    if not snap_paths:
        print(f"✗  No snapshots found in {DATA_DIR}/")
        print("   Run first: uv run collect")
        return

    snapshots = [json.loads(p.read_text()) for p in snap_paths]
    print(f"\nSynthetic Trainset Generator — domain level")
    print(f"Model: {MODEL}")
    print(f"Snapshots: {len(snapshots)}")
    print(f"{'═' * 60}")

    # Select balanced set of items, skip already-processed snapshots
    all_items = extract_items(snapshots)
    selected  = [i for i in select_balanced(all_items, EXAMPLES_PER_SNAPSHOT)
                 if i["snapshot_id"] not in done_snap_ids]

    if not selected:
        print(f"\n✓  All snapshots already processed. Nothing to do.")
        print(f"   Delete {out_path} to regenerate from scratch.")
        return

    print(f"\nItems to process: {len(selected)} "
          f"(skipping {len(done_snap_ids)} already-done snapshot(s))")
    by_type = {}
    for item in selected:
        by_type[item["source_type"]] = by_type.get(item["source_type"], 0) + 1
    for src, count in sorted(by_type.items()):
        print(f"  {src:<15} {count}")
    print()

    examples      = []
    auto_approved = 0
    needs_review  = 0

    for i, item in enumerate(selected, 1):
        title_short = item["title"][:52]
        print(f"  [{i:2}/{len(selected)}] [{item['source_type']:11}] {title_short}...")

        example = generate_example(item)

        if example:
            examples.append(example)
            score = example["quality"]["score"]
            flag  = "✓" if example["approved"] else "⚠"
            print(f"           {flag}  score={score:.2f}  "
                  f"maturity={example['output'].get('MATURITY', '?')}")
            if example["approved"]:
                auto_approved += 1
            else:
                needs_review += 1
                for prob in example["quality"]["problems"][:2]:
                    print(f"              → {prob}")
        else:
            print(f"           ✗  skipped")

        time.sleep(1.5)

    # Save (merge new examples with any pre-existing ones)
    all_examples = existing + examples
    out_path.write_text(json.dumps(all_examples, indent=2, ensure_ascii=False))
    write_review_report(all_examples)

    print(f"\n{'═' * 60}")
    print(f"  New this run: {len(examples)} examples  (total: {len(all_examples)})")
    print(f"  Auto-approved (score ≥ 0.7): {auto_approved}")
    print(f"  Needs review  (score < 0.7): {needs_review}")
    print(f"\n  Next steps:")
    print(f"  1. Review: trainset/review_report.md")
    print(f"  2. Edit:   trainset/trainset_domain.json")
    print(f"     (set approved: false for bad examples)")
    print(f"  3. Run:    uv run optimize")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
