"""
Synthetic Trainset Generator for SIMBA — domain level
=======================================================
Generuje labeled examples dla SIMBA które oceniają merytoryczną
jakość ekstrakcji wiedzy o AI — bez podziału na persony.

SIMBA optymalizuje domenowy "AI Knowledge Extractor":
  - Poprawna klasyfikacja maturity (research/beta/production)
  - Poprawna identyfikacja benchmarków i wyników numerycznych
  - Poprawne określenie co technologia zastępuje
  - Brak halucynacji — tylko to co jest w źródle

ACE personalizuje na wierzchu (osobny krok).

Użycie:
  python generate_trainset.py

Wymaga:
  data/snapshot_*.json (z collect.py)

Wynik:
  trainset/trainset_domain.json   ← labeled examples dla SIMBA
  trainset/review_report.md       ← raport do ręcznego review

Koszt: ~$0.05 (Gemini Flash, ~30 wywołań)
"""

import json
import random
import re
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

MODEL        = "gemini/gemini-2.5-flash"
DATA_DIR     = Path("data")
TRAINSET_DIR = Path("trainset")

EXAMPLES_PER_SNAPSHOT = 10
random.seed(42)

# ── Domain Extractor format ───────────────────────────────────────────────────
#
# To jest format który SIMBA będzie optymalizować.
# Nie ma tu nic per-persona — to jest czysta wiedza domenowa o AI item.
# ACE dostanie ten output i spersonalizuje prezentację.

EXTRACTOR_FORMAT = """
---EXTRACT---
TITLE: <tytuł, max 80 znaków>
SOURCE_TYPE: <arxiv | blog | hackernews | other>
MATURITY: <research | beta | production>
BENCHMARK: <nazwa benchmarku z: MMLU, HumanEval, GSM8K, SWE-bench, HELM, BIG-Bench, OSWorld, WebArena, GPQA, LiveCodeBench | N/A>
RESULT: <dokładny wynik numeryczny np. 87.3% | N/A>
REPLACES: <co konkretnie zastępuje lub ulepsza | N/A>
GITHUB_URL: <pełny URL github.com/org/repo | N/A>
LICENSE: <MIT | Apache | proprietary | unclear>
MATURITY_EVIDENCE: <jedno zdanie uzasadniające klasyfikację maturity>
---END---
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
- REPLACES: what existing tool, method, or approach does this make obsolete or improve?
  Be specific — "previous SOTA" is not acceptable, name the actual thing.
- GITHUB_URL: only if explicitly mentioned in the source. Do not search or infer.
- MATURITY_EVIDENCE: quote or closely paraphrase the specific phrase from the source
  that justifies your maturity classification.

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
# Metryka którą SIMBA będzie optymalizować.
# Sprawdza merytoryczną poprawność — nie format.

REQUIRED_FIELDS = [
    "TITLE", "SOURCE_TYPE", "MATURITY", "BENCHMARK",
    "RESULT", "REPLACES", "LICENSE", "MATURITY_EVIDENCE",
]

MATURITY_VALUES  = {"research", "beta", "production"}
SOURCE_TYPES     = {"arxiv", "blog", "hackernews", "other"}
LICENSE_VALUES   = {"MIT", "Apache", "proprietary", "unclear"}
KNOWN_BENCHMARKS = {
    "MMLU", "HumanEval", "GSM8K", "SWE-bench", "HELM",
    "BIG-Bench", "OSWorld", "WebArena", "GPQA", "LiveCodeBench", "N/A",
}


def score_extract(extract: dict, source_content: str) -> dict:
    """
    Ocenia merytoryczną jakość ekstraktu.
    Zwraca score 0.0-1.0 i listę problemów.

    To jest metryka którą SIMBA będzie optymalizować.
    """
    problems = []
    score    = 0.0
    checks   = 0

    # 1. Wszystkie pola obecne i niepuste
    for field in REQUIRED_FIELDS:
        checks += 1
        val = extract.get(field, "").strip()
        if val and val != "":
            score += 1
        else:
            problems.append(f"Missing or empty: {field}")

    # 2. Maturity z dozwolonych wartości
    checks += 1
    maturity = extract.get("MATURITY", "").lower()
    if maturity in MATURITY_VALUES:
        score += 1
    else:
        problems.append(f"Invalid MATURITY: '{maturity}' (must be research/beta/production)")

    # 3. Benchmark z dozwolonej listy
    checks += 1
    benchmark = extract.get("BENCHMARK", "N/A")
    if benchmark in KNOWN_BENCHMARKS:
        score += 1
    else:
        problems.append(f"Unknown BENCHMARK: '{benchmark}' — use N/A if unsure")

    # 4. Result jest N/A lub zawiera liczbę
    checks += 1
    result = extract.get("RESULT", "N/A")
    if result == "N/A" or re.search(r"\d+[\.,]\d+\s*%?", result):
        score += 1
    else:
        problems.append(f"RESULT should be numeric or N/A, got: '{result}'")

    # 5. Source type z dozwolonych wartości
    checks += 1
    src_type = extract.get("SOURCE_TYPE", "").lower()
    if src_type in SOURCE_TYPES:
        score += 1
    else:
        problems.append(f"Invalid SOURCE_TYPE: '{src_type}'")

    # 6. MATURITY_EVIDENCE nie jest generyczna
    checks += 1
    evidence = extract.get("MATURITY_EVIDENCE", "")
    generic_phrases = ["not mentioned", "no information", "unclear from source"]
    if len(evidence) > 20 and not any(g in evidence.lower() for g in generic_phrases):
        score += 1
    else:
        problems.append("MATURITY_EVIDENCE too short or generic")

    # 7. Anty-halucynacja: GITHUB_URL tylko jeśli w source
    checks += 1
    github = extract.get("GITHUB_URL", "N/A")
    if github == "N/A":
        score += 1   # brak URL jest zawsze OK
    elif "github.com" in source_content.lower():
        score += 1   # URL w source — OK
    else:
        problems.append(
            "GITHUB_URL present but not found in source — possible hallucination"
        )

    return {
        "score":    round(score / checks, 3),
        "problems": problems,
        "checks":   checks,
    }


# ── Item extraction from snapshots ────────────────────────────────────────────

def extract_items(snapshots: list[dict]) -> list[dict]:
    """
    Wyciąga itemy ze wszystkich snapshotów.
    Mix arXiv + RSS + HN żeby SIMBA nauczyła się obsługiwać różne formaty.
    """
    items = []

    for snap in snapshots:
        snap_id = snap["snapshot_id"]
        label   = snap["label"]

        # arXiv — najlepsze źródło dla benchmarków i maturity=research
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

        # RSS blog posts — najlepsze dla maturity=production/beta
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

        # HN — dobry mix, community signal
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
    Wybiera zbilansowany zestaw itemów:
    - Per snapshot: n itemów
    - Per source_type: mniej więcej równo arXiv/blog/hn
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

    try:
        response = litellm.completion(
            model    = MODEL,
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.1,   # bardzo niska — chcemy ekstrakcję faktów nie kreatywność
            max_tokens  = 400,
        )
        raw = response.choices[0].message.content
    except Exception as e:
        print(f"      ✗ API: {e}")
        return None

    # Parsuj output
    if "---EXTRACT---" not in raw or "---END---" not in raw:
        print(f"      ✗ Format error")
        return None

    chunk  = raw.split("---EXTRACT---")[1].split("---END---")[0].strip()
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

    # Oceń jakość
    quality = score_extract(output, item["content"])

    return {
        # Input dla SIMBA
        "input": {
            "title":       item["title"],
            "source":      item["source"],
            "source_type": item["source_type"],
            "date":        item["date"],
            "content":     item["content"],
        },
        # Expected output — wiedza domenowa
        "output":     output,
        "raw_output": raw,
        # Quality score — metryka SIMBA
        "quality": quality,
        # Metadata
        "snapshot_id": item["snapshot_id"],
        "label":       item["label"],
        # Review flags
        "approved":    quality["score"] >= 0.7,   # auto-approve jeśli score wysoki
        "review_note": (
            "; ".join(quality["problems"])
            if quality["problems"] else "auto-approved"
        ),
    }


# ── Review report ─────────────────────────────────────────────────────────────

def write_review_report(examples: list[dict]) -> None:
    """
    Generuje czytelny raport Markdown do ręcznego review.
    Pokazuje przykłady posortowane od najgorszych do najlepszych.
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
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        print(f"\n⏭  trainset_domain.json już istnieje ({len(existing)} przykładów)")
        print(f"   Usuń plik żeby wygenerować ponownie.")
        return

    # Załaduj snapshoty
    snap_paths = sorted(DATA_DIR.glob("snapshot_*.json"))
    if not snap_paths:
        print(f"✗  Brak snapshotów w {DATA_DIR}/")
        print("   Uruchom najpierw: python collect.py")
        return

    snapshots = [json.loads(p.read_text()) for p in snap_paths]
    print(f"\nSynthetic Trainset Generator — domain level")
    print(f"Model: {MODEL}")
    print(f"Snapshots: {len(snapshots)}")
    print(f"{'═' * 60}")

    # Wybierz zbilansowany zestaw itemów
    all_items = extract_items(snapshots)
    selected  = select_balanced(all_items, EXAMPLES_PER_SNAPSHOT)

    print(f"\nItemów do przetworzenia: {len(selected)}")
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
            print(f"           ✗  pominięto")

        time.sleep(1.5)

    # Zapisz
    out_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False))
    write_review_report(examples)

    print(f"\n{'═' * 60}")
    print(f"  Wygenerowano: {len(examples)} przykładów")
    print(f"  Auto-approved (score ≥ 0.7): {auto_approved}")
    print(f"  Wymaga review (score < 0.7): {needs_review}")
    print(f"\n  Następne kroki:")
    print(f"  1. Przejrzyj: trainset/review_report.md")
    print(f"  2. Edytuj:    trainset/trainset_domain.json")
    print(f"     (ustaw approved: false dla złych przykładów)")
    print(f"  3. Uruchom:   python optimize_prompts.py")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
