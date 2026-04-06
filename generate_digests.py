"""
AI Digest Generator — LiteLLM edition
=======================================
Generuje strukturalne outputy dla dwóch person (Alex i Jordan)
na podstawie danych zebranych przez collect.py.

Używa LiteLLM jako provider-agnostic layer nad Gemini 2.5 Pro.
Zmiana providera = zmiana jednej linii MODEL = "...".

Użycie:
  pip install litellm python-dotenv
  export GEMINI_API_KEY="twój_klucz"   # lub dodaj do .env
  python generate_digests.py

Alternatywni providerzy (zmień MODEL i klucz):
  MODEL = "anthropic/claude-sonnet-4-20250514"  → ANTHROPIC_API_KEY
  MODEL = "openai/gpt-4o"                       → OPENAI_API_KEY
  MODEL = "vertex_ai/gemini-2.5-pro"            → VERTEXAI_PROJECT + VERTEXAI_LOCATION

Wynik:
  digests/snapshot_A_jan2026_alex.json
  digests/snapshot_A_jan2026_jordan.json
  ... (6 plików łącznie)

Następny krok: python score_feedback.py
"""

import json
import os
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

# ── Konfiguracja ──────────────────────────────────────────────────────────────

MODEL        = "gemini/gemini-2.5-pro"   # zmień tu żeby użyć innego providera
DATA_DIR     = Path("data")
DIGESTS_DIR  = Path("digests")
PROMPTS_DIR  = Path("prompts")
MAX_ITEMS    = 20                         # max itemów per źródło (kontrola tokenów)

# Wyłącz verbose logi LiteLLM w produkcji
litellm.set_verbose = False

# ── Fallback system prompty per persona ──────────────────────────────────────
#
# Używane jeśli prompts/extractor_optimized.json nie istnieje.
# Gdy optimize_prompts.py zostanie uruchomiony, generate_digest()
# automatycznie załaduje zoptymalizowane instrukcje i few-shot examples
# zamiast tych hardcoded.

ALEX_SYSTEM_FALLBACK = """
You are a technical digest agent for Alex, a senior ML engineer.

Alex wants to know: does it work in production, what are the benchmarks,
is there code, what does it replace, what are hardware requirements.
Alex skips hype, funding news, and org changes entirely.

For each relevant item produce a REPLICATOR CARD in this exact format.
Do not add any text outside the card blocks.

---CARD---
TITLE: <paper or post title, max 80 chars>
ARXIV_ID: <e.g. 2502.12345 | N/A>
GITHUB_URL: <full github.com/org/repo URL | N/A>
BENCHMARK: <one of: MMLU, HumanEval, GSM8K, SWE-bench, HELM, BIG-Bench, OSWorld, WebArena, GPQA, LiveCodeBench | N/A>
RESULT: <numeric score e.g. 87.3% | N/A>
REPLACES: <existing tool or method this improves upon | N/A>
HARDWARE: <GPU/TPU spec or "CPU-only" | N/A>
LICENSE: <MIT | Apache | proprietary | unclear>
VERDICT: <one sentence max 20 words, no hype>
---END---

Rules:
- Include only ML engineering, model capabilities, tooling.
- Produce 5–10 cards. Quality over quantity.
""".strip()

JORDAN_SYSTEM_FALLBACK = """
You are a strategic digest agent for Jordan, a VP of Engineering.

Jordan makes decisions, not implementations. Jordan wants to know:
is this mature enough to care about, who is winning, what should
the team prioritise this quarter. No code, no architecture details.

For each relevant item produce a DECISION BRIEF in this exact format.
Do not add any text outside the brief blocks.

---BRIEF---
TITLE: <paper or post title, max 80 chars>
MATURITY: <research | beta | production>
LICENSE: <MIT | Apache | proprietary | unclear>
COMPETITOR: <main competing product or approach | N/A>
EFFORT: <days | weeks | months>
RECOMMENDATION: <go | wait | skip>
RATIONALE: <2–3 sentences. Include time horizon if relevant.>
---END---

Rules:
- Include only items with clear strategic or adoption implications.
- Skip pure research papers with no production path in < 12 months.
- Produce 5–8 briefs. Quality over quantity.
""".strip()


# ── Optimized prompt loader ───────────────────────────────────────────────────

def load_optimized_prompt(persona: str) -> tuple[str, list[dict], bool]:
    """
    Ładuje zoptymalizowany prompt z prompts/extractor_optimized.json.

    Zwraca (system_prompt, few_shot_demos, is_optimized).
    Jeśli plik nie istnieje — zwraca fallback prompt i is_optimized=False.

    Architektura dwuwarstwowa:
      - SIMBA zoptymalizował domenowy extractor (instrukcje + few-shot)
      - Persona-specific formatting jest dodawana na wierzchu
    """
    opt_path = PROMPTS_DIR / "extractor_optimized.json"

    if not opt_path.exists():
        # Fallback — optimize_prompts.py jeszcze nie był uruchomiony
        fallback = (
            ALEX_SYSTEM_FALLBACK
            if persona == "alex"
            else JORDAN_SYSTEM_FALLBACK
        )
        return fallback, [], False

    opt = json.loads(opt_path.read_text())

    # Zoptymalizowane instrukcje domenowe (SIMBA output)
    domain_instructions = opt.get("optimized_instructions", "")
    few_shot_demos      = opt.get("few_shot_demos", [])

    if not domain_instructions:
        fallback = (
            ALEX_SYSTEM_FALLBACK
            if persona == "alex"
            else JORDAN_SYSTEM_FALLBACK
        )
        return fallback, [], False

    # Buduj system prompt: domenowe instrukcje SIMBA + persona-specific format
    if persona == "alex":
        persona_section = """
You are presenting extracted knowledge to Alex, a senior ML engineer.
Format each item as a REPLICATOR CARD — technical, precise, no hype.
Do not add any text outside the card blocks.

---CARD---
TITLE: <paper or post title, max 80 chars>
ARXIV_ID: <arXiv ID from extraction | N/A>
GITHUB_URL: <GitHub URL from extraction | N/A>
BENCHMARK: <benchmark from extraction | N/A>
RESULT: <numeric result from extraction | N/A>
REPLACES: <what it replaces from extraction | N/A>
HARDWARE: <GPU/TPU spec or "CPU-only" | N/A>
LICENSE: <license from extraction>
VERDICT: <one sentence max 20 words, no hype>
---END---

Rules:
- Include only ML engineering, model capabilities, tooling.
- Produce 5–10 cards. Quality over quantity.
""".strip()
    else:
        persona_section = """
You are presenting extracted knowledge to Jordan, a VP of Engineering.
Format each item as a DECISION BRIEF — strategic, actionable, no code.
Do not add any text outside the brief blocks.

---BRIEF---
TITLE: <paper or post title, max 80 chars>
MATURITY: <maturity from extraction>
LICENSE: <license from extraction>
COMPETITOR: <main competing product or approach | N/A>
EFFORT: <days | weeks | months>
RECOMMENDATION: <go | wait | skip>
RATIONALE: <2–3 sentences. Include time horizon if relevant.>
---END---

Rules:
- Include only items with clear strategic or adoption implications.
- Skip pure research papers with no production path in < 12 months.
- Produce 5–8 briefs. Quality over quantity.
""".strip()

    # Złóż finalny system prompt
    system = f"{domain_instructions}\n\n{persona_section}"

    return system, few_shot_demos, True


# ── Build input text ──────────────────────────────────────────────────────────

def build_user_prompt(snapshot: dict, persona: str) -> str:
    """
    Buduje prompt użytkownika z danych snapshotu.
    Alex: priorytet arXiv + techniczne posty HN.
    Jordan: priorytet RSS blogi + top HN + wybrane papiery production-relevant.
    """
    lines = [
        f"# AI News Snapshot: {snapshot['label']}",
        f"# Period: {snapshot['date_from']} to {snapshot['date_to']}",
        "",
    ]

    if persona == "alex":
        lines.append("## arXiv Papers (sorted by relevance)")
        papers = sorted(
            snapshot["arxiv"],
            key=lambda x: x.get("persona_score", 0),
            reverse=True,
        )[:MAX_ITEMS]
        for p in papers:
            lines.append(f"### {p['title']}")
            lines.append(f"ID: {p['arxiv_id']} | {p['published'][:10]}")
            if p.get("github_url"):
                lines.append(f"Code: {p['github_url']}")
            lines.append(p.get("abstract", ""))
            lines.append("")

        lines.append("## HackerNews Top Posts")
        for h in sorted(snapshot["hn"], key=lambda x: x.get("points", 0), reverse=True)[:10]:
            lines.append(f"- [{h['points']}pts] {h['title']} — {h['url']}")

    else:  # jordan
        lines.append("## Lab Blog Posts")
        rss = sorted(
            snapshot["rss"],
            key=lambda x: x.get("persona_score", 0),
            reverse=True,
        )[:MAX_ITEMS]
        for r in rss:
            lines.append(f"### {r['title']}")
            lines.append(f"Source: {r['source']} | {r['published'][:10]}")
            lines.append(r.get("summary", ""))
            lines.append("")

        lines.append("## HackerNews Top Posts")
        for h in sorted(snapshot["hn"], key=lambda x: x.get("points", 0), reverse=True)[:10]:
            lines.append(f"- [{h['points']}pts] {h['title']} — {h['url']}")

        lines.append("")
        lines.append("## Production-Relevant Papers (selected)")
        prod_papers = [
            p for p in snapshot["arxiv"]
            if any(kw in (p.get("title", "") + p.get("abstract", "")).lower()
                   for kw in ["agent", "deploy", "production", "system", "evaluation", "benchmark"])
        ][:6]
        for p in prod_papers:
            lines.append(f"- {p['title']} ({p['arxiv_id']})")

    lines.append("")
    lines.append("Generate the digest now.")
    return "\n".join(lines)


# ── Parse structured output ───────────────────────────────────────────────────

def parse_blocks(raw: str, block_type: str) -> list[dict]:
    """
    Parsuje bloki CARD lub BRIEF z raw outputu modelu.
    Odporny na drobne odchylenia formatowania.
    """
    start = f"---{block_type}---"
    end   = "---END---"
    blocks = []

    for chunk in raw.split(start)[1:]:
        end_idx = chunk.find(end)
        content = chunk[:end_idx].strip() if end_idx != -1 else chunk.strip()

        card = {}
        for line in content.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                k = key.strip().upper().replace(" ", "_")
                v = val.strip()
                if k and v:
                    card[k] = v

        if len(card) >= 3:   # minimum sensownych pól żeby uznać za valid
            blocks.append(card)

    return blocks


# ── Generate one digest ───────────────────────────────────────────────────────

def generate_digest(snapshot: dict, persona: str) -> dict:
    system, few_shot_demos, is_optimized = load_optimized_prompt(persona)
    btype  = "CARD" if persona == "alex" else "BRIEF"
    prompt = build_user_prompt(snapshot, persona)

    # Buduj messages — few-shot demos z SIMBA wchodzą jako przykłady
    # w historii konwersacji przed głównym promptem
    messages = [{"role": "system", "content": system}]

    for demo in few_shot_demos[:2]:   # max 2 demo żeby nie przekroczyć budżetu
        demo_input = (
            f"Title: {demo.get('title', '')}\n"
            f"Content: {demo.get('content', demo.get('abstract', ''))[:400]}"
        )
        demo_output_fields = {
            k: v for k, v in demo.items()
            if k not in ("title", "content", "abstract", "source", "date")
        }
        demo_output = "\n".join(f"{k.upper()}: {v}" for k, v in demo_output_fields.items())
        messages.append({"role": "user",      "content": demo_input})
        messages.append({"role": "assistant", "content": f"---{btype}---\n{demo_output}\n---END---"})

    messages.append({"role": "user", "content": prompt})

    response = litellm.completion(
        model       = MODEL,
        messages    = messages,
        temperature = 0.2,
        max_tokens  = 4096,
    )

    raw    = response.choices[0].message.content
    blocks = parse_blocks(raw, btype)

    return {
        "snapshot_id":   snapshot["snapshot_id"],
        "label":         snapshot["label"],
        "date_from":     snapshot["date_from"],
        "date_to":       snapshot["date_to"],
        "persona":       persona,
        "model":         MODEL,
        "prompt_source": "simba_optimized" if is_optimized else "fallback_hardcoded",
        "raw_output":    raw,
        "cards":         blocks,
        "card_count":    len(blocks),
        "usage": {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError(
            "Brak GEMINI_API_KEY.\n"
            "  export GEMINI_API_KEY='twój_klucz'\n"
            "  lub dodaj do pliku .env"
        )

    DIGESTS_DIR.mkdir(exist_ok=True)

    # Pokaż jaki prompt będzie używany
    opt_path = PROMPTS_DIR / "extractor_optimized.json"
    if opt_path.exists():
        opt   = json.loads(opt_path.read_text())
        score = opt.get("optimization_log", {}).get("optimized_score", "?")
        print(f"  Prompt: SIMBA-optimized (score={score})")
    else:
        print(f"  Prompt: fallback hardcoded")
        print(f"  Tip: uruchom optimize_prompts.py żeby użyć SIMBA")
    print()

    snapshot_ids = ["A_jan2026", "B_feb2026", "C_mar2026"]
    personas     = ["alex", "jordan"]
    done = skipped = errors = 0

    for snap_id in snapshot_ids:
        snap_path = DATA_DIR / f"snapshot_{snap_id}.json"

        if not snap_path.exists():
            print(f"\n✗  Brak pliku: {snap_path} → uruchom collect.py")
            skipped += len(personas)
            continue

        snapshot = json.loads(snap_path.read_text())
        print(f"\n{'═' * 62}")
        print(f"  {snapshot['label']}")
        print(f"  arXiv {snapshot['counts']['arxiv']} | "
              f"HN {snapshot['counts']['hn']} | "
              f"RSS {snapshot['counts']['rss']}")
        print(f"{'═' * 62}")

        for persona in personas:
            out_path = DIGESTS_DIR / f"snapshot_{snap_id}_{persona}.json"

            if out_path.exists():
                print(f"  ⏭  {out_path.name} już istnieje → pomijam")
                skipped += 1
                continue

            print(f"  ⚙  Generowanie [{persona}] ...", end=" ", flush=True)

            try:
                result = generate_digest(snapshot, persona)
                out_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False)
                )
                done += 1
                tokens = result["usage"]["total_tokens"]
                print(f"✓  {result['card_count']} kart | {tokens} tokenów")

            except Exception as e:
                errors += 1
                print(f"✗  {e}")

            time.sleep(3)   # rate limit buffer

    total = len(snapshot_ids) * len(personas)
    print(f"\n\n{'═' * 62}")
    print(f"  Gotowe: {done}/{total} | pominięte: {skipped} | błędy: {errors}")
    print(f"  Pliki: {DIGESTS_DIR}/")
    print(f"  Następny krok: python score_feedback.py")
    print(f"{'═' * 62}")


if __name__ == "__main__":
    main()
