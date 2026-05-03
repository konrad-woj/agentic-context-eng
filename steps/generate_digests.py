"""
AI Digest Generator — LiteLLM edition
=======================================
Generates structured outputs for two personas (Alex and Jordan)
from the data collected by collect.py.

Uses LiteLLM as a provider-agnostic layer over Gemini 2.5 Pro.
Switch provider by changing one line: MODEL = "..."

Usage:
  uv run gen-digests

Alternative providers (change MODEL and the corresponding key in .env):
  MODEL = "anthropic/claude-sonnet-4-20250514"  → ANTHROPIC_API_KEY
  MODEL = "openai/gpt-4o"                       → OPENAI_API_KEY
  MODEL = "vertex_ai/gemini-2.5-pro"            → VERTEXAI_PROJECT + VERTEXAI_LOCATION

Local inference (no API key required):
  LLM_PROVIDER=ollama  → ollama serve  (then: ollama pull qwen3.6:35b-a3b-q4_K_M)
  LLM_PROVIDER=mlx     → start mlx_lm server first:
    ~/.unsloth/unsloth_qwen3_6_mlx/bin/python -m mlx_lm.server --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --port 8080

Output:
  digests/snapshot_A_jan2026_alex.json
  digests/snapshot_A_jan2026_jordan.json
  ... (6 files total)

Next step: uv run rate
"""

import json
import os
import time
from pathlib import Path

import litellm
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
if _PROVIDER == "ollama":
    _DEFAULT_MODEL = "ollama/qwen3.6:35b-a3b-q4_K_M"
    _LLM_KWARGS    = {"api_base": "http://localhost:11434"}
elif _PROVIDER == "mlx":
    _DEFAULT_MODEL = "openai/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit"
    _LLM_KWARGS    = {
        "api_base": "http://localhost:8080/v1",
        "api_key":  "fake",
    }
else:
    _DEFAULT_MODEL = "gemini/gemini-2.5-pro"
    _LLM_KWARGS    = {}
MODEL = os.environ.get("MODEL") or _DEFAULT_MODEL
ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "artifacts" / "data"
DIGESTS_DIR = ROOT / "artifacts" / "digests"
PROMPTS_DIR = ROOT / "artifacts" / "prompts"
MAX_ITEMS    = 20                         # max items per source (token budget control)

litellm.set_verbose = False

# ── Fallback system prompts per persona ──────────────────────────────────────
#
# Used if prompts/extractor_optimized.json does not exist.
# Once optimize_prompts.py has been run, generate_digest() automatically
# loads the SIMBA-optimized instructions and few-shot examples instead.

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
    Loads the SIMBA-optimized prompt from prompts/extractor_optimized.json.

    Returns (system_prompt, few_shot_demos, is_optimized).
    Falls back to the hardcoded prompt if the file does not exist.

    Two-layer architecture:
      - SIMBA optimized the domain extractor (instructions + few-shot)
      - Persona-specific formatting is added on top
    """
    opt_path = PROMPTS_DIR / "extractor_optimized.json"

    if not opt_path.exists():
        fallback = ALEX_SYSTEM_FALLBACK if persona == "alex" else JORDAN_SYSTEM_FALLBACK
        return fallback, [], False

    opt = json.loads(opt_path.read_text())

    domain_instructions = opt.get("optimized_instructions", "")
    few_shot_demos      = opt.get("few_shot_demos", [])

    if not domain_instructions:
        fallback = ALEX_SYSTEM_FALLBACK if persona == "alex" else JORDAN_SYSTEM_FALLBACK
        return fallback, [], False

    # Combine SIMBA domain instructions with persona-specific output format
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

    system = f"{domain_instructions}\n\n{persona_section}"
    return system, few_shot_demos, True


# ── Build user prompt ─────────────────────────────────────────────────────────

def build_user_prompt(snapshot: dict, persona: str) -> str:
    """
    Builds the user prompt from snapshot data.
    Alex: prioritizes arXiv + technical HN posts.
    Jordan: prioritizes RSS blog posts + top HN + production-relevant papers.
    Items are sorted by their persona-specific relevance score from collect.py.
    """
    score_field = "alex_score" if persona == "alex" else "jordan_score"

    lines = [
        f"# AI News Snapshot: {snapshot['label']}",
        f"# Period: {snapshot['date_from']} to {snapshot['date_to']}",
        "",
    ]

    if persona == "alex":
        lines.append("## arXiv Papers (sorted by relevance)")
        papers = sorted(
            snapshot["arxiv"],
            key=lambda x: x.get(score_field, 0),
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
            key=lambda x: x.get(score_field, 0),
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
    Parses CARD or BRIEF blocks from raw model output.
    Tolerant of minor formatting deviations.
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

        if len(card) >= 3:   # minimum viable fields to count as a valid block
            blocks.append(card)

    return blocks


# ── Generate one digest ───────────────────────────────────────────────────────

def generate_digest(snapshot: dict, persona: str) -> dict:
    system, few_shot_demos, is_optimized = load_optimized_prompt(persona)
    btype  = "CARD" if persona == "alex" else "BRIEF"
    prompt = build_user_prompt(snapshot, persona)

    # Few-shot demos from SIMBA are injected as conversation history
    # before the main prompt (max 2 to stay within token budget)
    system_content = "/no_think\n\n" + system if _PROVIDER == "mlx" else system
    messages = [{"role": "system", "content": system_content}]

    for demo in few_shot_demos[:2]:
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
        **_LLM_KWARGS,
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
    if _PROVIDER not in ("ollama", "mlx") and not os.environ.get("GEMINI_API_KEY"):
        raise EnvironmentError(
            "GEMINI_API_KEY not set.\n"
            "  Add it to .env or export it:\n"
            "  export GEMINI_API_KEY='your_key'\n"
            "  See .env.example for alternative providers.\n"
            "  To run locally without a key: LLM_PROVIDER=ollama or LLM_PROVIDER=mlx"
        )

    DIGESTS_DIR.mkdir(exist_ok=True)

    opt_path = PROMPTS_DIR / "extractor_optimized.json"
    if opt_path.exists():
        opt   = json.loads(opt_path.read_text())
        score = opt.get("optimization_log", {}).get("optimized_score", "?")
        print(f"  Prompt: SIMBA-optimized (score={score})")
    else:
        print(f"  Prompt: fallback hardcoded")
        print(f"  Tip: run uv run optimize to use the SIMBA-optimized prompt")
    print()

    snapshot_ids = ["A_jan2026", "B_feb2026", "C_mar2026"]
    personas     = ["alex", "jordan"]
    done = skipped = errors = 0

    for snap_id in snapshot_ids:
        snap_path = DATA_DIR / f"snapshot_{snap_id}.json"

        if not snap_path.exists():
            print(f"\n✗  Missing: {snap_path} — run uv run collect first")
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
                print(f"  ⏭  {out_path.name} already exists — skipping")
                skipped += 1
                continue

            print(f"  ⚙  Generating [{persona}] ...", end=" ", flush=True)

            try:
                result = generate_digest(snapshot, persona)
                out_path.write_text(
                    json.dumps(result, indent=2, ensure_ascii=False)
                )
                done += 1
                tokens = result["usage"]["total_tokens"]
                print(f"✓  {result['card_count']} cards | {tokens} tokens")

            except Exception as e:
                errors += 1
                print(f"✗  {e}")

            time.sleep(3)   # rate limit buffer

    total = len(snapshot_ids) * len(personas)
    print(f"\n\n{'═' * 62}")
    print(f"  Done: {done}/{total} | skipped: {skipped} | errors: {errors}")
    print(f"  Files: {DIGESTS_DIR}/")
    print(f"  Next step: uv run rate")
    print(f"{'═' * 62}")


if __name__ == "__main__":
    main()
