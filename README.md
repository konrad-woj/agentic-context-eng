# agentic-context-eng

Demo: SIMBA (DSPy) + ACE for automated AI news digest — two personas, diverging playbooks.

## Setup

```bash
uv sync
cp .env.example .env   # add GEMINI_API_KEY
```

Requires Python 3.11+.

## Project structure

```
steps/        — pipeline step scripts (collect, train, optimize, digest, rate, score)
lib/          — shared library code (ACE two-layer, MIPROv2 pipeline)
artifacts/    — all generated data (gitignored)
  data/       — collected AI news snapshots
  trainset/   — SIMBA training examples + review report
  prompts/    — SIMBA-optimized extraction prompt
  digests/    — generated persona digests
  ratings/    — interactive ratings
  scores/     — feedback scores + playbooks
docs/         — blog draft, reference docs
pipeline.py   — pipeline runner
```

## Running the pipeline

Run the full pipeline end-to-end:

```bash
uv run pipeline
```

Or run individual steps:

```bash
uv run pipeline --step collect
uv run pipeline --step gen-trainset
uv run pipeline --step optimize
uv run pipeline --step gen-digests
uv run pipeline --step rate
uv run pipeline --step score
```

Resume from a specific step (runs that step and all subsequent ones):

```bash
uv run pipeline --from optimize
```

Individual step commands still work too:

```bash
uv run collect        # → artifacts/data/snapshot_*.json
uv run gen-trainset   # → artifacts/trainset/trainset_domain.json
uv run optimize       # → artifacts/prompts/extractor_optimized.json
uv run gen-digests    # → artifacts/digests/*.json  (6 files: 3 snapshots × 2 personas)
uv run rate           # interactive TUI — ~30 min, saves progress on quit
uv run score          # → artifacts/scores/playbook_diff.json + artifacts/playbooks/*.json
```

After `gen-trainset`, review `artifacts/trainset/review_report.md` and set
`"approved": false` for any bad examples in `artifacts/trainset/trainset_domain.json`
before running `optimize`.

`gen-digests` falls back to a hardcoded prompt if `optimize` hasn't been run.
The SIMBA-optimized prompt produces better extractions but is not required.

## Switching LLM provider

Set `LLM_PROVIDER` in `.env` (copy from `.env.example`).

| Provider | `LLM_PROVIDER` | Default model | Key needed |
|----------|---------------|---------------|------------|
| MLX (local) | `mlx` | `openai/unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit` | — |
| Ollama (local) | `ollama` | `ollama/qwen3.6:35b-a3b-q4_K_M` | — |
| Gemini | `gemini` | `gemini/gemini-2.5-flash` | `GEMINI_API_KEY` |
| Anthropic | `gemini` | `anthropic/claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| OpenAI | `gemini` | `openai/gpt-4o` | `OPENAI_API_KEY` |

Override the default model by setting `MODEL` in `.env`. LiteLLM handles routing.

For MLX, start the inference server first:

```bash
~/.unsloth/unsloth_qwen3_6_mlx/bin/python -m mlx_lm.server \
  --model unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit --port 8080
```

## ACE library

`lib/ace_two_layer.py` and `lib/mipro_ace_pipeline.py` use the ACE library from
https://github.com/ace-agent/ace. Install it manually:

```bash
git clone https://github.com/ace-agent/ace.git /tmp/ace

# Add package discovery config so setuptools only picks up the ace/ dir:
cat >> /tmp/ace/pyproject.toml << 'EOF'

[tool.setuptools.packages.find]
include = ["ace*"]
EOF

uv pip install /tmp/ace
```

The main pipeline (`collect` → `score`) does **not** require ACE.
Only the reference architecture files in `lib/` need it.
