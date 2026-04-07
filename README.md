# agentic-context-eng

Demo: SIMBA (DSPy) + ACE for automated AI news digest — two personas, diverging playbooks.

## Quickstart

```bash
uv sync
cp .env.example .env   # add GEMINI_API_KEY
```

## Pipeline

Run steps in order:

```bash
uv run collect          # → data/snapshot_*.json
uv run gen-trainset     # → trainset/trainset_domain.json
uv run optimize         # → prompts/extractor_optimized.json
uv run gen-digests      # → digests/*.json  (6 files)
uv run rate             # → ratings/feedback.json  (~30 min manual)
uv run score            # → scores/playbook_diff.json
```

## ACE library

`ace_two_layer.py` and `mipro_ace_pipeline.py` require the ACE library from
https://github.com/ace-agent/ace. The repo's `pyproject.toml` is missing
explicit package discovery config, so direct `pip install git+...` fails.
Install it manually:

```bash
git clone https://github.com/ace-agent/ace.git /tmp/ace
# patch pyproject.toml to add [tool.setuptools.packages.find] include = ["ace*"]
uv pip install /tmp/ace
```

The main pipeline (collect → score) does **not** require ACE — it is only
needed for the reference architecture files.
