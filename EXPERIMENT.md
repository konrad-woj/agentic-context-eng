# Experiments

Log of alternative approaches explored on the `experiment` branch, benchmarked
or compared against the default pipeline (`steps/`, see `README.md`). New
experiments get appended as new numbered sections below.

## 1. Prompt optimization: SIMBA vs GEPA vs TextGrad

All three optimize the exact same thing: the domain-level "AI Knowledge
Extractor" Signature (`steps/optimize_prompts.py::AINewsExtractor`), against
the same `artifacts/trainset/trainset_domain.json`, scored by the same
metric (`lib/extraction_metric.py::score_against_expected`). That's
deliberate — it's what makes `compare-optimizers` (§2) a fair comparison
instead of three different rubrics in a trenchcoat.

### SIMBA (default)

`steps/optimize_prompts.py`, run via `uv run optimize`. DSPy's stochastic
mini-batch ascent optimizer — this is what the main pipeline uses to optimize
the domain extractor prompt. Samples mini-batches of hard examples and
generates self-reflective improvement rules locally. Cheap, fast, well-suited
to small datasets (15–30 examples). See `README.md` for how it fits into the
pipeline.

### GEPA (alternative)

`steps/optimize_prompts_gepa.py`, run via `uv run optimize-gepa`.

#### Why GEPA?

GEPA (Genetic-Pareto) is DSPy's reflective, evolutionary prompt optimizer —
it runs a Pareto search over candidate instructions, using a (typically
stronger) `reflection_lm` to propose mutations from textual feedback on each
candidate's failures. DSPy's own docs report GEPA outperforming MIPROv2 by
more than 10% in some evaluations — treat that as a vendor claim, not
independently verified in this repo.

Requires `dspy>=3.0` (already the pinned floor in `pyproject.toml`; the
`gepa` package ships as a transitive dependency of `dspy`, no separate
install needed).

#### How to run

```bash
export GEMINI_API_KEY="your_key"
uv run optimize-gepa
```

`REFLECTION_MODEL` (env var) sets the reflection/mutation model — defaults to
`gemini-2.5-pro` when `LLM_PROVIDER=gemini`. `GEPA_AUTO` (`light`/`medium`/
`heavy`, default `light`) is the cost/quality dial — each step costs a real
`reflection_lm` call, so higher tiers cost more.

Requires `artifacts/trainset/trainset_domain.json` from `uv run gen-trainset`
first.

Output: `artifacts/prompts/extractor_optimized_gepa.json`

#### When GEPA > SIMBA

Prefer GEPA when you can afford a larger reflection budget and want a
broader search than SIMBA's local mini-batch ascent — e.g. a harder
extraction task, or a larger/noisier trainset where SIMBA's cheap local
iteration plateaus. For this project's small, well-behaved trainset, SIMBA's
cheaper mini-batch approach is a reasonable default; GEPA is here so the
comparison is real rather than assumed.

### TextGrad (alternative)

`steps/optimize_prompts_textgrad.py`

Alternative prompt optimizer using TextGrad instead of DSPy/SIMBA.

#### Why TextGrad?

TextGrad treats the system prompt as a differentiable Variable and
backpropagates textual feedback (LLM-generated "gradients") to improve it.
Both start from the same initial prompt and optimize the same domain
extractor.

TextGrad doesn't support Gemini natively, so this script wraps LiteLLM in a
custom `EngineLM` implementation (`GeminiEngine`) — TextGrad calls it through
both `__call__` and `generate`, both handled. This means `GEMINI_API_KEY`
works the same way here as in the rest of the pipeline.

#### How to run

```bash
export GEMINI_API_KEY="your_key"
uv run steps/optimize_prompts_textgrad.py
```

`uv run` reads the inline PEP 723 metadata block at the top of the script and
installs `textgrad`, `litellm`, and `python-dotenv` into an isolated
environment automatically — no manual `pip install`, no separate requirements
file to maintain alongside `pyproject.toml`. This is deliberately **not** a
registered `pyproject.toml` script — a console-script entry point would run
inside the main project's `uv`-managed environment, which doesn't (and
shouldn't) include `textgrad`.

Requires `artifacts/trainset/trainset_domain.json` from `uv run gen-trainset`
first.

Output: `artifacts/prompts/extractor_optimized_textgrad.json`

#### When TextGrad > SIMBA/GEPA

Use TextGrad when optimizing a node in a multi-step pipeline — e.g.
propagating Jordan's low rating backward through the full
collect → extract → present chain — since TextGrad backpropagates through
arbitrary text variables, not just a single prompt. For a single prompt with
a small dataset, SIMBA (or GEPA, for a broader search) is simpler.

## 2. Comparing optimizers

`steps/compare_optimizers.py`, run via `uv run compare-optimizers`.

Discovers every `artifacts/prompts/extractor_optimized*.json` present and
builds one table from each optimizer's own recorded `optimization_log`. This
is valid apples-to-apples specifically *because* of §1's shared
Signature/trainset/metric — no reconstruction or re-running needed for a fair
comparison. An experimental `--rescore` flag attempts a live re-score with a
single shared baseline pass instead of trusting each optimizer's own
independently-sampled baseline; it falls back to recorded values per artifact
if that isn't possible (e.g. for the TextGrad artifact, which isn't a DSPy
program).

Output: `artifacts/prompts/optimizer_comparison.md`

## 3. Context architecture: ACE two-layer / MIPROv2 + ACE

Reference implementations exploring how a SIMBA/MIPROv2-optimized prompt
structure and an ACE playbook could compose in production. Not wired into
the main pipeline — both require the external `ace` package
(`ace-agent/ace`, see `README.md` for install instructions).

- `lib/ace_two_layer.py` — Layer 1: a global playbook trained offline on a
  representative set of domain tasks. Layer 2: a per-user playbook that
  evolves online. New users cold-start from a copy of the global playbook;
  strategies that prove useful across multiple users get promoted back up
  from user → global.
- `lib/mipro_ace_pipeline.py` — MIPROv2 optimizes prompt *structure*
  (instructions + few-shot examples, offline, one-time); ACE fills that
  structure with evolving domain *knowledge* (strategies, failure patterns,
  online, continuous). Explicit token-budget split: ~20% for the MIPROv2
  structure (read-only anchor), ~80% for the ACE playbook (evolving).
