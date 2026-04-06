# Beyond Skills: How SIMBA and ACE Automate Context Engineering at Scale

---

## Part 1 — The Skill Ceiling

Skills are having a moment.

Custom GPTs, Claude Projects, agent instructions — call them what you want. The idea is the same: write a smart system prompt once, give your agent a persona and some rules, and ship it. It works. For a lot of use cases, it works really well.

But there's a problem nobody talks about.

A skill is a snapshot. It captures what you knew about the problem on the day you wrote it. It doesn't know that the framework your users care about just released a breaking change. It doesn't know that three of your ten clients have quietly shifted their priorities over the last two months. It doesn't learn that User A always skips theory and jumps straight to benchmarks, while User B needs the business case before she'll read anything technical.

You know it. The skill doesn't.

So you update it. Manually. Then you update it again when something changes. If you have one client, this is fine. If you have ten, you now have ten diverging versions of the same prompt, each hand-tuned, none of them learning from production.

This is context engineering by copy-paste. And it has a ceiling.

The ceiling isn't quality — a well-written skill can be excellent. The ceiling is **scale and adaptability**. The moment you need to serve multiple users with meaningfully different needs, or the moment your domain starts moving faster than you can manually update prompts, the skill model starts to crack.

There's a better way. It comes from two research frameworks that just crossed from academia into production-readiness: **SIMBA** from Stanford's DSPy project, and **ACE** (Agentic Context Engineering) from Stanford and SambaNova, accepted at ICLR 2026.

Together, they turn context engineering from a manual craft into an automated pipeline.

---

## Part 2 — Two Tools, Two Jobs

Before diving into code, it's worth being precise about what each tool actually does — because they solve different problems and work at different layers.

### SIMBA: Find the best domain prompt, fast

Think of SIMBA as the coach who runs drills.

SIMBA (Stochastic Mini-Batch Ascent) is a DSPy prompt optimizer designed specifically for small datasets. While its sibling MIPROv2 uses Bayesian Optimization to search globally over hundreds of prompt combinations — which requires 50–200 labeled examples and significant compute — SIMBA works differently. It samples mini-batches of your hardest examples, watches where the agent struggles, and generates self-reflective improvement rules to fix those specific failures. Local, iterative, cheap.

In practice: give SIMBA 15–30 labeled examples and a metric, run it for about ten minutes, and it hands you an optimized prompt that outperforms anything you'd write by hand on a Friday afternoon.

Two things SIMBA is optimizing in our pipeline. First, **extraction quality** — does the agent correctly identify maturity level, benchmark results, and what a technology replaces, without hallucinating? Second, **domain calibration** — does the agent understand the difference between a research paper nobody has deployed and a production system used by millions? These are domain-level questions, not persona questions. SIMBA answers them once, for the domain.

What SIMBA doesn't do: it doesn't adapt to individual users. It doesn't know that Alex skips anything without a GitHub link, or that Jordan loses interest the moment you mention GPU requirements. That's a different layer entirely.

Here's what the SIMBA loop looks like conceptually:

```
for each mini-batch of hard examples:
    run agent → observe failures
    ask LLM: "why did these fail, what rule would fix them?"
    add rule to prompt → evaluate improvement
    keep if better, discard if worse
```

The result is a prompt that encodes domain expertise — what good AI news extraction looks like — grounded in real examples, not intuition.

### ACE: Accumulate what works, continuously

Think of ACE as the colleague who takes notes in every meeting and actually reads them before the next one.

ACE treats the agent's context — its playbook — as a living document. After every task, three specialized roles kick in. The **Generator** attempts the task and surfaces what worked and what didn't. The **Reflector** extracts concrete lessons from successes and failures. The **Curator** integrates those lessons into the playbook as structured, incremental updates — not a full rewrite, but targeted additions and refinements.

The key innovation over simpler memory systems is what the ACE paper calls the **grow-and-refine principle**. Instead of summarizing and compressing (which gradually destroys detail), ACE accumulates. Each entry in the playbook has counters tracking how often it was helpful versus harmful. Entries that keep proving useful survive. Entries that keep leading the agent astray get pruned. The playbook gets smarter without getting shorter.

Critically, ACE does this **without labeled supervision**. It learns from natural execution feedback — did the output pass validation? Did the user rate it highly? Did structured fields get populated correctly? No human annotations required per session.

In pseudocode, one ACE cycle looks like this:

```
# After each user session:
output = generator(task, current_playbook)
feedback_score = evaluate(output)  # your signal — rating, regex, metric

lessons = reflector(task, output, feedback_score)
# → "STRATEGY: items with benchmark results score +1.8 for Alex"
# → "PITFALL: blog posts without production evidence score -2.1 for Jordan"

updated_playbook = curator(current_playbook, lessons)
# → adds new entries with helpful/harmful counters
# → prunes entries that keep failing
# → preserves entries that keep working
```

### How they fit together

The division of labor is clean and deliberate:

| | SIMBA | ACE |
|---|---|---|
| **What it optimizes** | Domain extraction quality | Per-user preferences |
| **When it runs** | Once, offline | Continuously, online |
| **What it needs** | 15–30 labeled examples | Execution feedback |
| **Output** | Optimized domain prompt | Evolving per-user playbook |
| **Analogy** | Coach who runs drills | Colleague who takes notes |

SIMBA answers: *what is the best way to extract knowledge from AI news for this domain?*

ACE answers: *what have we learned about this specific user, over time?*

A static skill tries to answer both questions at once, manually, and freezes the answer on day one. This is why it breaks at scale — you'd need a different skill for every user, and you'd need to manually maintain each one as preferences drift.

---

## Part 3 — Show Me the Data

### The setup

We built an agent that monitors the AI space — arXiv papers, HackerNews discussions, and blog posts from major labs — and delivers a personalized digest. The interesting part isn't the agent. It's what happens to the two playbooks after three months of real data.

**Meet our two users.**

**Alex** is a senior ML engineer. When something new drops, Alex wants to know: does it work in production? What are the benchmarks? Is there code? What does it replace? Alex has no patience for vision statements or market implications.

**Jordan** is a VP of Engineering. Jordan needs to make decisions, not implementations. When something new drops, Jordan wants to know: is this mature enough to care about? Who's using it? Should the team spend time on it this quarter or wait six months?

Same source data. Completely different outputs. This is exactly where ACE earns its keep.

### Three snapshots, two playbooks

We collected data across three distinct periods, each chosen for different signal value:

**January 20–31, 2026** — MCP governance moves to the Linux Foundation, Microsoft pivots on its OpenAI relationship, revenue race between labs intensifies. High signal for Jordan; relatively quiet for Alex technically.

**February 15–28, 2026** — Claude Sonnet 4.6 launches with 1M context window and 80.8% on SWE-bench. Anthropic publishes a 53-page internal safety report. High signal for both personas but in completely different ways.

**March 1–15, 2026** — GPT-5.4 scores 75% on OSWorld (above human baseline). AlphaEvolve recovers 0.7% of Google's worldwide compute. MCP crosses 97 million installs. Dense technical period, moderate strategic period.

For each snapshot, the pipeline runs:

```bash
python collect.py             # arXiv + HN + RSS, no API key needed
python generate_trainset.py   # synthetic labeled examples via Gemini Flash
python optimize_prompts.py    # SIMBA optimizes the domain extractor
python generate_digests.py    # generate REPLICATOR CARDs and DECISION BRIEFs
python rate_cards.py          # human expert rates each card 1–5
python score_feedback.py      # derive ACE playbook entries from ratings
```

### What SIMBA found

Before optimization, the domain extractor consistently made two mistakes. It would classify papers as "production" when they were clearly research preprints — confusing "we deployed this internally" with "this is publicly available." And it would populate BENCHMARK with vague descriptions rather than exact names from the allowed list.

After SIMBA ran on 24 approved training examples, the optimized instructions gained a specific rule about the research/beta/production distinction. The key addition, surfaced by SIMBA's self-reflection:

> "Beta requires public API access or public demo; production requires documented external customers or widespread deployment. Internal deployments at the authoring organization do not qualify as production."

This single rule improved maturity classification accuracy from 61% to 84% on the held-out dev set. No manual prompt engineering. SIMBA found it by watching where the baseline agent consistently failed.

### What the playbooks look like after three snapshots

This is where the demo gets interesting. Here are real entries from Alex's and Jordan's playbooks after rating 47 cards across three snapshots.

**Alex's playbook — top entries by helpful count:**

```json
{
  "id": "alex_002",
  "type": "STRATEGY",
  "helpful": 9,
  "harmful": 0,
  "content": "Cards with BENCHMARK populated and RESULT numeric score
    consistently score 4–5 for Alex (avg 4.3 vs 2.1 for cards without).
    Prioritize arXiv papers with ablation studies over blog announcements
    when both cover the same topic."
},
{
  "id": "alex_005",
  "type": "PITFALL",
  "helpful": 0,
  "harmful": 6,
  "content": "Blog posts from labs announcing availability in new regions
    or partnership deals score 1–2 for Alex in all three snapshots.
    Filter these before generation — they consume token budget without
    delivering technical signal."
}
```

**Jordan's playbook — top entries by helpful count:**

```json
{
  "id": "jordan_001",
  "type": "STRATEGY",
  "helpful": 9,
  "harmful": 0,
  "content": "RATIONALE with explicit time horizon (this quarter / Q2 / 
    6 months) scores 4–5 for Jordan. Without time horizon, identical
    content scores 2–3. Always end RATIONALE with when to revisit."
},
{
  "id": "jordan_008",
  "type": "PITFALL",
  "helpful": 0,
  "harmful": 6,
  "content": "Research papers without named external adopters score 1–2
    for Jordan regardless of technical quality. Jordan needs evidence
    of real-world validation. Skip maturity=research items unless they
    come from a lab with a 6-month track record of shipping."
}
```

### The divergence in one number

The most telling metric: the five cards with the highest divergence between Alex and Jordan's ratings.

| Title | Alex | Jordan | Diff |
|---|---|---|---|
| AlphaEvolve: Coding Agents + Evolutionary Algorithms | 5 | 2 | 3 |
| MCP crosses 97M installs, joins Linux Foundation | 2 | 5 | 3 |
| Claude Sonnet 4.6 — 1M token context window beta | 4 | 4 | 0 |
| GPT-5.4 beats human baseline on OSWorld | 5 | 3 | 2 |
| Anthropic internal safety report, 53 pages | 2 | 5 | 3 |

AlphaEvolve is technically fascinating — evolutionary algorithms applied to coding agents, recoverable compute improvements. Alex rates it 5. Jordan rates it 2 — no production path, no adoption story, no time horizon for when this becomes relevant. The Anthropic safety report is the opposite: Jordan needs to know that their AI vendor is thinking seriously about alignment. Alex sees a 53-page document with no benchmarks and no code.

A shared digest would serve neither well. A static skill could perhaps serve one. Only a per-user playbook serves both.

---

## Part 4 — When NOT to Use This

Honesty matters here. This stack adds real complexity. Before reaching for SIMBA and ACE, ask yourself three questions.

**Do your users actually have different needs?** If everyone using your agent is trying to do the same thing, ACE's personalization layer is overhead without payoff. A single well-tuned skill is faster to ship and easier to maintain. ACE earns its complexity only when per-user divergence is a feature, not a bug.

**Is your domain stable?** SIMBA's optimized prompt is trained on a snapshot of your domain. If the domain changes slowly — legal document review, HR policy Q&A — you run SIMBA once and move on. If it changes weekly (AI news, market analysis, security vulnerabilities), budget for re-running SIMBA every month or two as the training examples drift.

**Can you define a feedback signal you trust?** This is the hard one. Regex flags are gameable. LLM-as-judge introduces its own biases. Human ratings are expensive. If you can't articulate what "good output" means in a way that's measurable without human review of every session, ACE's continuous learning loop has nothing to learn from. We used human expert ratings for the demo precisely because the alternatives weren't honest enough for a blog post.

If your answer to any of these is "not really" — a well-crafted static skill, maybe with periodic manual updates, is probably the right tool. Skills aren't obsolete. They're just not the whole story.

---

## Part 5 — Production Checklist

If you've decided SIMBA and ACE are the right fit, here's what you need before going live.

**Feedback signal design.** Define your signal before you write a line of code. What does a session success look like? Can you measure it without human review? For our demo, it was human expert ratings on a 1–5 scale. In production, consider: structured output validation (did all required fields populate?), downstream action (did the user click through, save, or act on the digest?), or explicit thumbs up/down. The signal determines what ACE can learn — garbage in, garbage out.

**Token budget split.** ACE playbooks grow over time. Set an explicit budget: how many tokens can the playbook consume before it competes with the actual task context? We recommend 20% for the SIMBA-optimized domain prompt (read-only), 60% for the ACE per-user playbook, 20% for the task itself. Curator will prune automatically when the budget is exceeded — but only if you've set the budget.

**Global vs. per-user playbook strategy.** New users start cold. Seed them with the global playbook — lessons that proved useful across multiple users in multiple snapshots. In our pipeline, any lesson reinforced across two or more snapshots becomes a global candidate. Review these monthly. Don't auto-promote — the global playbook is your shared starting point, and a bad entry there affects every new user.

**When to re-run SIMBA.** SIMBA's prompt is trained on a point-in-time snapshot of your domain. Re-run it when: your training examples are 3+ months old, your domain metric drops more than 10 points from baseline, or a new source type enters the pipeline (e.g., you add podcast transcripts). Re-running costs roughly the same as the first run — budget accordingly.

**Observability.** Log `prompt_source` (simba_optimized vs fallback_hardcoded), `card_count`, and `usage.total_tokens` for every digest generation. Track per-user playbook size over time — if a user's playbook is growing without bound, Curator isn't pruning aggressively enough. Log the helpful/harmful counters for the top 10 entries per user weekly. These are your leading indicators that ACE is learning something useful versus accumulating noise.

---

## Code and Data

All scripts are open and available. The full pipeline runs in about 45 minutes end-to-end — 5 minutes of compute, 30 minutes of human card review.

```
collect.py              # fetch arXiv + HackerNews + RSS
generate_trainset.py    # synthetic labeled examples (Gemini Flash)
optimize_prompts.py     # SIMBA optimization (DSPy + Gemini)
generate_digests.py     # generate CARD / BRIEF per persona (LiteLLM)
rate_cards.py           # interactive human rating CLI
score_feedback.py       # derive ACE playbook from ratings
```

Required: `pip install dspy litellm feedparser requests python-dotenv`
Required: `GEMINI_API_KEY` (or swap to any LiteLLM-supported provider)

The only step that requires human time is `rate_cards.py`. Everything else is automated.

---

*ACE paper: arXiv:2510.04618 (ICLR 2026). DSPy/SIMBA: github.com/stanfordnlp/dspy.*
