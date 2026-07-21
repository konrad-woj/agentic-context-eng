"""
Score Feedback — v2 (human ratings as ground truth)
=====================================================
Aggregates ratings from rate_cards.py and builds:
  1. Quality statistics per snapshot per persona
  2. ACE lessons — what the generator did well/poorly per persona
  3. Playbook diff — how Alex's and Jordan's playbooks diverged
  4. Promotion candidates — lessons ready for the global playbook

Ground truth source: ratings/feedback.json (expert ratings 1–5)

Usage:
  uv run score

Output:
  scores/summary.json
  scores/ace_lessons_alex.json
  scores/ace_lessons_jordan.json
  scores/playbook_diff.json      ← main artifact for the blog
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).parent.parent
RATINGS_DIR = ROOT / "artifacts" / "ratings"
DIGESTS_DIR = ROOT / "artifacts" / "digests"
SCORES_DIR  = ROOT / "artifacts" / "scores"

# Card is "useful" if rated above this threshold
USEFUL_THRESHOLD = 4
# Card is "weak" if rated at or below this threshold
WEAK_THRESHOLD   = 2
# A lesson must appear in at least this many snapshots to be promoted to the global playbook
PROMOTION_MIN_SNAPSHOTS = 2


# ── Load data ─────────────────────────────────────────────────────────────────

def load_feedback() -> dict:
    path = RATINGS_DIR / "feedback.json"
    if not path.exists():
        raise FileNotFoundError(
            "ratings/feedback.json not found\n"
            "Run first: uv run rate"
        )
    return json.loads(path.read_text())


def load_digest(snapshot_id: str, persona: str) -> dict | None:
    path = DIGESTS_DIR / f"snapshot_{snapshot_id}_{persona}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ── Per-card analysis ─────────────────────────────────────────────────────────

def analyze_card_patterns(
    rated_cards: list[dict],
    persona: str,
) -> dict:
    """
    For each rated card, analyzes what distinguishes it from others.
    Looks for patterns in fields that correlate with high vs low ratings.

    Returns dict:
      field_name → {"in_useful": count, "in_weak": count, "signal": float}
    where signal > 0 means the field correlates with usefulness.
    """
    rating_key = f"rating_{persona}"
    useful_key = f"useful_for_{persona}"

    useful_cards = [c for c in rated_cards if c.get(useful_key)]
    weak_cards   = [c for c in rated_cards
                    if (c.get(rating_key) or 0) <= WEAK_THRESHOLD]

    if not useful_cards and not weak_cards:
        return {}

    # Load full card data from digests to access all fields
    card_details: dict[str, dict] = {}
    for snap_id in set(c["snapshot_id"] for c in rated_cards):
        digest = load_digest(snap_id, persona)
        if not digest:
            continue
        for card in digest.get("cards", []):
            title = (card.get("TITLE") or card.get("title") or "").strip()[:80]
            key   = f"{snap_id}::{title.lower()}"
            card_details[key] = card

    # Which fields are populated (non-N/A) in useful vs weak cards
    field_stats: dict[str, dict] = defaultdict(lambda: {"in_useful": 0, "in_weak": 0})

    for card_ref in rated_cards:
        key    = card_ref["key"]
        rating = card_ref.get(rating_key) or 0
        detail = card_details.get(key, {})

        for field, value in detail.items():
            if str(value).strip() in ("", "N/A", "n/a"):
                continue
            field_upper = field.upper()
            if rating >= USEFUL_THRESHOLD:
                field_stats[field_upper]["in_useful"] += 1
            elif rating <= WEAK_THRESHOLD:
                field_stats[field_upper]["in_weak"] += 1

    # Signal: difference between proportion in useful vs weak cards
    n_useful = max(len(useful_cards), 1)
    n_weak   = max(len(weak_cards),   1)

    result = {}
    for field, counts in field_stats.items():
        prop_useful = counts["in_useful"] / n_useful
        prop_weak   = counts["in_weak"]   / n_weak
        result[field] = {
            "in_useful_cards": counts["in_useful"],
            "in_weak_cards":   counts["in_weak"],
            "signal":          round(prop_useful - prop_weak, 2),
        }

    return dict(sorted(result.items(), key=lambda x: abs(x[1]["signal"]), reverse=True))


# ── ACE lesson generation ─────────────────────────────────────────────────────

def generate_ace_lessons(
    rated_cards:    list[dict],
    field_patterns: dict,
    persona:        str,
    snapshot_id:    str,
) -> list[dict]:
    """
    Generates lessons for the ACE playbook from:
    - User ratings (ground truth)
    - Field patterns (correlations)

    Each lesson has type STRATEGY (do more of this) or PITFALL (avoid this).
    """
    rating_key = f"rating_{persona}"
    useful_key = f"useful_for_{persona}"

    rated = [c for c in rated_cards if c.get(rating_key) is not None]
    if not rated:
        return []

    useful = [c for c in rated if c.get(useful_key)]
    weak   = [c for c in rated if (c.get(rating_key) or 0) <= WEAK_THRESHOLD]
    avg    = sum(c[rating_key] for c in rated) / len(rated)

    lessons = []

    # Lesson 1: overall digest quality
    if avg >= 4.0:
        lessons.append({
            "type":        "STRATEGY",
            "snapshot_id": snapshot_id,
            "lesson":      (
                f"High overall quality (avg {avg:.1f}/5) — "
                f"current generator prompt is well-calibrated for this persona. "
                f"Maintain current item selection criteria."
            ),
            "evidence":    f"{len(useful)}/{len(rated)} cards rated useful (≥4)",
            "confidence":  "high" if len(rated) >= 5 else "low",
        })
    elif avg <= 2.5:
        lessons.append({
            "type":        "PITFALL",
            "snapshot_id": snapshot_id,
            "lesson":      (
                f"Low overall quality (avg {avg:.1f}/5) — "
                f"generator is producing items irrelevant to this persona. "
                f"Tighten source filtering before generation."
            ),
            "evidence":    f"{len(weak)}/{len(rated)} cards rated weak (≤2)",
            "confidence":  "high" if len(rated) >= 5 else "low",
        })

    # Lesson 2: fields that correlate with usefulness
    for field, stats in list(field_patterns.items())[:4]:
        signal = stats["signal"]

        if signal >= 0.4 and stats["in_useful_cards"] >= 2:
            lessons.append({
                "type":        "STRATEGY",
                "snapshot_id": snapshot_id,
                "lesson":      (
                    f"Cards with {field} populated score significantly higher "
                    f"(signal +{signal}). Prioritize items where {field} is available."
                ),
                "evidence":    (
                    f"{field} present in {stats['in_useful_cards']} useful cards "
                    f"vs {stats['in_weak_cards']} weak cards"
                ),
                "confidence":  "medium",
            })

        elif signal <= -0.3 and stats["in_weak_cards"] >= 2:
            lessons.append({
                "type":        "PITFALL",
                "snapshot_id": snapshot_id,
                "lesson":      (
                    f"Cards with {field} populated score lower for this persona "
                    f"(signal {signal}). This field may indicate content "
                    f"that doesn't match persona needs."
                ),
                "evidence":    (
                    f"{field} present in {stats['in_weak_cards']} weak cards "
                    f"vs {stats['in_useful_cards']} useful cards"
                ),
                "confidence":  "medium",
            })

    # Lesson 3: divergence insight (only when both personas are rated)
    divergent = [
        c for c in rated_cards
        if c.get("divergence", 0) >= 3
        and c.get(rating_key) is not None
    ]
    if divergent:
        top = sorted(divergent, key=lambda x: x["divergence"], reverse=True)[0]
        other_persona = "jordan" if persona == "alex" else "alex"
        other_rating  = top.get(f"rating_{other_persona}", "?")
        this_rating   = top.get(rating_key, "?")
        lessons.append({
            "type":        "STRATEGY",
            "snapshot_id": snapshot_id,
            "lesson":      (
                f"Strong persona divergence detected. "
                f"'{top['title'][:50]}' scored {this_rating}/5 for this persona "
                f"vs {other_rating}/5 for the other. "
                f"Per-user filtering is essential — a shared digest would serve neither well."
            ),
            "evidence":    f"Divergence score: {top['divergence']} points",
            "confidence":  "high",
        })

    return lessons


# ── Playbook diff ─────────────────────────────────────────────────────────────

def build_playbook_diff(
    alex_lessons:   list[dict],
    jordan_lessons: list[dict],
    feedback:       dict,
) -> dict:
    """
    Aggregates lessons across all snapshots.
    Lessons that repeat across multiple snapshots get higher confidence
    and become candidates for the global playbook.
    """

    def aggregate(lessons: list[dict]) -> list[dict]:
        # Group similar lessons by their first 60 characters
        groups: dict[str, list[dict]] = defaultdict(list)
        for lesson in lessons:
            key = lesson["lesson"][:60]
            groups[key].append(lesson)

        result = []
        for key, group in groups.items():
            snaps = list({lesson["snapshot_id"] for lesson in group})
            base  = group[0].copy()
            base["reinforced_by_snapshots"] = len(snaps)
            base["snapshot_ids"]            = snaps
            base["promote_to_global"]       = len(snaps) >= PROMOTION_MIN_SNAPSHOTS
            result.append(base)

        return sorted(result, key=lambda x: x["reinforced_by_snapshots"], reverse=True)

    alex_agg   = aggregate(alex_lessons)
    jordan_agg = aggregate(jordan_lessons)

    alex_global   = [l for l in alex_agg   if l["promote_to_global"]]
    jordan_global = [l for l in jordan_agg if l["promote_to_global"]]

    divergent_cards = feedback.get("most_divergent_cards", [])

    alex_strategies   = [l for l in alex_agg   if l["type"] == "STRATEGY"]
    jordan_strategies = [l for l in jordan_agg if l["type"] == "STRATEGY"]
    alex_pitfalls     = [l for l in alex_agg   if l["type"] == "PITFALL"]
    jordan_pitfalls   = [l for l in jordan_agg if l["type"] == "PITFALL"]

    n_snaps = len({
        l["snapshot_id"]
        for l in alex_lessons + jordan_lessons
    })

    return {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "snapshots_processed": n_snaps,
        "ground_truth_source": "human_expert_ratings_1_5",
        "rated_cards":         feedback.get("rated_cards", 0),

        "alex_playbook": {
            "persona":           "Alex — Senior ML Engineer",
            "stats":             feedback.get("stats_alex", {}),
            "strategies":        alex_strategies,
            "pitfalls":          alex_pitfalls,
            "all_lessons":       alex_agg,
            "promote_to_global": alex_global,
        },

        "jordan_playbook": {
            "persona":           "Jordan — VP of Engineering",
            "stats":             feedback.get("stats_jordan", {}),
            "strategies":        jordan_strategies,
            "pitfalls":          jordan_pitfalls,
            "all_lessons":       jordan_agg,
            "promote_to_global": jordan_global,
        },

        "divergence_analysis": {
            "most_divergent_cards": divergent_cards,
            "summary": (
                f"After {n_snaps} snapshots the playbooks have diverged. "
                f"Alex accumulated {len(alex_agg)} lessons "
                f"({len(alex_global)} ready for global playbook). "
                f"Jordan accumulated {len(jordan_agg)} lessons "
                f"({len(jordan_global)} ready for global playbook). "
                f"The most divergent cards — rated very differently by the two personas — "
                f"are the clearest evidence that a static shared digest would serve neither well."
            ),
        },

        "global_playbook_candidates": {
            "description": (
                f"Lessons reinforced across ≥{PROMOTION_MIN_SNAPSHOTS} snapshots. "
                "These are stable enough to seed the global playbook for new users."
            ),
            "alex":   alex_global,
            "jordan": jordan_global,
        },
    }


# ── ACE Playbook format ───────────────────────────────────────────────────────
#
# Compatible with ace-agent/ace:
# - entries: list of entries with id, content, helpful, harmful, source
# - helpful/harmful: counters used by the ACE Curator for pruning decisions
# - source: "global" (from SIMBA/global playbook) or "user" (from user ratings)
#
# Mapping from lessons:
#   STRATEGY lesson → helpful counter increments with each snapshot
#   PITFALL lesson  → harmful counter increments with each snapshot
#   promote_to_global → source="global_candidate"

def lessons_to_ace_playbook(
    lessons:  list[dict],
    persona:  str,
    feedback: dict,
) -> dict:
    """
    Translates lessons from generate_ace_lessons() into the ACE playbook format.

    ACE entry format:
    {
        "id":         str,          # unique identifier
        "source":     str,          # "user" | "global_candidate" | "global"
        "content":    str,          # lesson text — what the agent should do/avoid
        "helpful":    int,          # how many times this strategy proved helpful
        "harmful":    int,          # how many times this strategy proved harmful
        "type":       str,          # "STRATEGY" | "PITFALL"
        "confidence": str,          # "high" | "medium" | "low"
        "snapshots":  list[str],    # which snapshots this was observed in
    }
    """
    entries   = []
    seen_keys = set()

    for i, lesson in enumerate(lessons):
        content_key = lesson["lesson"][:60]
        if content_key in seen_keys:
            continue
        seen_keys.add(content_key)

        lesson_type = lesson.get("type", "STRATEGY")
        n_snapshots = lesson.get("reinforced_by_snapshots", 1)
        to_global   = lesson.get("promote_to_global", False)

        # helpful/harmful counters:
        # STRATEGY confirmed in N snapshots → helpful=N*3, harmful=0
        # PITFALL  confirmed in N snapshots → helpful=0,   harmful=N*3
        # Higher helpful = harder for the Curator to prune
        if lesson_type == "STRATEGY":
            helpful = n_snapshots * 3
            harmful = 0
        else:  # PITFALL
            helpful = 0
            harmful = n_snapshots * 3

        entries.append({
            "id":         f"{persona}_{i:03d}",
            "source":     "global_candidate" if to_global else "user",
            "content":    lesson["lesson"],
            "helpful":    helpful,
            "harmful":    harmful,
            "type":       lesson_type,
            "confidence": lesson.get("confidence", "medium"),
            "evidence":   lesson.get("evidence", ""),
            "snapshots":  lesson.get("snapshot_ids", [lesson.get("snapshot_id", "")]),
        })

    # Calibration meta-entry — agent knows how much data backs its playbook
    stats = feedback.get(f"stats_{persona}", {})
    if stats:
        entries.insert(0, {
            "id":       f"{persona}_meta",
            "source":   "user",
            "content":  (
                f"Calibration data: {feedback.get('rated_cards', 0)} cards rated by human expert. "
                f"Average score for this persona: {stats.get('avg', '?')}/5. "
                f"Useful cards (≥4): {stats.get('useful', 0)}. "
                f"Weak cards (≤2): {stats.get('useless', 0)}."
            ),
            "helpful":  feedback.get("rated_cards", 0),
            "harmful":  0,
            "type":     "META",
            "confidence": "high",
            "evidence": "human expert ratings",
            "snapshots": [],
        })

    return {
        "persona":          persona,
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "ground_truth":     "human_expert_ratings_1_5",
        "token_budget":     64000,
        "entry_count":      len(entries),
        "entries":          entries,
        "global_candidates": [
            e for e in entries if e["source"] == "global_candidate"
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    SCORES_DIR.mkdir(exist_ok=True)

    feedback    = load_feedback()
    rated_cards = feedback.get("cards", [])

    if not rated_cards:
        print("✗  No rated cards in feedback.json")
        print("   Run first: uv run rate")
        return

    print(f"\nScore Feedback v2 — human ratings as ground truth")
    print(f"{'═' * 60}")
    print(f"  Rated cards:  {feedback['rated_cards']}")
    print(f"  Rater:        {feedback['rater']}")
    print(f"{'═' * 60}\n")

    # Group cards per snapshot
    by_snapshot: dict[str, list[dict]] = defaultdict(list)
    for card in rated_cards:
        by_snapshot[card["snapshot_id"]].append(card)

    all_alex_lessons:   list[dict] = []
    all_jordan_lessons: list[dict] = []

    for snap_id, cards in sorted(by_snapshot.items()):
        print(f"  Snapshot {snap_id}  ({len(cards)} cards)")

        for persona, lessons_list in [
            ("alex",   all_alex_lessons),
            ("jordan", all_jordan_lessons),
        ]:
            patterns = analyze_card_patterns(cards, persona)
            lessons  = generate_ace_lessons(cards, patterns, persona, snap_id)
            lessons_list.extend(lessons)

            rating_key = f"rating_{persona}"
            rated_here = [c for c in cards if c.get(rating_key) is not None]
            if rated_here:
                avg = sum(c[rating_key] for c in rated_here) / len(rated_here)
                bar = "█" * int(avg * 2) + "░" * (10 - int(avg * 2))
                print(f"    {persona:6}  {bar}  {avg:.1f}/5"
                      f"  ({len(lessons)} lessons)")

        print()

    # Save raw lessons per persona (for debugging)
    for persona, lessons in [
        ("alex",   all_alex_lessons),
        ("jordan", all_jordan_lessons),
    ]:
        path = SCORES_DIR / f"ace_lessons_{persona}.json"
        path.write_text(json.dumps(lessons, indent=2, ensure_ascii=False))
        print(f"  ✓  {path.name}  ({len(lessons)} lessons)")

    # Playbook diff — divergence analysis (for blog post)
    diff = build_playbook_diff(
        all_alex_lessons,
        all_jordan_lessons,
        feedback,
    )
    diff_path = SCORES_DIR / "playbook_diff.json"
    diff_path.write_text(json.dumps(diff, indent=2, ensure_ascii=False))
    print(f"  ✓  {diff_path.name}")

    # ── ACE Playbooks — ace-agent/ace compatible format ──────────────────────
    #
    # Main pipeline artifact:
    # - generate_digests.py loads these playbooks as initial_playbook
    # - ACE Curator updates them after each session
    # - Entries with source="global_candidate" are promoted to the global
    #   playbook after verification via build_global_playbook() in ace_two_layer.py

    print(f"\n  ACE Playbooks (ace-agent/ace format):")
    for persona, lessons in [
        ("alex",   all_alex_lessons),
        ("jordan", all_jordan_lessons),
    ]:
        # Use aggregated lessons so reinforced_by_snapshots is populated
        agg_lessons = diff[f"{persona}_playbook"]["all_lessons"]

        playbook = lessons_to_ace_playbook(agg_lessons, persona, feedback)

        scores_path = SCORES_DIR / f"playbook_{persona}.json"
        scores_path.write_text(json.dumps(playbook, indent=2, ensure_ascii=False))

        # Write to artifacts/playbooks/ where ace_two_layer.py expects to find them
        pb_dir = ROOT / "artifacts" / "playbooks"
        pb_dir.mkdir(parents=True, exist_ok=True)
        pb_path = pb_dir / f"{persona}_initial.json"
        pb_path.write_text(json.dumps(playbook, indent=2, ensure_ascii=False))

        n_global = len(playbook["global_candidates"])
        n_user   = playbook["entry_count"] - n_global
        print(f"    {persona:6}  {playbook['entry_count']} entries "
              f"({n_user} user | {n_global} global candidates)")
        print(f"           → {scores_path}")
        print(f"           → {pb_path}  ← used by ace_two_layer.py")

    # Summary
    summary = {
        "generated_at":               datetime.now(timezone.utc).isoformat(),
        "rated_cards":                feedback["rated_cards"],
        "stats_alex":                 feedback.get("stats_alex", {}),
        "stats_jordan":               feedback.get("stats_jordan", {}),
        "lessons_alex":               len(all_alex_lessons),
        "lessons_jordan":             len(all_jordan_lessons),
        "global_candidates_alex":     len(diff["global_playbook_candidates"]["alex"]),
        "global_candidates_jordan":   len(diff["global_playbook_candidates"]["jordan"]),
        "playbook_format":            "ace-agent/ace compatible",
    }
    (SCORES_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print(f"\n{'═' * 60}")
    print(f"  Divergence summary:")
    print(f"  {diff['divergence_analysis']['summary']}")
    print(f"\n  Global playbook candidates:")
    print(f"    Alex:   {len(diff['global_playbook_candidates']['alex'])} lessons")
    print(f"    Jordan: {len(diff['global_playbook_candidates']['jordan'])} lessons")
    print(f"\n  Playbooks ready in playbooks/")
    print(f"  Connect them to ace_two_layer.py as initial_playbook")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
