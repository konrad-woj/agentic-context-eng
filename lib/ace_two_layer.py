"""
ACE — Two-layer architecture: Global Playbook + Per-User Playbook
Solves the cold start problem.

Layer 1: Global playbook — trained offline on a representative set of domain tasks
Layer 2: Per-user playbook — evolves online for each user independently
"""

import json
import os
from pathlib import Path
from ace import ACE
from utils import initialize_clients


# ── Configuration ─────────────────────────────────────────────────────────────

ACE_CONFIG = {
    "num_epochs": 1,
    "max_num_rounds": 3,
    "curator_frequency": 1,
    "eval_steps": 100,
    "online_eval_frequency": 15,
    "save_steps": 50,
    "playbook_token_budget": 80000,
    "task_name": "your_domain",
    "json_mode": False,
    "no_ground_truth": False,
}

_ROOT                = Path(__file__).parent.parent
GLOBAL_PLAYBOOK_PATH = str(_ROOT / "artifacts" / "playbooks" / "global_playbook.json")
USER_PLAYBOOKS_DIR   = str(_ROOT / "artifacts" / "playbooks" / "users")

# A strategy is promoted to the global playbook once this many users share it
PROMOTION_THRESHOLD = 3


# ── ACE initialization ────────────────────────────────────────────────────────

def create_ace_system(api_provider: str = "openai") -> ACE:
    return ACE(
        api_provider=api_provider,
        generator_model="gpt-4o",   # replace with your model
        reflector_model="gpt-4o",
        curator_model="gpt-4o",
        max_tokens=4096,
    )


# ── Layer 1: Global Playbook (offline) ───────────────────────────────────────

def build_global_playbook(domain_tasks: list[dict], ace: ACE) -> dict:
    """
    Runs ACE offline on a representative set of domain tasks.
    Saves result as global_playbook.json.

    domain_tasks: list of dicts {"input": ..., "expected_output": ..., "feedback_fn": ...}
    """
    config = {**ACE_CONFIG, "save_dir": "./runs/global/"}

    print("Building global playbook (offline)...")
    result = ace.run_offline(
        tasks=domain_tasks,
        config=config,
    )

    global_playbook = result["final_playbook"]
    Path(GLOBAL_PLAYBOOK_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_PLAYBOOK_PATH, "w") as f:
        json.dump(global_playbook, f, indent=2)

    print(f"Global playbook saved: {len(global_playbook)} entries")
    return global_playbook


def load_global_playbook() -> dict:
    if not Path(GLOBAL_PLAYBOOK_PATH).exists():
        raise FileNotFoundError(
            "Global playbook not found. Run build_global_playbook() first."
        )
    with open(GLOBAL_PLAYBOOK_PATH) as f:
        return json.load(f)


# ── Layer 2: Per-User Playbook (online) ──────────────────────────────────────

def get_user_playbook_path(user_id: str) -> Path:
    return Path(USER_PLAYBOOKS_DIR) / f"{user_id}.json"


def load_user_playbook(user_id: str) -> dict:
    """
    Loads a user's playbook.
    New users (cold start) start from a copy of the global playbook.
    """
    path = get_user_playbook_path(user_id)

    if path.exists():
        with open(path) as f:
            return json.load(f)
    else:
        print(f"New user {user_id} → initializing from global playbook (cold start fix)")
        global_pb = load_global_playbook()
        save_user_playbook(user_id, global_pb)
        return global_pb


def save_user_playbook(user_id: str, playbook: dict) -> None:
    path = get_user_playbook_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(playbook, f, indent=2)


def run_user_session(user_id: str, task: dict, ace: ACE) -> dict:
    """
    Runs one session for a user.
    ACE adapts the user's playbook online based on task feedback.

    task: {"input": ..., "feedback_fn": callable}
    """
    user_playbook = load_user_playbook(user_id)

    config = {
        **ACE_CONFIG,
        "save_dir": f"./runs/users/{user_id}/",
        "initial_playbook": user_playbook,   # seeded from global or previous session
    }

    result = ace.run_online(
        task=task,
        config=config,
    )

    # Persist the updated playbook for the next session
    updated_playbook = result["updated_playbook"]
    save_user_playbook(user_id, updated_playbook)

    return result


# ── Strategy promotion: user → global ────────────────────────────────────────

def promote_strategies_to_global(min_users: int = PROMOTION_THRESHOLD) -> None:
    """
    Scans all user playbooks.
    If a strategy (matched by content fingerprint) appears in >= min_users
    users → promotes it to the global playbook.

    Simple heuristic: match by strategy key prefix.
    For production, use embeddings for semantic deduplication.
    """
    user_dir = Path(USER_PLAYBOOKS_DIR)
    if not user_dir.exists():
        return

    # Collect all strategies across all users
    strategy_counts:  dict[str, int]  = {}
    strategy_content: dict[str, dict] = {}

    for user_file in user_dir.glob("*.json"):
        with open(user_file) as f:
            playbook = json.load(f)

        for entry in playbook.get("entries", []):
            key = entry.get("id") or entry.get("content", "")[:80]
            strategy_counts[key]  = strategy_counts.get(key, 0) + 1
            strategy_content[key] = entry

    promoted = [
        strategy_content[k]
        for k, count in strategy_counts.items()
        if count >= min_users
    ]

    if not promoted:
        print("No new strategies to promote.")
        return

    # Load global playbook and append new entries (with dedup)
    global_pb    = load_global_playbook()
    existing_ids = {e.get("id") for e in global_pb.get("entries", [])}

    added = 0
    for strategy in promoted:
        if strategy.get("id") not in existing_ids:
            global_pb.setdefault("entries", []).append(strategy)
            existing_ids.add(strategy.get("id"))
            added += 1

    with open(GLOBAL_PLAYBOOK_PATH, "w") as f:
        json.dump(global_pb, f, indent=2)

    print(f"Promotion complete: +{added} new strategies added to global playbook")


# ── Usage example ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ace = create_ace_system(api_provider="openai")

    # 1. Build global playbook once (offline, at project start)
    domain_tasks = [
        {
            "input": "Example domain task #1",
            "expected_output": "expected result",
            "feedback_fn": lambda output, expected: output == expected,
        },
        # ... more domain-representative tasks
    ]
    # build_global_playbook(domain_tasks, ace)  # uncomment on first run

    # 2. Handle a user session (online)
    user_id = "user_123"
    task = {
        "input": "Specific task for this user",
        "feedback_fn": lambda output: len(output) > 0,  # replace with your feedback signal
    }
    result = run_user_session(user_id, task, ace)
    print("Session output:", result.get("output"))

    # 3. Run periodically (e.g. nightly) to promote popular strategies to global playbook
    promote_strategies_to_global(min_users=PROMOTION_THRESHOLD)
