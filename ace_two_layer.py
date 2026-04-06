"""
ACE - Architektura dwuwarstwowa: Global Playbook + Per-User Playbook
Rozwiązanie cold start problem.

Warstwa 1: Global playbook - trenowany offline na zbiorze zadań domeny
Warstwa 2: Per-user playbook - ewoluuje online dla każdego użytkownika osobno
"""

import json
import os
from pathlib import Path
from ace import ACE
from utils import initialize_clients


# ── Konfiguracja ──────────────────────────────────────────────────────────────

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

GLOBAL_PLAYBOOK_PATH = "./playbooks/global_playbook.json"
USER_PLAYBOOKS_DIR   = "./playbooks/users/"

# Próg: ile użytkowników musi mieć daną strategię, żeby awansowała do globalnego
PROMOTION_THRESHOLD = 3


# ── Inicjalizacja ACE ──────────────────────────────────────────────────────────

def create_ace_system(api_provider: str = "openai") -> ACE:
    return ACE(
        api_provider=api_provider,
        generator_model="gpt-4o",   # zamień na swój model
        reflector_model="gpt-4o",
        curator_model="gpt-4o",
        max_tokens=4096,
    )


# ── Warstwa 1: Global Playbook (offline) ──────────────────────────────────────

def build_global_playbook(domain_tasks: list[dict], ace: ACE) -> dict:
    """
    Uruchamia ACE offline na reprezentatywnym zbiorze zadań domeny.
    Wynik zapisuje jako global_playbook.json.

    domain_tasks: lista słowników {"input": ..., "expected_output": ..., "feedback_fn": ...}
    """
    config = {**ACE_CONFIG, "save_dir": "./runs/global/"}

    print("Budowanie global playbooka (offline)...")
    result = ace.run_offline(
        tasks=domain_tasks,
        config=config,
    )

    global_playbook = result["final_playbook"]
    Path(GLOBAL_PLAYBOOK_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(GLOBAL_PLAYBOOK_PATH, "w") as f:
        json.dump(global_playbook, f, indent=2)

    print(f"Global playbook zapisany: {len(global_playbook)} wpisów")
    return global_playbook


def load_global_playbook() -> dict:
    if not Path(GLOBAL_PLAYBOOK_PATH).exists():
        raise FileNotFoundError(
            "Global playbook nie istnieje. Uruchom najpierw build_global_playbook()."
        )
    with open(GLOBAL_PLAYBOOK_PATH) as f:
        return json.load(f)


# ── Warstwa 2: Per-User Playbook (online) ─────────────────────────────────────

def get_user_playbook_path(user_id: str) -> Path:
    return Path(USER_PLAYBOOKS_DIR) / f"{user_id}.json"


def load_user_playbook(user_id: str) -> dict:
    """
    Ładuje playbook użytkownika.
    Jeśli użytkownik jest nowy (cold start) → startuje z kopii global playbooka.
    """
    path = get_user_playbook_path(user_id)

    if path.exists():
        with open(path) as f:
            return json.load(f)
    else:
        print(f"Nowy użytkownik {user_id} → inicjalizacja z global playbooka (cold start fix)")
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
    Uruchamia jedną sesję dla użytkownika.
    ACE adaptuje jego playbook online na podstawie feedbacku z zadania.

    task: {"input": ..., "feedback_fn": callable}
    """
    user_playbook = load_user_playbook(user_id)

    config = {
        **ACE_CONFIG,
        "save_dir": f"./runs/users/{user_id}/",
        "initial_playbook": user_playbook,  # seed z global lub poprzedniej sesji
    }

    result = ace.run_online(
        task=task,
        config=config,
    )

    # Zapisz zaktualizowany playbook użytkownika
    updated_playbook = result["updated_playbook"]
    save_user_playbook(user_id, updated_playbook)

    return result


# ── Promocja strategii: user → global ────────────────────────────────────────

def promote_strategies_to_global(min_users: int = PROMOTION_THRESHOLD) -> None:
    """
    Przegląda playbooki wszystkich użytkowników.
    Jeśli dana strategia (po fingerprint treści) pojawia się u >= min_users
    użytkowników → awansuje do global playbooka.

    Prosta heurystyka: porównanie po kluczach strategii.
    W produkcji warto użyć embeddingów do semantic dedup.
    """
    user_dir = Path(USER_PLAYBOOKS_DIR)
    if not user_dir.exists():
        return

    # Zbierz wszystkie strategie ze wszystkich userów
    strategy_counts: dict[str, int] = {}
    strategy_content: dict[str, dict] = {}

    for user_file in user_dir.glob("*.json"):
        with open(user_file) as f:
            playbook = json.load(f)

        for entry in playbook.get("entries", []):
            key = entry.get("id") or entry.get("content", "")[:80]
            strategy_counts[key] = strategy_counts.get(key, 0) + 1
            strategy_content[key] = entry

    # Zbierz kandydatów do promocji
    promoted = [
        strategy_content[k]
        for k, count in strategy_counts.items()
        if count >= min_users
    ]

    if not promoted:
        print("Brak nowych strategii do promocji.")
        return

    # Załaduj globalny playbook i dodaj nowe wpisy (z dedup)
    global_pb = load_global_playbook()
    existing_ids = {e.get("id") for e in global_pb.get("entries", [])}

    added = 0
    for strategy in promoted:
        if strategy.get("id") not in existing_ids:
            global_pb.setdefault("entries", []).append(strategy)
            existing_ids.add(strategy.get("id"))
            added += 1

    with open(GLOBAL_PLAYBOOK_PATH, "w") as f:
        json.dump(global_pb, f, indent=2)

    print(f"Promocja zakończona: +{added} nowych strategii w global playbooku")


# ── Przykład użycia ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    ace = create_ace_system(api_provider="openai")

    # 1. Zbuduj global playbook raz (offline, na starcie projektu)
    domain_tasks = [
        {
            "input": "Przykładowe zadanie z Twojej domeny #1",
            "expected_output": "oczekiwany wynik",
            "feedback_fn": lambda output, expected: output == expected,
        },
        # ... więcej zadań reprezentatywnych dla domeny
    ]
    # build_global_playbook(domain_tasks, ace)  # odkomentuj przy pierwszym uruchomieniu

    # 2. Obsługa sesji użytkownika (online)
    user_id = "user_123"
    task = {
        "input": "Konkretne zadanie tego użytkownika",
        "feedback_fn": lambda output: len(output) > 0,  # Twój sygnał feedbacku
    }
    result = run_user_session(user_id, task, ace)
    print("Wynik sesji:", result.get("output"))

    # 3. Cyklicznie (np. co noc) promuj popularne strategie do global playbooka
    promote_strategies_to_global(min_users=PROMOTION_THRESHOLD)
