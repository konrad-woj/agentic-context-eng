"""
MIPROv2 + ACE — Integracja dwóch warstw optymalizacji

Architektura:
  Warstwa 1 (MIPROv2, offline, jednorazowo):
    → optymalizuje STRUKTURĘ promptu: instrukcje + few-shot examples
    → wymaga: labeled dataset (50-200 przykładów)
    → wynik: zoptymalizowany DSPy program zapisany do pliku

  Warstwa 2 (ACE, online, ciągłe):
    → wypełnia tę strukturę WIEDZĄ DZIEDZINOWĄ: strategie, wzorce porażek
    → wymaga: tylko sygnał feedbacku z wykonania (bez labeled data)
    → wynik: ewoluujący playbook per-user

Kluczowy podział token budget:
  - instrukcje MIPROv2 (struktura):  ~20% kontekstu  → read-only
  - ACE playbook (wiedza):           ~80% kontekstu  → ewoluujący
"""

import json
from pathlib import Path
from typing import Callable

import dspy
from dspy.teleprompt import MIPROv2
from ace import ACE


# ─────────────────────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────────────────────

LM_MODEL          = "openai/gpt-4o"
MIPRO_SAVE_PATH   = "./optimized/mipro_program"
GLOBAL_PLAYBOOK   = "./playbooks/global.json"
USER_PLAYBOOKS    = "./playbooks/users/"

# Token budget — rozdziel świadomie między warstwy
TOTAL_TOKEN_BUDGET   = 80_000
MIPRO_BUDGET         = int(TOTAL_TOKEN_BUDGET * 0.20)   # struktura (read-only)
ACE_PLAYBOOK_BUDGET  = int(TOTAL_TOKEN_BUDGET * 0.80)   # wiedza (ewoluująca)


# ─────────────────────────────────────────────────────────────
# WARSTWA 1: MIPROv2 — optymalizacja struktury promptu
# ─────────────────────────────────────────────────────────────

class DomainSignature(dspy.Signature):
    """
    Zamień na opis swojego zadania.
    MIPROv2 będzie optymalizować tę instrukcję i few-shot examples.
    ACE będzie dodawać wiedzę dziedzinową ponad tę strukturę.
    """
    context    = dspy.InputField(desc="Kontekst zadania i strategie z ACE playbooka")
    user_input = dspy.InputField(desc="Zapytanie użytkownika")
    output     = dspy.OutputField(desc="Odpowiedź agenta")


class DomainAgent(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought(DomainSignature)

    def forward(self, context: str, user_input: str) -> dspy.Prediction:
        return self.predictor(context=context, user_input=user_input)


def run_mipro_optimization(
    trainset: list[dspy.Example],
    metric_fn: Callable,
    auto: str = "medium",          # "light" | "medium" | "heavy"
    force_rerun: bool = False,
) -> DomainAgent:
    """
    Uruchamia MIPROv2 na labeled datasecie.
    Wynik zapisuje do pliku — nie musisz uruchamiać przy każdym starcie.

    trainset:  lista dspy.Example z polami pasującymi do DomainSignature
    metric_fn: funkcja (example, prediction) -> float/bool
    """
    save_path = Path(MIPRO_SAVE_PATH)

    # Użyj zapisanego programu jeśli istnieje i nie wymuszasz ponownego treningu
    if save_path.with_suffix(".json").exists() and not force_rerun:
        print("Ładowanie zapisanego programu MIPROv2...")
        program = DomainAgent()
        program.load(str(save_path))
        return program

    print(f"Uruchamianie MIPROv2 (auto={auto})...")
    print(f"Szacowany koszt: ~$2-10 USD w zależności od rozmiaru datasetu")

    lm = dspy.LM(LM_MODEL)
    dspy.configure(lm=lm)

    optimizer = MIPROv2(
        metric=metric_fn,
        auto=auto,
        # Nie ustawiaj num_candidates / num_trials gdy auto != None
    )

    optimized = optimizer.compile(
        DomainAgent(),
        trainset=trainset,
        max_bootstrapped_demos=3,   # ile auto-generowanych few-shotów
        max_labeled_demos=4,        # ile labeled few-shotów z trainset
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(save_path))
    print(f"Program MIPROv2 zapisany: {save_path}.json")

    return optimized


def extract_mipro_instructions(optimized_program: DomainAgent) -> str:
    """
    Wyciąga zoptymalizowane instrukcje z DSPy programu.
    To będzie read-only anchor dla ACE playbooka.
    """
    predictor = optimized_program.predictor

    instructions = ""

    # Instrukcja z podpisu (signature)
    if hasattr(predictor, "signature") and predictor.signature.instructions:
        instructions += f"## Zoptymalizowane instrukcje\n{predictor.signature.instructions}\n\n"

    # Few-shot examples zoptymalizowane przez MIPROv2
    if hasattr(predictor, "demos") and predictor.demos:
        instructions += "## Zoptymalizowane przykłady (few-shot)\n"
        for i, demo in enumerate(predictor.demos[:3]):  # max 3 przykłady
            instructions += f"\n### Przykład {i+1}\n"
            if hasattr(demo, "user_input"):
                instructions += f"Input: {demo.user_input}\n"
            if hasattr(demo, "output"):
                instructions += f"Output: {demo.output}\n"

    return instructions.strip()


# ─────────────────────────────────────────────────────────────
# WARSTWA 2: ACE — ewoluująca wiedza dziedzinowa
# ─────────────────────────────────────────────────────────────

def build_ace_initial_playbook(mipro_instructions: str) -> dict:
    """
    Tworzy seed dla ACE playbooka na bazie wyników MIPROv2.
    MIPROv2 dostarcza strukturę → ACE ją wzbogaca w czasie.
    """
    return {
        "entries": [
            {
                "id":      "mipro_anchor",
                "source":  "mipro",           # read-only — nie nadpisuj
                "content": mipro_instructions,
                "helpful": 999,               # wysoki licznik = trudny do usunięcia
                "harmful": 0,
            }
        ],
        "token_budget": ACE_PLAYBOOK_BUDGET,
        "mipro_budget": MIPRO_BUDGET,
    }


def compose_final_context(
    mipro_instructions: str,
    ace_playbook: dict,
) -> str:
    """
    Składa finalny kontekst dla agenta:
      - MIPROv2 instrukcje (read-only, zawsze na górze)
      - ACE playbook (ewoluujące strategie domenowe, poniżej)

    Pilnuje separacji token budgetów między warstwami.
    """
    ace_entries = [
        e for e in ace_playbook.get("entries", [])
        if e.get("source") != "mipro"       # wyklucz anchor MIPROv2
        and e.get("helpful", 0) > e.get("harmful", 0)  # tylko net-positive wpisy
    ]

    ace_content = "\n".join(
        f"- {e['content']}" for e in ace_entries
    ) if ace_entries else "(brak zgromadzonych strategii — pierwsze uruchomienie)"

    return f"""
{mipro_instructions}

## Strategie z doświadczenia (ACE Playbook)
{ace_content}
""".strip()


def run_ace_session(
    user_id:       str,
    user_input:    str,
    mipro_program: DomainAgent,
    mipro_instructions: str,
    feedback_fn:   Callable[[str], bool],
    ace:           ACE,
) -> str:
    """
    Pełna sesja z użytkownikiem:
    1. Złóż kontekst (MIPROv2 struktura + ACE wiedza)
    2. Uruchom agenta przez DSPy
    3. Oceń wynik przez feedback_fn
    4. ACE zaktualizuje playbook na podstawie feedbacku
    """
    # Załaduj playbook użytkownika (cold start → global)
    user_pb = load_user_playbook(user_id)

    # Złóż kontekst
    context = compose_final_context(mipro_instructions, user_pb)

    # Uruchom agenta (DSPy używa struktury z MIPROv2)
    lm = dspy.LM(LM_MODEL)
    dspy.configure(lm=lm)
    result = mipro_program(context=context, user_input=user_input)
    output = result.output

    # Oceń wynik
    success = feedback_fn(output)

    # ACE aktualizuje playbook online na podstawie feedbacku
    ace_config = {
        "task_name":           "domain_agent",
        "playbook_token_budget": ACE_PLAYBOOK_BUDGET,
        "save_dir":            f"./runs/users/{user_id}/",
        "initial_playbook":    user_pb,
        "no_ground_truth":     True,    # feedback bez labeled data
    }

    ace_result = ace.run_online(
        task={
            "input":       user_input,
            "output":      output,
            "context":     context,
            "success":     success,
        },
        config=ace_config,
    )

    # Zapisz zaktualizowany playbook (z ochroną anchora MIPROv2)
    updated_pb = ace_result["updated_playbook"]
    updated_pb = protect_mipro_anchor(updated_pb, mipro_instructions)
    save_user_playbook(user_id, updated_pb)

    return output


def protect_mipro_anchor(playbook: dict, mipro_instructions: str) -> dict:
    """
    Upewnia się że ACE nie nadpisał anchora z MIPROv2.
    Jeśli anchor zniknął — przywraca go.
    """
    entries = playbook.get("entries", [])
    has_anchor = any(e.get("source") == "mipro" for e in entries)

    if not has_anchor:
        entries.insert(0, {
            "id":      "mipro_anchor",
            "source":  "mipro",
            "content": mipro_instructions,
            "helpful": 999,
            "harmful": 0,
        })
        playbook["entries"] = entries

    return playbook


# ─────────────────────────────────────────────────────────────
# ZARZĄDZANIE PLAYBOKAMI (z poprzedniego pliku, skrócone)
# ─────────────────────────────────────────────────────────────

def load_user_playbook(user_id: str) -> dict:
    path = Path(USER_PLAYBOOKS) / f"{user_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    # Cold start: inicjalizuj z global playbooka
    if Path(GLOBAL_PLAYBOOK).exists():
        print(f"Cold start {user_id} → seed z global playbooka")
        return json.loads(Path(GLOBAL_PLAYBOOK).read_text())
    return {"entries": [], "token_budget": ACE_PLAYBOOK_BUDGET}


def save_user_playbook(user_id: str, playbook: dict) -> None:
    path = Path(USER_PLAYBOOKS) / f"{user_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(playbook, indent=2))


# ─────────────────────────────────────────────────────────────
# PRZYKŁAD UŻYCIA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Faza 1: MIPROv2 (raz, offline) ──────────────────────

    # Twój labeled dataset — minimum ~50 przykładów
    trainset = [
        dspy.Example(
            context="",                            # początkowo pusty
            user_input="Przykładowe zapytanie #1",
            output="Oczekiwana odpowiedź #1",
        ).with_inputs("context", "user_input"),
        # ... więcej przykładów
    ]

    def my_metric(example, prediction, trace=None) -> bool:
        """Twoja metryka sukcesu dla MIPROv2."""
        return prediction.output == example.output

    optimized_program  = run_mipro_optimization(trainset, my_metric, auto="medium")
    mipro_instructions = extract_mipro_instructions(optimized_program)

    # Zbuduj global playbook na bazie MIPROv2 (seed dla nowych userów)
    global_pb = build_ace_initial_playbook(mipro_instructions)
    Path(GLOBAL_PLAYBOOK).parent.mkdir(parents=True, exist_ok=True)
    Path(GLOBAL_PLAYBOOK).write_text(json.dumps(global_pb, indent=2))
    print("Global playbook zainicjowany z MIPROv2 anchor")

    # ── Faza 2: ACE (ciągłe, online per-user) ───────────────

    ace = ACE(
        api_provider="openai",
        generator_model="gpt-4o",
        reflector_model="gpt-4o",
        curator_model="gpt-4o",
        max_tokens=4096,
    )

    def my_feedback(output: str) -> bool:
        """Twój sygnał feedbacku — bez labeled data."""
        return len(output) > 10   # zastąp prawdziwą logiką

    # Sesja użytkownika
    response = run_ace_session(
        user_id="user_123",
        user_input="Konkretne zapytanie użytkownika",
        mipro_program=optimized_program,
        mipro_instructions=mipro_instructions,
        feedback_fn=my_feedback,
        ace=ace,
    )
    print("Odpowiedź agenta:", response)
