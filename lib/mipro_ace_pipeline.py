"""
MIPROv2 + ACE — Two-layer optimization integration

Architecture:
  Layer 1 (MIPROv2, offline, one-time):
    → optimizes prompt STRUCTURE: instructions + few-shot examples
    → requires: labeled dataset (50–200 examples)
    → output: optimized DSPy program saved to disk

  Layer 2 (ACE, online, continuous):
    → fills that structure with DOMAIN KNOWLEDGE: strategies, failure patterns
    → requires: only a feedback signal from execution (no labeled data)
    → output: evolving per-user playbook

Token budget split:
  - MIPROv2 instructions (structure):  ~20% of context  → read-only
  - ACE playbook (knowledge):          ~80% of context  → evolving
"""

import json
from pathlib import Path
from typing import Callable

import dspy
from dspy.teleprompt import MIPROv2
from ace import ACE


# ── Configuration ─────────────────────────────────────────────────────────────

LM_MODEL         = "openai/gpt-4o"
_ROOT            = Path(__file__).parent.parent
MIPRO_SAVE_PATH  = str(_ROOT / "artifacts" / "optimized" / "mipro_program")
GLOBAL_PLAYBOOK  = str(_ROOT / "artifacts" / "playbooks" / "global.json")
USER_PLAYBOOKS   = str(_ROOT / "artifacts" / "playbooks" / "users")

# Token budget — allocate deliberately between layers
TOTAL_TOKEN_BUDGET  = 80_000
MIPRO_BUDGET        = int(TOTAL_TOKEN_BUDGET * 0.20)   # structure (read-only)
ACE_PLAYBOOK_BUDGET = int(TOTAL_TOKEN_BUDGET * 0.80)   # knowledge (evolving)


# ── Layer 1: MIPROv2 — prompt structure optimization ──────────────────────────

class DomainSignature(dspy.Signature):
    """
    Replace with your task description.
    MIPROv2 will optimize these instructions and few-shot examples.
    ACE will add domain knowledge on top of this structure.
    """
    context    = dspy.InputField(desc="Task context and strategies from the ACE playbook")
    user_input = dspy.InputField(desc="User query")
    output     = dspy.OutputField(desc="Agent response")


class DomainAgent(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought(DomainSignature)

    def forward(self, context: str, user_input: str) -> dspy.Prediction:
        return self.predictor(context=context, user_input=user_input)


def run_mipro_optimization(
    trainset:     list[dspy.Example],
    metric_fn:    Callable,
    auto:         str  = "medium",   # "light" | "medium" | "heavy"
    force_rerun:  bool = False,
) -> DomainAgent:
    """
    Runs MIPROv2 on a labeled dataset.
    Saves the result to disk — no need to re-run on every startup.

    trainset:  list of dspy.Example with fields matching DomainSignature
    metric_fn: function (example, prediction) -> float | bool
    """
    save_path = Path(MIPRO_SAVE_PATH)

    # Load saved program if it exists and re-run is not forced
    if save_path.with_suffix(".json").exists() and not force_rerun:
        print("Loading saved MIPROv2 program...")
        program = DomainAgent()
        program.load(str(save_path))
        return program

    print(f"Running MIPROv2 (auto={auto})...")
    print(f"Estimated cost: ~$2–10 USD depending on dataset size")

    lm = dspy.LM(LM_MODEL)
    dspy.configure(lm=lm)

    optimizer = MIPROv2(
        metric=metric_fn,
        auto=auto,
        # Do not set num_candidates / num_trials when auto is set
    )

    optimized = optimizer.compile(
        DomainAgent(),
        trainset=trainset,
        max_bootstrapped_demos=3,   # auto-generated few-shots
        max_labeled_demos=4,        # labeled few-shots from trainset
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(save_path))
    print(f"MIPROv2 program saved: {save_path}.json")

    return optimized


def extract_mipro_instructions(optimized_program: DomainAgent) -> str:
    """
    Extracts optimized instructions from the DSPy program.
    This becomes the read-only anchor for the ACE playbook.
    """
    predictor    = optimized_program.predictor
    instructions = ""

    if hasattr(predictor, "signature") and predictor.signature.instructions:
        instructions += f"## Optimized instructions\n{predictor.signature.instructions}\n\n"

    if hasattr(predictor, "demos") and predictor.demos:
        instructions += "## Optimized few-shot examples\n"
        for i, demo in enumerate(predictor.demos[:3]):
            instructions += f"\n### Example {i+1}\n"
            if hasattr(demo, "user_input"):
                instructions += f"Input: {demo.user_input}\n"
            if hasattr(demo, "output"):
                instructions += f"Output: {demo.output}\n"

    return instructions.strip()


# ── Layer 2: ACE — evolving domain knowledge ──────────────────────────────────

def build_ace_initial_playbook(mipro_instructions: str) -> dict:
    """
    Creates the seed for the ACE playbook from MIPROv2 results.
    MIPROv2 provides the structure → ACE enriches it over time.
    """
    return {
        "entries": [
            {
                "id":      "mipro_anchor",
                "source":  "mipro",   # read-only — do not overwrite
                "content": mipro_instructions,
                "helpful": 999,       # high counter = hard to prune
                "harmful": 0,
            }
        ],
        "token_budget": ACE_PLAYBOOK_BUDGET,
        "mipro_budget": MIPRO_BUDGET,
    }


def compose_final_context(
    mipro_instructions: str,
    ace_playbook:       dict,
) -> str:
    """
    Assembles the final context for the agent:
      - MIPROv2 instructions (read-only, always at the top)
      - ACE playbook (evolving domain strategies, below)

    Enforces token budget separation between layers.
    """
    ace_entries = [
        e for e in ace_playbook.get("entries", [])
        if e.get("source") != "mipro"                           # exclude MIPROv2 anchor
        and e.get("helpful", 0) > e.get("harmful", 0)           # only net-positive entries
    ]

    ace_content = "\n".join(
        f"- {e['content']}" for e in ace_entries
    ) if ace_entries else "(no strategies accumulated yet — first run)"

    return f"""
{mipro_instructions}

## Strategies from experience (ACE Playbook)
{ace_content}
""".strip()


def run_ace_session(
    user_id:            str,
    user_input:         str,
    mipro_program:      DomainAgent,
    mipro_instructions: str,
    feedback_fn:        Callable[[str], bool],
    ace:                ACE,
) -> str:
    """
    Full user session:
    1. Compose context (MIPROv2 structure + ACE knowledge)
    2. Run agent via DSPy
    3. Evaluate result via feedback_fn
    4. ACE updates the playbook based on feedback
    """
    user_pb = load_user_playbook(user_id)
    context = compose_final_context(mipro_instructions, user_pb)

    lm = dspy.LM(LM_MODEL)
    dspy.configure(lm=lm)
    result = mipro_program(context=context, user_input=user_input)
    output = result.output

    success = feedback_fn(output)

    ace_config = {
        "task_name":             "domain_agent",
        "playbook_token_budget": ACE_PLAYBOOK_BUDGET,
        "save_dir":              f"./runs/users/{user_id}/",
        "initial_playbook":      user_pb,
        "no_ground_truth":       True,   # feedback without labeled data
    }

    ace_result = ace.run_online(
        task={
            "input":   user_input,
            "output":  output,
            "context": context,
            "success": success,
        },
        config=ace_config,
    )

    # Save updated playbook with MIPROv2 anchor protection
    updated_pb = ace_result["updated_playbook"]
    updated_pb = protect_mipro_anchor(updated_pb, mipro_instructions)
    save_user_playbook(user_id, updated_pb)

    return output


def protect_mipro_anchor(playbook: dict, mipro_instructions: str) -> dict:
    """
    Ensures ACE did not overwrite the MIPROv2 anchor.
    Restores it if it is missing.
    """
    entries    = playbook.get("entries", [])
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


# ── Playbook management ───────────────────────────────────────────────────────

def load_user_playbook(user_id: str) -> dict:
    path = Path(USER_PLAYBOOKS) / f"{user_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    # Cold start: seed from global playbook
    if Path(GLOBAL_PLAYBOOK).exists():
        print(f"Cold start {user_id} → seeding from global playbook")
        return json.loads(Path(GLOBAL_PLAYBOOK).read_text())
    return {"entries": [], "token_budget": ACE_PLAYBOOK_BUDGET}


def save_user_playbook(user_id: str, playbook: dict) -> None:
    path = Path(USER_PLAYBOOKS) / f"{user_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(playbook, indent=2))


# ── Usage example ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Phase 1: MIPROv2 (once, offline) ──────────────────────────────────────

    # Your labeled dataset — minimum ~50 examples
    trainset = [
        dspy.Example(
            context="",                          # empty at start
            user_input="Example query #1",
            output="Expected answer #1",
        ).with_inputs("context", "user_input"),
        # ... more examples
    ]

    def my_metric(example, prediction, trace=None) -> bool:
        """Your success metric for MIPROv2."""
        return prediction.output == example.output

    optimized_program  = run_mipro_optimization(trainset, my_metric, auto="medium")
    mipro_instructions = extract_mipro_instructions(optimized_program)

    # Build global playbook from MIPROv2 (seed for new users)
    global_pb = build_ace_initial_playbook(mipro_instructions)
    Path(GLOBAL_PLAYBOOK).parent.mkdir(parents=True, exist_ok=True)
    Path(GLOBAL_PLAYBOOK).write_text(json.dumps(global_pb, indent=2))
    print("Global playbook initialized with MIPROv2 anchor")

    # ── Phase 2: ACE (continuous, online per-user) ────────────────────────────

    ace = ACE(
        api_provider="openai",
        generator_model="gpt-4o",
        reflector_model="gpt-4o",
        curator_model="gpt-4o",
        max_tokens=4096,
    )

    def my_feedback(output: str) -> bool:
        """Your feedback signal — no labeled data required."""
        return len(output) > 10   # replace with real logic

    response = run_ace_session(
        user_id="user_123",
        user_input="Specific user query",
        mipro_program=optimized_program,
        mipro_instructions=mipro_instructions,
        feedback_fn=my_feedback,
        ace=ace,
    )
    print("Agent response:", response)
