"""
Pipeline runner — run the full pipeline or any individual step.

Usage:
  uv run pipeline                        # run all steps
  uv run pipeline --step collect
  uv run pipeline --step gen-trainset
  uv run pipeline --step optimize
  uv run pipeline --step gen-digests
  uv run pipeline --step rate
  uv run pipeline --step score
  uv run pipeline --from optimize        # run from a step onwards
"""

import argparse
import sys

STEPS = [
    ("collect",       "steps.collect",            "Collect AI news snapshots"),
    ("gen-trainset",  "steps.generate_trainset",   "Generate SIMBA training set"),
    ("optimize",      "steps.optimize_prompts",    "Optimize extraction prompt via SIMBA"),
    ("gen-digests",   "steps.generate_digests",    "Generate persona digests"),
    ("rate",          "steps.rate_cards",          "Rate digest cards interactively"),
    ("score",         "steps.score_feedback",      "Score feedback and build playbooks"),
]

STEP_NAMES = [name for name, _, _ in STEPS]


def run_step(name: str, module_path: str) -> None:
    import importlib
    print(f"\n{'═' * 60}")
    print(f"  Step: {name}")
    print(f"{'═' * 60}")
    mod = importlib.import_module(module_path)
    mod.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="ACE pipeline runner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--step",
        choices=STEP_NAMES,
        help="Run a single step",
    )
    group.add_argument(
        "--from",
        dest="from_step",
        choices=STEP_NAMES,
        help="Run from this step to the end",
    )
    args = parser.parse_args()

    if args.step:
        _, module_path, _ = next(s for s in STEPS if s[0] == args.step)
        run_step(args.step, module_path)
        return

    if args.from_step:
        start = STEP_NAMES.index(args.from_step)
        steps_to_run = STEPS[start:]
    else:
        steps_to_run = STEPS

    print("ACE Pipeline")
    for i, (name, _, description) in enumerate(steps_to_run, 1):
        print(f"  {i}. {name:16} — {description}")

    for name, module_path, _ in steps_to_run:
        try:
            run_step(name, module_path)
        except Exception as e:
            print(f"\n✗ Step '{name}' failed: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"\n{'═' * 60}")
    print("  Pipeline complete.")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
