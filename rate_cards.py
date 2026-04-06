"""
Card Rater — interaktywna ocena kart digestu
=============================================
Wyświetla karty wygenerowane przez generate_digests.py
i zbiera Twoją ocenę przydatności dla każdej persony.

Użycie:
  python rate_cards.py

Sterowanie:
  1-5  → ocena (1=bezużyteczna, 5=bardzo przydatna)
  s    → pomiń kartę (skip)
  q    → zapisz i zakończ

Wynik:
  ratings/ratings.json   ← surowe oceny
  ratings/feedback.json  ← gotowy input dla score_feedback.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

DIGESTS_DIR = Path("digests")
RATINGS_DIR = Path("ratings")

# ── Opisy person — widoczne podczas oceniania ─────────────────────────────────

PERSONAS = {
    "alex": {
        "name":        "Alex — Senior ML Engineer",
        "description": (
            "Chce wiedzieć: czy działa produkcyjnie, jakie benchmarki, "
            "czy jest kod, co zastępuje, latencja i koszt. "
            "Pomija hype, funding i org changes."
        ),
        "color": "\033[94m",   # niebieski
    },
    "jordan": {
        "name":        "Jordan — VP of Engineering",
        "description": (
            "Chce wiedzieć: czy dojrzałe, kto wygrywa, co zrobić w tym kwartale. "
            "Nie potrzebuje detali implementacyjnych ani kodu."
        ),
        "color": "\033[92m",   # zielony
    },
}

# ── Terminal helpers ──────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
YELLOW = "\033[93m"
RED    = "\033[91m"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def hr(char: str = "─", width: int = 62) -> str:
    return char * width


def print_card(card: dict, index: int, total: int, snapshot_label: str):
    """Wyświetla kartę w czytelnym formacie."""
    print(f"\n{hr('═')}")
    print(f"{BOLD}Karta {index}/{total}{RESET}  {DIM}{snapshot_label}{RESET}")
    print(hr())

    # Pola w kolejności ważności
    priority_fields = ["TITLE", "ARXIV_ID", "GITHUB_URL", "BENCHMARK",
                       "RESULT", "REPLACES", "HARDWARE", "LICENSE", "VERDICT",
                       "MATURITY", "COMPETITOR", "EFFORT", "RECOMMENDATION", "RATIONALE"]

    for field in priority_fields:
        # Szukaj case-insensitive
        val = None
        for k, v in card.items():
            if k.upper() == field:
                val = v
                break
        if val and val not in ("N/A", "n/a", ""):
            label = f"{field}:"
            print(f"  {BOLD}{label:<16}{RESET} {val}")

    # Pozostałe pola których nie ma w liście
    shown = {f.upper() for f in priority_fields}
    for k, v in card.items():
        if k.upper() not in shown and v and v not in ("N/A", ""):
            print(f"  {DIM}{k:<16}{RESET} {v}")

    print(hr())


def ask_rating(persona_key: str) -> int | None:
    """
    Pyta o ocenę dla danej persony.
    Zwraca 1-5, None jeśli skip, lub rzuca SystemExit jeśli quit.
    """
    p = PERSONAS[persona_key]
    color = p["color"]

    print(f"\n{color}{BOLD}{p['name']}{RESET}")
    print(f"{DIM}{p['description']}{RESET}")
    print(f"\n  Jak przydatna jest ta karta dla {p['name'].split(' — ')[0]}?")
    print(f"  {BOLD}1{RESET} = bezużyteczna  "
          f"{BOLD}2{RESET} = słaba  "
          f"{BOLD}3{RESET} = OK  "
          f"{BOLD}4{RESET} = dobra  "
          f"{BOLD}5{RESET} = świetna")
    print(f"  {DIM}[s] pomiń  [q] zapisz i zakończ{RESET}\n")

    while True:
        try:
            raw = input("  Ocena: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n\nPrzerwano — zapisuję dotychczasowe oceny...")
            raise SystemExit(0)

        if raw == "q":
            raise SystemExit(0)
        if raw == "s":
            return None
        if raw in ("1", "2", "3", "4", "5"):
            return int(raw)
        print(f"  {RED}Wpisz 1-5, s lub q.{RESET}")


# ── Load all cards ────────────────────────────────────────────────────────────

def load_all_cards() -> list[dict]:
    """
    Ładuje karty ze wszystkich digestów.
    Deduplikuje po tytule — ta sama karta może pojawić się
    w digestach alex i jordan dla tego samego snapshotu.
    """
    digest_files = sorted(DIGESTS_DIR.glob("snapshot_*.json"))
    if not digest_files:
        print(f"✗  Brak digestów w {DIGESTS_DIR}/")
        print("   Uruchom najpierw: python generate_digests.py")
        sys.exit(1)

    cards_by_title: dict[str, dict] = {}

    for path in digest_files:
        digest = json.loads(path.read_text())
        snap_id = digest["snapshot_id"]
        label   = digest["label"]

        for card in digest.get("cards", []):
            title = (
                card.get("TITLE") or
                card.get("title") or
                "untitled"
            ).strip()[:80]

            key = f"{snap_id}::{title.lower()}"

            if key not in cards_by_title:
                cards_by_title[key] = {
                    "key":          key,
                    "snapshot_id":  snap_id,
                    "label":        label,
                    "title":        title,
                    "card":         card,
                    "rating_alex":   None,
                    "rating_jordan": None,
                    "skipped":       False,
                }

    return list(cards_by_title.values())


def load_existing_ratings() -> dict[str, dict]:
    """Ładuje istniejące oceny żeby można było wznowić sesję."""
    path = RATINGS_DIR / "ratings.json"
    if path.exists():
        data = json.loads(path.read_text())
        return {r["key"]: r for r in data}
    return {}


def save_ratings(cards: list[dict]):
    RATINGS_DIR.mkdir(exist_ok=True)
    path = RATINGS_DIR / "ratings.json"
    path.write_text(json.dumps(cards, indent=2, ensure_ascii=False))


def save_feedback(cards: list[dict]):
    """
    Generuje feedback.json gotowy do użycia przez score_feedback.py
    i jako input dla ACE playbook builder.
    """
    rated = [c for c in cards if not c["skipped"] and
             (c["rating_alex"] is not None or c["rating_jordan"] is not None)]

    feedback = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "total_cards":    len(cards),
        "rated_cards":    len(rated),
        "skipped_cards":  sum(1 for c in cards if c["skipped"]),
        "rater":          "human_expert",
        "scale":          "1-5 (1=useless, 5=excellent)",
        "cards": [
            {
                "key":           c["key"],
                "snapshot_id":   c["snapshot_id"],
                "label":         c["label"],
                "title":         c["title"],
                "rating_alex":   c["rating_alex"],
                "rating_jordan": c["rating_jordan"],
                # Sygnały dla ACE — czy karta była użyteczna per persona
                "useful_for_alex":   (c["rating_alex"]   or 0) >= 4,
                "useful_for_jordan": (c["rating_jordan"] or 0) >= 4,
                # Różnica ocen — karty z dużą różnicą pokazują divergencję person
                "divergence": abs(
                    (c["rating_alex"]   or 0) -
                    (c["rating_jordan"] or 0)
                ),
            }
            for c in rated
        ],
    }

    # Statystyki
    alex_ratings   = [c["rating_alex"]   for c in rated if c["rating_alex"]]
    jordan_ratings = [c["rating_jordan"] for c in rated if c["rating_jordan"]]

    if alex_ratings:
        feedback["stats_alex"] = {
            "avg":     round(sum(alex_ratings) / len(alex_ratings), 2),
            "useful":  sum(1 for r in alex_ratings if r >= 4),
            "useless": sum(1 for r in alex_ratings if r <= 2),
        }
    if jordan_ratings:
        feedback["stats_jordan"] = {
            "avg":     round(sum(jordan_ratings) / len(jordan_ratings), 2),
            "useful":  sum(1 for r in jordan_ratings if r >= 4),
            "useless": sum(1 for r in jordan_ratings if r <= 2),
        }

    # Najbardziej divergentne karty — złoto dla blogu
    divergent = sorted(
        feedback["cards"],
        key=lambda x: x["divergence"],
        reverse=True,
    )[:5]
    feedback["most_divergent_cards"] = divergent

    path = RATINGS_DIR / "feedback.json"
    path.write_text(json.dumps(feedback, indent=2, ensure_ascii=False))
    return feedback


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    clear()
    print(f"\n{BOLD}Card Rater{RESET} — oceniasz karty dla Alexa i Jordan")
    print(f"{DIM}Dane z: {DIGESTS_DIR}/{RESET}\n")

    all_cards = load_all_cards()
    existing  = load_existing_ratings()

    # Przywróć poprzednie oceny
    restored = 0
    for card in all_cards:
        if card["key"] in existing:
            prev = existing[card["key"]]
            card["rating_alex"]   = prev.get("rating_alex")
            card["rating_jordan"] = prev.get("rating_jordan")
            card["skipped"]       = prev.get("skipped", False)
            restored += 1

    # Karty do oceny (pomiń już ocenione)
    to_rate = [
        c for c in all_cards
        if not c["skipped"]
        and (c["rating_alex"] is None or c["rating_jordan"] is None)
    ]

    print(f"  Łącznie kart:     {len(all_cards)}")
    print(f"  Już ocenionych:   {restored}")
    print(f"  Do oceny teraz:   {len(to_rate)}")
    print(f"\n  {DIM}Możesz przerwać w dowolnym momencie (q) — postęp jest zapisywany.{RESET}")
    input(f"\n  Naciśnij Enter żeby zacząć...")

    try:
        for i, card in enumerate(to_rate, 1):
            clear()
            print_card(card["card"], i, len(to_rate), card["label"])

            # Ocena dla Alexa
            if card["rating_alex"] is None:
                rating_a = ask_rating("alex")
                if rating_a is None:
                    card["skipped"] = True
                    save_ratings(all_cards)
                    continue
                card["rating_alex"] = rating_a

            # Ocena dla Jordan
            if card["rating_jordan"] is None:
                rating_j = ask_rating("jordan")
                if rating_j is None:
                    card["skipped"] = True
                    save_ratings(all_cards)
                    continue
                card["rating_jordan"] = rating_j

            # Zapisz po każdej karcie — nie stracisz postępu
            save_ratings(all_cards)

            # Krótkie podsumowanie oceny
            a = card["rating_alex"]
            j = card["rating_jordan"]
            diff = abs(a - j)
            diff_label = (
                f"{YELLOW}↑ duża różnica ({diff} pkt) — ciekawe dla blogu!{RESET}"
                if diff >= 2 else ""
            )
            print(f"\n  {DIM}Alex: {a}/5  Jordan: {j}/5  {diff_label}{RESET}")
            input(f"  {DIM}Enter → następna karta...{RESET}")

    except SystemExit:
        pass

    # Finalne statystyki i zapis
    clear()
    feedback = save_feedback(all_cards)

    rated = feedback["rated_cards"]
    print(f"\n{BOLD}Gotowe!{RESET}\n")
    print(f"  Oceniono kart:  {rated}/{len(all_cards)}")

    if "stats_alex" in feedback:
        s = feedback["stats_alex"]
        print(f"\n  {PERSONAS['alex']['color']}{BOLD}Alex{RESET}")
        print(f"    Średnia:     {s['avg']}/5")
        print(f"    Użyteczne:   {s['useful']} kart (≥4)")
        print(f"    Słabe:       {s['useless']} kart (≤2)")

    if "stats_jordan" in feedback:
        s = feedback["stats_jordan"]
        print(f"\n  {PERSONAS['jordan']['color']}{BOLD}Jordan{RESET}")
        print(f"    Średnia:     {s['avg']}/5")
        print(f"    Użyteczne:   {s['useful']} kart (≥4)")
        print(f"    Słabe:       {s['useless']} kart (≤2)")

    if feedback.get("most_divergent_cards"):
        print(f"\n  {YELLOW}{BOLD}Najbardziej divergentne karty (złoto dla blogu):{RESET}")
        for c in feedback["most_divergent_cards"][:3]:
            print(f"    [{c['divergence']} pkt różnicy] {c['title'][:55]}")

    print(f"\n  Pliki:")
    print(f"    ratings/ratings.json  — surowe oceny")
    print(f"    ratings/feedback.json — input dla score_feedback.py\n")


if __name__ == "__main__":
    main()
