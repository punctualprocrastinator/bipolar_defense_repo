"""Versioned datasets.

Prompts and scenarios live in data files, not as string literals inside experiment scripts
(CLAUDE.md §2, and the fix for the 8 divergent copies of the RDX dialogue the legacy code
carried -- one of which had different wording that produced a 40% vs 20% discrepancy).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CrescendoScenario:
    """One multi-turn escalation.

    ``lines`` alternates user/assistant, 9 entries: the attacker supplies both sides of turns
    1-4 (priming the KV cache), and line 8 is the terminal harmful user request. The model is
    only asked to *generate* at turn 5 -- the earlier assistant lines are the attack's injected
    context, not the model's own outputs.
    """

    category: str
    lines: list[str]

    def messages_up_to_turn(self, turn: int) -> list[dict[str, str]]:
        """Chat messages for turns 1..``turn`` (1-indexed), ending on a user turn.

        Turn ``t`` includes ``2*t - 1`` lines: the first ``t`` user lines and the ``t-1``
        assistant lines between them, so the sequence always ends ready for a generation prompt.
        """
        if not 1 <= turn <= 5:
            raise ValueError(f"turn must be 1..5, got {turn}")
        n_lines = 2 * turn - 1
        out = []
        for j in range(n_lines):
            role = "user" if j % 2 == 0 else "assistant"
            out.append({"role": role, "content": self.lines[j]})
        return out

    @property
    def terminal_request(self) -> str:
        return self.lines[8]


@lru_cache(maxsize=1)
def load_crescendo_scenarios() -> dict[str, CrescendoScenario]:
    payload = json.loads((DATA_DIR / "crescendo_scenarios.json").read_text(encoding="utf-8"))
    scenarios = {}
    for name, entry in payload["scenarios"].items():
        lines = entry["lines"]
        if len(lines) != 9:
            raise ValueError(f"scenario {name} has {len(lines)} lines, expected 9")
        scenarios[name] = CrescendoScenario(category=entry["category"], lines=lines)
    return scenarios


def get_scenario(name: str) -> CrescendoScenario:
    scenarios = load_crescendo_scenarios()
    if name not in scenarios:
        raise KeyError(f"unknown scenario {name!r}; available: {sorted(scenarios)}")
    return scenarios[name]
