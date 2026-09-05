"""Chain-assembly tests: the compromised attacker must lead the chain and never be defended, and the
defense must land on exactly one receiver. Model-free (bundles are stand-ins)."""

from __future__ import annotations

from bsc.agents import DefenseSpec
from bsc.experiments.propagation_chain import (
    COMPROMISED_SYSTEM,
    DEFENDED_POSITION,
    RECEIVER_SYSTEM,
    build_chain_agents,
)
from bsc.hooks import Head, HeadEdit

DEF = DefenseSpec(kind="steering", edits={Head(1, 2): HeadEdit(scale=3.0)})


def test_chain_shape_and_roles():
    ag = build_chain_agents("RECV", "ATK", 4, None, DEFENDED_POSITION)
    assert [a.name for a in ag] == ["A", "B", "C", "D"]
    assert ag[0].bundle == "ATK" and ag[0].role_system == COMPROMISED_SYSTEM
    assert all(a.bundle == "RECV" and a.role_system == RECEIVER_SYSTEM for a in ag[1:])


def test_attacker_is_never_defended():
    ag = build_chain_agents("RECV", "ATK", 4, DEF, DEFENDED_POSITION)
    assert ag[0].defense is None  # position 0 is the compromised peer


def test_defense_lands_on_exactly_one_receiver():
    ag = build_chain_agents("RECV", "ATK", 4, DEF, DEFENDED_POSITION)
    defended = [i for i, a in enumerate(ag) if a.defense is not None]
    assert defended == [DEFENDED_POSITION]


def test_no_defense_means_all_undefended():
    ag = build_chain_agents("RECV", "ATK", 4, None, DEFENDED_POSITION)
    assert all(a.defense is None for a in ag)


def test_chain_length_respected():
    for n in (2, 3, 5):
        assert len(build_chain_agents("R", "A", n, None, 1)) == n
