"""Shared fixtures.

The whole suite runs on CPU with no network and no model download (CLAUDE.md §5). The tiny
GQA model below is what makes that possible for the hook tests: it has the same module names
and the same query/KV head asymmetry as Qwen2.5-7B (28 query heads over 4 KV heads), at a size
that runs in milliseconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from bsc.circuits import CircuitMap, HeadScore
from bsc.hooks import Head
from bsc.models import HeadGeometry, ModelBundle, ResolvedDevice


class TinyAttention(nn.Module):
    """Attention block exposing an ``o_proj`` whose input is the per-head concatenation."""

    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(torch.tanh(self.q_proj(x)))


class TinyLayer(nn.Module):
    def __init__(self, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(hidden, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.self_attn(x)


class TinyInner(nn.Module):
    def __init__(self, n_layers: int, hidden: int, n_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TinyLayer(hidden, n_heads) for _ in range(n_layers))


class TinyModel(nn.Module):
    """Mimics the ``model.model.layers[i].self_attn.o_proj`` path that hooks target."""

    def __init__(self, n_layers: int = 4, hidden: int = 32, n_heads: int = 8) -> None:
        super().__init__()
        self.model = TinyInner(n_layers, hidden, n_heads)
        self.lm_head = nn.Linear(hidden, 16, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(x)


@pytest.fixture
def tiny_bundle() -> ModelBundle:
    """A ModelBundle wrapping TinyModel, with GQA geometry (8 query heads, 2 KV heads)."""
    torch.manual_seed(0)
    model = TinyModel(n_layers=4, hidden=32, n_heads=8).eval()
    geometry = HeadGeometry(
        n_layers=4,
        hidden_size=32,
        num_attention_heads=8,
        num_key_value_heads=2,  # GQA: 4 query heads per KV head, as in Qwen2.5-7B
        head_dim=4,
    )
    return ModelBundle(
        model=model,
        tokenizer=None,
        geometry=geometry,
        resolved=ResolvedDevice(torch.device("cpu"), torch.float32, "test fixture"),
        attn_implementation="eager",
        name="tiny-test-model",
    )


@pytest.fixture
def tiny_input() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 6, 32)


@pytest.fixture
def circuit() -> CircuitMap:
    """A small circuit map with an unambiguous ordering."""
    return CircuitMap(
        scores=[
            HeadScore(Head(3, 0), 2.0),
            HeadScore(Head(3, 1), 1.5),
            HeadScore(Head(2, 4), 1.0),
            HeadScore(Head(1, 2), 0.25),
            HeadScore(Head(0, 0), -0.1),
            HeadScore(Head(2, 7), -0.5),
            HeadScore(Head(1, 6), -1.2),
        ],
        model="tiny-test-model",
        method="unit_test",
    )


@pytest.fixture
def legacy_circuit_file(tmp_path: Path) -> Path:
    """A file in the exact legacy on-disk schema, to pin the loader against real artifacts."""
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "method": "harm_contrastive_patching",
        "num_heads": 28,
        "patch_layers": [17, 18, 19],
        "circuit_nodes": [
            {"layer": 25, "head": 1, "score": 1.873150634765625},
            {"layer": 25, "head": 3, "score": 0.5},
            {"layer": 20, "head": 3, "score": 0.1517},
            {"layer": 19, "head": 2, "score": -0.3008},
            {"layer": 25, "head": 4, "score": -0.6602},
        ],
        "top_10": [{"layer": 25, "head": 1, "score": 1.873150634765625}],
        "compliance_bottom_5": [{"layer": 25, "head": 4, "score": -0.6602}],
    }
    path = tmp_path / "circuit_map.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
