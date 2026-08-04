"""Global determinism control.

Every experiment calls :func:`set_seed` exactly once, at the top, with a seed that came from
config. See CLAUDE.md §1.1.

Determinism in this project is *seed-level*, not bitwise-across-hardware: bf16 matmuls on
different GPU architectures (and CPU vs. CUDA) produce different low-order bits, so a run on
Blackwell will not byte-match a run on Turing. What we guarantee is that the same code, same
seed, and same device produce the same result, and that the exact device is recorded in the
manifest so a mismatch is diagnosable rather than mysterious.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

# Set before the first CUDA context is created to make cuBLAS GEMMs reproducible.
# Harmless on CPU-only builds.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass(frozen=True)
class SeedState:
    """Record of what determinism settings were actually applied, for the manifest."""

    seed: int
    deterministic_algorithms: bool
    cudnn_deterministic: bool
    cublas_workspace_config: str | None


def set_seed(seed: int, *, strict: bool = False) -> SeedState:
    """Seed every RNG this project can touch.

    Args:
        seed: The seed. Must come from config, never a literal at the call site.
        strict: If True, ask PyTorch to raise on any nondeterministic kernel. Use for
            correctness-critical runs (circuit discovery, ablations). Leave False for
            generation-heavy runs, where some kernels have no deterministic implementation
            and strict mode would simply crash.

    Returns:
        The applied settings, to be embedded in the run manifest.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(strict, warn_only=not strict)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return SeedState(
        seed=seed,
        deterministic_algorithms=strict,
        cudnn_deterministic=True,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def trial_seed(base_seed: int, trial_index: int) -> int:
    """Derive a per-trial seed.

    Sampled generation trials must use *derived, recorded* seeds rather than letting the RNG
    advance implicitly — otherwise re-running trial 3 alone gives a different result than
    trial 3 within a full sweep, and the baseline/defended comparison stops being paired.

    Legacy note: the original scripts used ``SEED_BASE + i`` with ``SEED_BASE = 1000``. That is
    reproducible but collides across experiments (experiment A trial 5 == experiment B trial 5).
    The hash-mix here keeps trials independent while staying a pure function of the inputs.
    """
    if trial_index < 0:
        raise ValueError(f"trial_index must be non-negative, got {trial_index}")
    # Splitmix64-style finalizer: cheap, deterministic, no cross-experiment aliasing.
    x = (base_seed * 0x9E3779B97F4A7C15 + trial_index + 1) & 0xFFFF_FFFF_FFFF_FFFF
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFF_FFFF_FFFF_FFFF
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & 0xFFFF_FFFF_FFFF_FFFF
    x ^= x >> 31
    return x % (2**31 - 1)


def prompt_seed(base_seed: int, name: str) -> int:
    """Stable per-prompt base seed from a name.

    Do NOT use the builtin ``hash(name)`` for this: Python salts string hashing per process
    (unless PYTHONHASHSEED is fixed), so ``cfg.seed + hash(name)`` gives *different* seeds in two
    separate runs. Within one run it is fine (baseline and treatments share the process, so the
    pairing holds), but two runs of the same config would not reproduce each other's baseline —
    which is exactly the reproducibility guarantee this project promises. hashlib is stable
    across processes and machines.
    """
    import hashlib

    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return (base_seed + int.from_bytes(digest, "big")) % (2**31 - 1)
