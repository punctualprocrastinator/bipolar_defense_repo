"""Environment and code provenance capture.

Everything here answers one reviewer question: "on what, exactly, was this number produced?"
Captured once per run and embedded in ``manifest.json``. See CLAUDE.md §1.2.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

# Packages whose versions can silently change a result. Extend rather than replace.
TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "numpy",
    "nnsight",
    "nanogcg",
    "tokenizers",
    "safetensors",
)


@dataclass(frozen=True)
class GitInfo:
    available: bool
    commit: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    # Full diff of tracked files at run time. Without this, a "dirty" flag tells a reviewer
    # the code differed from the commit but not how, which makes the run unreproducible.
    diff_sha256: str | None = None


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    cuda_available: bool
    device_name: str | None = None
    capability: str | None = None
    total_memory_gb: float | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    device_count: int = 0


@dataclass(frozen=True)
class Provenance:
    python: str
    platform: str
    argv: list[str]
    packages: dict[str, str]
    git: GitInfo
    device: DeviceInfo
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def collect_git(repo_root: Path) -> GitInfo:
    """Capture git state. Degrades gracefully — this repo is currently not a git repo."""
    commit = _run_git(["rev-parse", "HEAD"], repo_root)
    if commit is None:
        return GitInfo(available=False)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    status = _run_git(["status", "--porcelain"], repo_root)
    diff = _run_git(["diff", "HEAD"], repo_root)
    return GitInfo(
        available=True,
        commit=commit,
        branch=branch,
        dirty=bool(status),
        diff_sha256=hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    )


def collect_packages() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def collect_device() -> DeviceInfo:
    """Capture GPU identity.

    Deliberately the only place besides ``bsc.models`` that touches ``torch.cuda``
    (CLAUDE.md §1.7). Records compute capability so a Blackwell (sm_100/sm_120) run is
    distinguishable from a Turing (sm_75) smoke test in the artifact itself.
    """
    import torch

    if not torch.cuda.is_available():
        return DeviceInfo(device="cpu", cuda_available=False, device_count=0)

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    driver = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0:
            driver = out.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = None

    return DeviceInfo(
        device=f"cuda:{idx}",
        cuda_available=True,
        device_name=props.name,
        capability=f"sm_{props.major}{props.minor}",
        total_memory_gb=round(props.total_memory / 1024**3, 2),
        cuda_version=torch.version.cuda,
        driver_version=driver,
        device_count=torch.cuda.device_count(),
    )


def collect(repo_root: Path, tracked_env: tuple[str, ...] = ()) -> Provenance:
    """Collect the full provenance record for a run."""
    import os

    return Provenance(
        python=sys.version.split()[0],
        platform=platform.platform(),
        argv=list(sys.argv),
        packages=collect_packages(),
        git=collect_git(repo_root),
        device=collect_device(),
        env={k: os.environ[k] for k in tracked_env if k in os.environ},
    )


def config_hash(config: Any) -> str:
    """Stable short hash of a config, used in run directory names.

    Uses sorted-key JSON so that dict ordering never changes the hash, which would otherwise
    make two identical configs land in different run directories.
    """
    if hasattr(config, "to_dict"):
        payload = config.to_dict()
    elif hasattr(config, "__dataclass_fields__"):
        payload = asdict(config)
    else:
        payload = config
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:10]
