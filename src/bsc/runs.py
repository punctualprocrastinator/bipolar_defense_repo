"""Run directories, manifests, logging, and artifact IO.

One rule (CLAUDE.md §1.2, §1.4): a result that is not inside a run directory with a manifest
does not exist. ``RunContext`` is the only sanctioned way to write an experiment artifact.

Typical use::

    with RunContext.create("crescendo_defense", cfg) as run:
        run.log.info("starting")
        run.save_json("turn_results.json", results)
    # manifest.json, including status and duration, is written on exit — even on exception.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from bsc import provenance
from bsc.determinism import SeedState

def _find_repo_root() -> Path:
    """Walk up from this file to the directory containing ``pyproject.toml``.

    Anchoring on a marker file rather than a fixed number of ``parents[]`` hops keeps the package
    working when installed, vendored, or bootstrapped into a remote GPU sandbox at a different
    depth -- all three happen in this project.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here.parents[2]


# Env-overridable so a sandbox run can redirect artifacts without editing source.
REPO_ROOT = Path(os.environ.get("BSC_REPO_ROOT") or _find_repo_root())
RUNS_ROOT = Path(os.environ.get("BSC_RUNS_ROOT") or (REPO_ROOT / "runs"))


def _jsonable(obj: Any) -> Any:
    """Convert dataclasses/Paths/tensors into something ``json.dump`` accepts."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and hasattr(obj, "shape"):  # 0-d tensor / numpy scalar
        try:
            return obj.item()
        except (ValueError, RuntimeError):
            return str(obj)
    if hasattr(obj, "tolist"):  # ndarray / tensor
        return obj.tolist()
    return obj


class RunContext:
    """A single experiment run, its directory, and its manifest."""

    def __init__(
        self,
        experiment: str,
        config: Any,
        directory: Path,
        *,
        notes: str = "",
        tracked_env: tuple[str, ...] = (),
    ) -> None:
        self.experiment = experiment
        self.config = config
        self.dir = directory
        self.notes = notes
        self._tracked_env = tracked_env
        self._start = time.time()
        self._started_at = datetime.now(timezone.utc)
        self._seed_state: SeedState | None = None
        self._artifacts: list[str] = []
        self._metrics: dict[str, Any] = {}
        self._status = "running"
        self._error: str | None = None

        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = self._make_logger()

    @classmethod
    def create(
        cls,
        experiment: str,
        config: Any,
        *,
        runs_root: Path | None = None,
        notes: str = "",
        tracked_env: tuple[str, ...] = (),
    ) -> RunContext:
        """Create a fresh run directory: ``runs/<experiment>/<UTC timestamp>-<confighash>/``.

        The config hash in the name means two runs of the same config are visibly related, while
        the timestamp guarantees a previous run is never overwritten (CLAUDE.md §1.4).
        """
        root = runs_root or RUNS_ROOT
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = root / experiment / f"{stamp}-{provenance.config_hash(config)}"
        return cls(experiment, config, directory, notes=notes, tracked_env=tracked_env)

    def _make_logger(self) -> logging.Logger:
        log = logging.getLogger(f"bsc.run.{self.experiment}.{self.dir.name}")
        log.setLevel(logging.INFO)
        log.propagate = False
        log.handlers.clear()

        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")

        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        log.addHandler(stream)

        # Every run keeps its own full log next to its artifacts, so a surprising number can be
        # traced back to the console output that produced it.
        file_handler = logging.FileHandler(self.dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        log.addHandler(file_handler)
        return log

    # -- recording ---------------------------------------------------------------

    def record_seed(self, state: SeedState) -> None:
        self._seed_state = state

    def record_metric(self, name: str, value: Any) -> None:
        """Register a headline number for the manifest and ``runs/INDEX.md``."""
        self._metrics[name] = _jsonable(value)

    def save_json(self, name: str, payload: Any) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(payload), f, indent=2)
        self._artifacts.append(name)
        self.log.info("wrote %s", name)
        return path

    def save_text(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self._artifacts.append(name)
        self.log.info("wrote %s", name)
        return path

    def save_torch(self, name: str, obj: Any) -> Path:
        import torch

        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, path)
        self._artifacts.append(name)
        self.log.info("wrote %s", name)
        return path

    # -- manifest ----------------------------------------------------------------

    def write_manifest(self) -> Path:
        # The seed is authoritative in config.seed (the CLI seeds from it before the experiment
        # builds this context). Record it explicitly here too so a manifest is self-contained
        # even when the richer SeedState was not threaded through.
        seed_record = _jsonable(self._seed_state)
        if seed_record is None:
            cfg_seed = getattr(self.config, "seed", None)
            if cfg_seed is not None:
                seed_record = {"seed": cfg_seed, "source": "config.seed"}
        manifest = {
            "experiment": self.experiment,
            "status": self._status,
            "error": self._error,
            "started_at": self._started_at.isoformat(),
            "duration_sec": round(time.time() - self._start, 2),
            "notes": self.notes,
            "config": _jsonable(self.config),
            "config_hash": provenance.config_hash(self.config),
            "seed": seed_record,
            "metrics": self._metrics,
            "artifacts": self._artifacts,
            "provenance": provenance.collect(REPO_ROOT, self._tracked_env).to_dict(),
        }
        path = self.dir / "manifest.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return path

    def append_to_index(self) -> None:
        """Append a one-line summary to ``runs/INDEX.md`` (CLAUDE.md §6)."""
        index = RUNS_ROOT / "INDEX.md"
        index.parent.mkdir(parents=True, exist_ok=True)
        if not index.exists():
            index.write_text(
                "# Run index\n\n"
                "Appended automatically by `bsc.runs.RunContext`. One line per run.\n\n"
                "| finished (UTC) | experiment | status | run dir | headline metrics |\n"
                "|---|---|---|---|---|\n",
                encoding="utf-8",
            )
        rel = self.dir.relative_to(RUNS_ROOT) if self.dir.is_relative_to(RUNS_ROOT) else self.dir
        metrics = ", ".join(f"{k}={v}" for k, v in self._metrics.items()) or "—"
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        with index.open("a", encoding="utf-8") as f:
            f.write(f"| {stamp} | {self.experiment} | {self._status} | `{rel}` | {metrics} |\n")

    # -- context manager ---------------------------------------------------------

    def __enter__(self) -> RunContext:
        self.log.info("run dir: %s", self.dir)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc is not None:
            # A crashed run still gets a manifest, marked failed. Silent empty directories are
            # how "we ran that, I think?" ambiguity creeps into a project.
            self._status = "failed"
            self._error = "".join(traceback.format_exception(exc_type, exc, tb))[-4000:]
            self.log.error("run failed: %s", exc)
        else:
            self._status = "completed"
        self.write_manifest()
        self.append_to_index()
        self.log.info("run %s in %.1fs -> %s", self._status, time.time() - self._start, self.dir)
        for handler in self.log.handlers:
            handler.close()
        return False  # never swallow the exception
