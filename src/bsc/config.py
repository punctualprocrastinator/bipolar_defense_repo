"""Typed, hashable, YAML-backed experiment configuration.

No magic numbers in function bodies (CLAUDE.md §1.3). Everything an experiment depends on is a
field here, so the manifest fully determines the run.

Load order, later overriding earlier:
    dataclass defaults  ->  YAML file  ->  ``--set key=value`` CLI overrides
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


@dataclass
class ModelConfig:
    """Which model, loaded how."""

    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    device: str = "auto"  # auto | cpu | cuda | cuda:N
    attn_implementation: str = "auto"  # auto | sdpa | eager | flash_attention_2
    trust_remote_code: bool = False
    # Head-level hooks read o_proj's *input*, which fused/flash kernels may not expose the same
    # way across versions. "eager" is the safe choice for interpretability; "auto" picks sdpa
    # for plain generation. Force it explicitly for any run that patches heads.
    require_eager_for_hooks: bool = True
    max_memory_gb: float | None = None
    seed: int = 0


@dataclass
class GenerationConfig:
    """Generation settings. Shared by baseline and defended runs so the comparison is paired."""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    # Behavioral claims need n_trials sampled generations, not one greedy sample
    # (CLAUDE.md §2.1). The greedy run is recorded alongside but reported separately.
    n_trials: int = 8
    record_greedy: bool = True


@dataclass
class InterventionConfig:
    """The bipolar defense, parameterised.

    ``refusal_multiplier=1.0`` with ``ablate_compliance=False`` is the identity intervention,
    which every unit test asserts is a strict no-op (CLAUDE.md §5).
    """

    enabled: bool = False
    mode: str = "multiplicative"  # multiplicative | additive_steering | none
    refusal_multiplier: float = 3.0
    steering_alpha: float = 1.0
    steering_vectors_path: str | None = None
    ablate_compliance: bool = True
    compliance_scale: float = 0.0  # 0.0 == zero-ablation; kept as a knob for partial ablation
    # Explicit per CLAUDE.md §3.3: during KV-cached decode only one position exists, so
    # "last" and "all" coincide at decode but differ during prefill. Never leave implicit.
    positions: str = "last"  # last | all
    n_refusal_heads: int = 11
    n_compliance_heads: int = 5


@dataclass
class JudgeConfig:
    """How a generation is scored as refusal vs. compliance."""

    method: str = "keyword"  # keyword | llm | both
    # Only the opening of a response is scanned: a response that refuses and then complies is a
    # jailbreak, and scanning the whole string would score it as a refusal.
    prefix_chars: int = 240
    llm_model: str | None = None
    # When method="both", disagreements are written to disk for manual review rather than
    # silently resolved toward whichever judge the author prefers.
    record_disagreements: bool = True


@dataclass
class DataConfig:
    """Datasets and, critically, splits."""

    dataset: str = "custom"  # custom | harmbench | advbench | jbb
    split: str = "test"  # dev | test  — tune on dev, report on test (CLAUDE.md §2.2)
    dev_fraction: float = 0.3
    limit: int | None = None
    benign_categories: list[str] = field(default_factory=list)
    path: str | None = None


@dataclass
class CircuitConfig:
    """Circuit discovery / loading."""

    circuit_map_path: str | None = None
    # Selection by count is what the legacy code did. Selecting by an effect-size threshold is
    # the defensible alternative; both are supported so the paper can report sensitivity to it.
    selection: str = "top_k"  # top_k | threshold
    top_k_refusal: int = 11
    top_k_compliance: int = 5
    score_threshold: float = 0.0
    patch_metric: str = "refusal_logit_diff"


@dataclass
class EvalConfig:
    """Statistics. A rate without n and a CI is not a result (CLAUDE.md §1.5)."""

    bootstrap_samples: int = 10_000
    confidence: float = 0.95
    seed: int = 12345


@dataclass
class ExperimentConfig:
    """Root config object. This is what gets hashed into the run directory name."""

    experiment: str = "unnamed"
    seed: int = 0
    strict_determinism: bool = False
    notes: str = ""
    dry_run: bool = False
    # Multiplier grid for the Crescendo sweep; empty means use the experiment's default grid.
    intervention_sweep: list[float] = field(default_factory=list)
    # Attacker model for scenario harvesting (empty = use the same model as the target).
    attacker_model: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    circuit: CircuitConfig = field(default_factory=CircuitConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Fail fast on impossible configs, before a model is loaded."""
        errors: list[str] = []
        if self.generation.n_trials < 1:
            errors.append("generation.n_trials must be >= 1")
        if not 0.0 < self.eval.confidence < 1.0:
            errors.append("eval.confidence must be in (0, 1)")
        if self.data.split not in {"dev", "test", "all"}:
            errors.append(f"data.split must be dev|test|all, got {self.data.split!r}")
        if self.intervention.positions not in {"last", "all"}:
            errors.append(f"intervention.positions must be last|all, got {self.intervention.positions!r}")
        if self.intervention.mode not in {"multiplicative", "additive_steering", "none"}:
            errors.append(f"intervention.mode invalid: {self.intervention.mode!r}")
        if self.intervention.mode == "additive_steering" and not self.intervention.steering_vectors_path:
            errors.append("intervention.mode=additive_steering requires steering_vectors_path")
        if self.judge.method not in {"keyword", "llm", "both"}:
            errors.append(f"judge.method invalid: {self.judge.method!r}")
        if self.judge.method in {"llm", "both"} and not self.judge.llm_model:
            errors.append("judge.method requires llm_model to be set")
        if errors:
            raise ValueError("invalid config:\n  - " + "\n  - ".join(errors))


# -- loading ---------------------------------------------------------------------


def _coerce(value: Any, target: Any) -> Any:
    """Coerce a YAML/CLI scalar into the annotated field type."""
    origin = get_origin(target)
    if origin is not None:
        args = [a for a in get_args(target) if a is not type(None)]
        if value is None:
            return None
        if origin is list:
            return list(value)
        # Optional[X] -> coerce to X
        if len(args) == 1:
            return _coerce(value, args[0])
        return value
    if target is bool and isinstance(value, str):
        low = value.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
        raise ValueError(f"cannot parse {value!r} as bool")
    if target in (int, float, str) and value is not None:
        return target(value)
    return value


def _from_dict(cls: type[T], payload: dict[str, Any]) -> T:
    """Build a (possibly nested) dataclass from a plain dict, rejecting unknown keys.

    Unknown keys are an error rather than a warning: a typo'd YAML key that gets silently
    ignored means the run does something other than what the config says, which is the exact
    failure mode this whole module exists to prevent.
    """
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in payload:
            continue
        value = payload[f.name]
        ftype = hints[f.name]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _from_dict(ftype, value)
        else:
            kwargs[f.name] = _coerce(value, ftype)
    return cls(**kwargs)


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load config from YAML and apply ``key.subkey=value`` overrides."""
    payload: dict[str, Any] = {}
    if path is not None:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = _from_dict(ExperimentConfig, payload)

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must be key=value, got {override!r}")
        key, raw = override.split("=", 1)
        _apply_override(cfg, key.strip(), raw.strip())

    cfg.validate()
    return cfg


def _apply_override(cfg: Any, dotted: str, raw: str) -> None:
    parts = dotted.split(".")
    target = cfg
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"unknown config section: {part!r} in {dotted!r}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise ValueError(f"unknown config key: {dotted!r}")
    hints = get_type_hints(type(target))
    value = yaml.safe_load(raw)  # parses 3.0 -> float, true -> bool, null -> None, [a,b] -> list
    setattr(target, leaf, _coerce(value, hints[leaf]))


def save_config(cfg: ExperimentConfig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    return p
