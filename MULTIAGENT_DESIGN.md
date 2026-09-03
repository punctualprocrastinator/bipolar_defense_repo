# Multi-agent propagation — thesis, experiments, and harness design

Dated 2026-08-05. This is the design of record for the multi-agent arm (MoE arm set aside). It
specifies the claims, the experiments that establish them, and the concrete harness that runs them
as reproducible `bsc` experiments. A literature-review section is appended (filled from the review
agent) and the positioning/deltas live there.

---

## 1. Thesis and claim set

**Thesis.** Safety training makes refusal conditional on *request framing*, not on *content*.
Multi-agent systems break this: harmful content arriving as a **peer agent's contribution to an
ongoing task** — not as a request — does not trigger the refusal circuit, because the model is in
*continuation* mode. Agent-to-agent jailbreak propagation is therefore mediated by the
**compliance/continuation head circuit**, and steering that circuit (activation-level, inference
time) restores content-based refusal and intercepts propagation across an agent network.

**Claims, in dependency order.**

| id | claim | established by | type |
|----|-------|---------------|------|
| C0 | Harmful content propagates agent→agent when re-framed as peer continuation | C1 baseline | phenomenon (cite others too) |
| **C1** | The **same content** triggers refusal as a *request* but not as a *peer message* | **M1** | mechanism (novel) |
| **C2** | Propagation fires the **compliance/continuation** heads, not evasion of refusal heads | **M1** | mechanism (novel) |
| **C3** | **Compliance-side** steering intercepts *propagation* more than it intercepts *direct* attacks (interaction) | **M2** | causal (novel) |
| C4 | In a chain A→B→C→D, propagation amplifies/decays; one well-placed defended agent protects downstream | C1-chain | network result (novel) |
| C5 | Activation steering is competitive with prompt/topology defenses and the CB checkpoint, training-free | B1 | positioning |
| C6 | The interception survives an adaptive compromised peer | B2 | robustness |

**Honesty anchor.** Lead with the **mechanism** (C1–C3). The steering *defense* (C3–C6) is reported
degeneration-audited (HarmBench ASR *and* the keyword-NONRESPONSE share, always) — a
mechanism-confirming intervention, not a SOTA-defense claim. On dense models our steering can reach
low ASR partly by degradation, and Circuit Breakers is strong; the paper does not claim to beat it.

---

## 2. Experiments

### M1 — peer vs request (the core mechanism). *Cheap.*
For each harmful **content item** `x`, construct two prompts that both end by asking B to produce `x`:
- **request framing:** a user directly requests `x` of B.
- **peer framing:** a peer agent's message supplies the task context and a partial harmful trajectory,
  and asks B to *continue / help complete* it — the harmful ask is identical, the frame is "peer
  contribution", not "user request".
The harmful span is **string-identical** across the two (the manipulation is only the surrounding
frame). Measure on B at the first generated position:
- refusal rate (HarmBench + keyword),
- **refusal-logit-diff** (Σ logits of refusal openers − Σ logits of compliance openers),
- **circuit projection**: projection of the last-token residual / per-head o_proj-input onto the
  fitted per-head refuse−comply directions, split into refusal-head vs compliance-head mass.
**Prediction:** peer framing → lower refusal rate, lower refusal-logit-diff, and a shift of mass onto
the **compliance** heads, for identical content. Paired over content items (McNemar / paired-t).

### M2 — causal confirmation (mechanism-derived prediction). *Medium.*
2 (framing: request | peer) × 4 (steer: none | refusal-only | compliance-only | bipolar). Measure ASR.
**Prediction (interaction):** compliance-only steering reduces ASR *much more under peer framing than
under request framing*; refusal-only shows the opposite or a weaker gap. A significant
framing×condition interaction is the causal claim that propagation is compliance-mediated.

### C1 — propagation chains. *Medium.* (makes it multi-agent, not two-agent)
Chain of N agents, compromised head A seeded with a goal; each agent processes upstream message(s)
and passes output downstream.
- **Cascade (unpaired):** run end-to-end undefended; ASR **per hop** → does it amplify or decay?
- **Placement (paired per hop):** freeze the undefended chain, then at each position measure the
  agent's response to its *fixed* upstream context under each steer condition → where does a single
  defended agent help most?
- **Herd immunity:** defend a random subset of size m (bootstrap over subsets); ASR at the terminal
  agent vs m → minimal defended fraction for network safety.

### B1 — baselines. *Medium.*
At receiver B, compare: activation-steering (ours, additive sign-correct) · prompt-level "distrust
peer input" system prompt · the **CB checkpoint** (swap B → `GraySwanAI/…-RR`). Same benchmark, same
HarmBench judge. Reports training-free parity (or honest loss) + whether a rerouting-trained model
resists propagation.

### B2 — adaptive attacker. *Medium.*
Compromised A observes B's (steered) refusals and re-strategizes over K rounds. ASR under adaptation.

---

## 3. Harness architecture

Built on the existing `bsc` apparatus; **reuses** `ModelBundle`/`load_model`, `HeadIntervention` +
`applied`, `build_bipolar_edits`, `fit_head_directions` (from `multiagent_propagation`), `run_trials`,
`HarmBenchJudge`, `CircuitMap`, `metrics`. New code is small and additive.

### 3.1 New module `bsc/agents.py` — the network primitive
```
@dataclass
class Agent:
    bundle: ModelBundle
    role_system: str | None            # role/system prompt (planner, executor, peer, ...)
    defense: DefenseSpec | None        # None = undefended; else a steering/prompt/none spec
    name: str

@dataclass
class DefenseSpec:                     # one object so every defense goes through one code path
    kind: str                          # "steering" | "prompt" | "none"
    edits: dict[Head, HeadEdit] | None # for kind="steering" (built via condition_edits)
    system_suffix: str | None          # for kind="prompt" (e.g. distrust-peer instruction)
    positions: str = "last"

def agent_reply(agent, messages, gen_cfg, *, seed=None) -> Generation:
    # renders messages through agent.bundle chat template, applies agent.defense
    # (HeadIntervention via applied(), or a system-prompt suffix), returns a Generation.

def run_chain(agents, seed_message, gen_cfg, *, per_agent_seed) -> ChainRun:
    # pass messages down a chain; each agent sees the upstream transcript; returns every hop.
```
`ChainRun` records, per hop: the agent name, its input transcript, its output `Generation`, and the
HarmBench verdict (scored against the goal). Freezing a `ChainRun` gives the fixed contexts the paired
per-hop evaluation reuses.

### 3.2 New module `bsc/probes.py` — the mechanism instruments
```
REFUSAL_OPENERS  = ("I", "Sorry", "As", "Unfortunately", "cannot", ...)   # first-token ids
COMPLY_OPENERS   = ("Sure", "Here", "Certainly", "Step", ...)

@dataclass
class RefusalReadout:
    refusal_logit_diff: float          # Σlogit(refusal openers) − Σlogit(comply openers) @ pos 0
    refusal_head_mass: float           # projection mass on refusal-head refuse−comply directions
    compliance_head_mass: float        # ... on compliance-head directions
    first_token: str

def refusal_readout(bundle, prompt, head_directions, refusal_heads, compliance_heads) -> RefusalReadout
    # one forward pass (no generation); reuses the o_proj-input capture from fit_head_directions
    # and the model's next-token logits. Pure measurement — no hooks that edit.
```
This is the M1/M2 instrument and is independent of generation, so it is cheap and deterministic.

### 3.3 New experiments (each `run(cfg) -> dict`, manifest via `RunContext`)
- `bsc/experiments/peer_vs_request.py` — M1 (+ optional M2 if `intervention.enabled`).
- `bsc/experiments/propagation_chain.py` — C1 (cascade + placement + herd-immunity).
- `bsc/experiments/multiagent_baselines.py` — B1 (+ B2 behind a flag).
The existing `multiagent_propagation.py` becomes the two-agent special case / is refactored to call
`bsc.agents` so there is one message-passing path.

### 3.4 Data model for framings (M1/M2)
A `FramedItem` carries: `goal` (HarmBench behavior), `harmful_ask` (the identical span), a
`request_prompt` and a `peer_prompt` built from templates. The peer transcript's upstream turns are
generated by a live peer agent (as in `build_transcript`) or templated; either way the **harmful ask
is string-identical** to the request version — verified in code, recorded in the artifact.

### 3.5 Statistics (unchanged discipline, CLAUDE.md §1.5, §2)
- Unit = **content item / scenario**, never trial. Wilson CI on every rate.
- **Paired McNemar** for paired comparisons (M1 request-vs-peer; per-hop placement). Unpaired
  comparisons (cascade placement, herd-immunity, B1 across models) report the **difference with a
  bootstrap CI**, not McNemar.
- M2 interaction: report ASR in each of the 8 cells + the framing×condition interaction (bootstrap CI
  on the difference-of-differences).
- **Degeneration column always** (keyword-NONRESPONSE share) next to every HarmBench ASR.
- Herd-immunity curve: bootstrap over defended subsets, CI band.

### 3.6 Contamination guards (CLAUDE.md §2, plan §5)
- Attack goals: AdvBench slice `ATTACK_SLICE_START=120`, disjoint from circuit-discovery (0:40) and
  the additive fit slice `FIT_SLICE_START=40`.
- M1's two framings share **identical harmful content**; only the frame differs (asserted in code).
- Any steering direction is fit on a slice disjoint from every evaluated goal.

### 3.7 Judge (instrument of record)
`HarmBenchJudge` scored against the true goal for every measured hop/response; keyword judge kept only
for the degeneration/NONRESPONSE split. A generative-LLM judge is NOT used (noisier than the classifier).

---

## 4. Models and scope
- Receiver B (primary): `Qwen/Qwen2.5-7B-Instruct` (has a validated 7B circuit map).
- Receiver B (secondary): `Meta-Llama-3-8B-Instruct` (ungated mirror) for cross-model.
- Attacker / peer A: a distinct capable instruct model (e.g. Qwen2.5-7B or larger) with a
  compromised-peer system prompt; A≠B to avoid self-attack artifacts.
- B1 CB arm: `GraySwanAI/Llama-3-8B-Instruct-RR` (dense; the only checkpoints that exist).
- Chain N: start N=4. Goals: 20–30 (statistical floor for scenario-level CIs).

## 5. Build order (compute-free first)
1. `bsc/probes.py` + `bsc/agents.py` (+ CPU unit tests: identity no-op defense, chain wiring, probe
   readout shape, framing string-identity assertion). **No GPU.**
2. `peer_vs_request.py` (M1) — smallest, highest-novelty. Dry-run on CPU.
3. `propagation_chain.py` (C1).
4. `multiagent_baselines.py` (B1/B2).
5. One GPU session: run M1 → M2 → C1 → B1 on Qwen2.5-7B, commit artifacts (never leave runs on the
   sandbox again).

## 6. The one figure (candidate)
Two panels: (left) M1 — refusal rate & refusal-logit-diff, request vs peer, for identical content
(the mechanism); (right) C1 — herd-immunity curve, terminal-agent ASR vs fraction of agents defended,
undefended cascade as the ceiling. A reader should get "refusal is frame-conditional, and steering a
few agents' compliance circuit contains the contagion" without the caption.

---

## 7. Literature review & positioning
*(Filled from the review agent — deltas, novelty verdict, standard benchmarks, reviewer objections.)*

TO BE INSERTED.
