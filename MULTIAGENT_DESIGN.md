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

| id | claim | established by | novelty (per §7.3 review) |
|----|-------|---------------|------|
| C0 | Harmful content propagates agent→agent when re-framed as peer continuation | C1 baseline | phenomenon — **prior** (cite Prompt Infection 2410.07283, Agent Smith 2402.08567, TMCHT 2410.16155) |
| C1 | The **same content** triggers refusal as a *request* but not as a *peer message* | M1 | **partially scooped** (Safety Relay 2608.30585 causal; Continuation Framing 2608.08212 behavioral, *failed to find the mechanism*) → our angle = circuit-level in the peer regime |
| C2a | Continuation/compliance vs refusal heads exist and are causal | (prior) | **scooped — cite, don't claim** (Struggle 2603.08234, Head-Spec 2606.28153) |
| C2b | *Propagation* fires the **compliance/continuation** heads, not evasion of refusal heads | M1 | **novel** (no one links propagation to this circuit) |
| C3 | **Compliance-side** steering intercepts *propagation* more than *direct* attacks (interaction) | M2 | **novel** (causal) |
| **C4** | In a chain A→B→C→D, propagation amplifies/decays; steering **one** node's circuit halts network spread | C1-chain | **THE HEADLINE — genuinely open** (mechanism × propagation × network) |
| C5 | Activation steering is competitive with prompt/topology defenses, **AcMAS**, and the CB checkpoint, training-free | B1 | positioning |
| C6 | The interception survives an adaptive compromised peer | B2 | robustness |

**Honesty anchor (revised after the review).**
- **Headline is C4** (propagation × circuit × network steering) — the one unscooped intersection. **Do
  NOT claim "first activation-level MAS defense"** — AcMAS (2607.06807) already is one. Claim "first
  **circuit-grounded, refusal-restoring, propagation-intercepting**" defense; contrast with AcMAS's
  generic-anomaly-manifold steering.
- The heads (C2a) are **prior** — cite the Struggle/Head-Specialization papers, build on them.
- The framing idea (C1) is **partially scooped** — cite Safety Relay + Continuation Framing; our
  contribution is the *circuit-level account in the peer-contribution regime*, which 2608.08212's
  negative steering result leaves open.
- The steering *defense* is reported **degeneration-audited** (HarmBench ASR *and* keyword-NONRESPONSE
  share, always) — a mechanism-confirming intervention, not a SOTA claim; CB is strong and we don't
  claim to beat it.
- **Mandatory causal hygiene** (reviewer objection 3, our own scar): patch/ablate ablations, the
  identity-strength no-op unit test, and a **matched random-direction control** on every steering result.

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

### B1 — baselines. *Medium.* (expanded per §7.5)
At receiver B, compare on the same benchmark/judge/seeds:
- **ours** — additive sign-correct circuit steering;
- **random-direction control** — steer a matched-norm random direction on the same heads (isolates
  the circuit from generic activation perturbation; reviewer objection 3);
- **prompt-level** "distrust peer input" system prompt;
- **AcMAS-style restorative latent intervention** (2607.06807) — steer B's final-layer activations
  toward a normal manifold (the activation-defense counterexample; show it fails on framing-conditioned
  harmful content that is not an activation anomaly);
- **CB checkpoint** — swap B → `GraySwanAI/…-RR` (does a rerouting-trained model resist propagation?).
Reports training-free parity (or honest loss), the circuit-vs-generic-manifold contrast, and the
circuit-vs-random-direction contrast.

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
    layer_refusal_diff: list[float]    # LOGIT-LENS: refusal_logit_diff decoded per layer (optional)
    first_token: str

def refusal_readout(bundle, prompt, head_directions, refusal_heads, compliance_heads,
                    *, lens=None) -> RefusalReadout
    # one forward pass (no generation); reuses the o_proj-input capture from fit_head_directions
    # and the model's next-token logits. Pure measurement — no hooks that edit.
```
This is the M1/M2 instrument and is independent of generation, so it is cheap and deterministic.

**Logit-lens layer readout (M1 upgrade, optional).** `lens` decodes each layer's residual through the
unembedding (logit lens — no training, works on Qwen2.5-7B / Llama-3-8B; cf. LogitLens4LLMs) to a
per-layer `refusal_logit_diff`. The killer figure: the refusal signal is **present at mid-layers but
suppressed at the output under peer framing** (corroborating Safety Relay 2608.30585 / JailbreakLens
2411.11114). Treated as a *measurement instrument*: validate its per-layer diff against actual refusal
behavior before quoting (CLAUDE.md §2.3). Off the defense critical path — an M1 mechanism enhancement.

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
- Harmful base sets: AdvBench (already wired) + JailbreakBench/HarmBench for comparability to the
  mechanistic priors (2603.08234 / 2606.28153). Optional agent-native substrate: **Agent-SafetyBench
  (2412.14470)** / **AgentHarm** / **InjecAgent** (the last matches AcMAS for a head-to-head).
- Topologies (C1): line first, then star and dense (TMCHT's set) — report per-topology.

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

## 7. Literature review & positioning (2026-08-05)

**Verification note.** Every arXiv id below resolved this session unless flagged. **INFA-Guard** and
**PropGuard** (named in BIPOLARMULTIAGENT4DAY) could NOT be located — treat as hallucinated, do not
cite. Topology-guided "G-Safeguard"-type methods were not independently verified and are excluded.

### 7.1 Delta table

| Paper | arXiv | What they do | Level | Our delta |
|---|---|---|---|---|
| Prompt Infection | 2410.07283 | Self-replicating LLM→LLM prompt injection across a MAS; logistic spread; defense = LLM Tagging | Prompt | We give the *internal mechanism* of recipient compliance; they're black-box |
| Agent Smith | 2402.08567 | 1 image jailbreaks ~1M agents in ~27–31 rounds | Input/memory | We measure/steer at the circuit level; they only measure infection curves |
| Troublemaker / TMCHT | 2410.16155 | 1 attacker misleads a "town"; line/star topologies, 100 agents | Memory/RAG | Mechanism + activation defense; theirs has neither |
| CORBA | 2502.14529 | Contagious recursive *blocking* (denial-of-collaboration) | Prompt/topology | Different threat model (DoC, not harmful content) |
| ClawWorm | 2603.15727 | Self-propagating attacks; aggregate ASR ~64.5% | Prompt/tool | Behavioral spread only, no mechanism |
| Latent Attack in Latent MAS | 2605.28214 | **Activation attack**: extract attack directions, inject into node hidden states + edge KV handoffs; defense = detection only | Activation (attack) | Closest activation MAS work, but an *attack*+detector, never the refusal/continuation circuit |
| **AcMAS / "When Agents Go Rogue"** | **2607.06807** | **Activation MAS *defense***: anomaly detection + "restorative latent intervention" steering activations to a normal manifold | **Activation (defense)** | **The counterexample.** Steers to a *generic manifold*, not a refusal/continuation circuit; evals prompt-injection/tool/memory (InjecAgent/PoisonRAG), not framing-conditioned harmful-content refusal |
| AgentLens | 2606.22673 | Inference-time detect+steer in a 10-D safety subspace, single multi-turn coding agent | Activation (single-agent) | Not multi-agent/propagation; subspace not a head circuit; ~18.5% collapse cost |
| AutoDefense | 2403.04783 | Multi-agent response *filtering* (analyzer+judge); ASR 55.7%→7.95% | Prompt/orchestration | We defend inside activations, not via added filter agents |
| Foresight-Guided Defense | 2605.01758 | Predict a message's downstream effect before spread | Prompt/prediction | Prompt-space, no circuit |
| The Struggle: Continuation vs Refusal | 2603.08234 | Causal safety heads vs continuation heads + scaling defense (0.58→0.044) | Activation (single-model) | **Our core prior.** Single-model, no MAS, no framing, no propagation |
| Attention Head Specialization (ACH/SAH) | 2606.28153 | Attack-suppressed vs robust safety heads | Activation (single-model) | Corroborates safety heads; not agentic/framing |
| Safety Relay in Roleplay | 2608.30585 | Causal "safety-relay attenuation": harm still detected, refusal expression weakens; removing the change *restores refusal* | Activation (single-model) | **Sharpest framing prior**, but single-model roleplay, *no defense*, no propagation |
| Continuation Framing (Harmful Content Is Not Enough) | 2608.08212 | Identical harmful text as demo/tool-output (continue) vs document/request → +30–32pp; "content necessary but insufficient" | Behavioral | **Empirically scoops the framing claim** — but reports a **negative** steering result, calls it a behavioral boundary, *failed to find the mechanism*. We claim it |
| **JailbreakLens** (He, Wang, Chu et al., ZJU) | **2411.11114** | Representation+circuit analysis of jailbreaks on 5 LLMs, 7 strategies: jailbreaks **amplify affirmative components / suppress refusal components** | Activation (single-model, single-prompt) | **Partial scoop of the single-model mechanism** (found post-review) — cite and build on; we add the *peer-contribution framing*, *propagation*, and *network steering* they do not have |
| Arditi et al., Refusal = single direction | 2406.11717 | Difference-in-means refusal direction | Activation (single-model) | We use a framing-conditioned multi-head circuit in a *network*; this is our steering substrate |
| Circuit Breakers / RR | 2406.04313 | LoRA reroute harmful reps | Training | We are inference-time, targeted, multi-agent |
| Agent-SafetyBench | 2412.14470 | 349 envs, 2000 cases; MAS splits score *higher* risk | Benchmark | Adopt as eval substrate; extend with propagation metrics |

### 7.2 Verdict on the load-bearing claim
**"No prior MAS defense is activation-level" is FALSE — do not write it.** AcMAS (2607.06807) is a
genuine activation-level MAS defense; Latent Attack (2605.28214) does activation detection between
agents. **Defensible reframe:** *no prior work steers a mechanistically identified refusal/continuation
head circuit to restore content-based refusal and intercept harmful-content propagation across an agent
network; existing activation MAS defenses (AcMAS) steer toward a generic anomaly manifold on
injection/tool/memory tasks.* Not "first activation-level MAS defense" — "first **circuit-grounded,
refusal-restoring, propagation-intercepting** one."

### 7.3 Novelty split of our thesis
- **(a) refusal conditional on framing, not content** — *partially scooped.* Safety Relay (2608.30585,
  causal, single-model roleplay) + Continuation Framing (2608.08212, behavioral, *failed to find the
  mechanism*). A clean **circuit-level** account in the peer-contribution regime is still open; 2608.08212's
  negative steering result is an opening, not a wall.
- **(b) continuation vs refusal heads exist and are causal** — *scooped* (2603.08234, 2606.28153; and
  **JailbreakLens 2411.11114** — jailbreaks amplify affirmative / suppress refusal components, single-model
  single-prompt). **Cite and build on; do not claim discovery.**
- **(c) propagation is mediated by that circuit; steering one node intercepts network propagation** —
  **GENUINELY OPEN.** No paper connects agent-to-agent propagation to any internal circuit. **This is the
  headline.**

### 7.4 Benchmarks / judge / protocol to adopt
- Harmful base sets: **AdvBench / JailbreakBench / HarmBench / StrongREJECT / MaliciousInstruct** (match
  2603.08234 / 2606.28153 for direct comparability to our mechanistic priors).
- Agent safety: **Agent-SafetyBench (2412.14470)**, **AgentHarm** (HarmScore + RefusalRate; its own
  finding that message-monitoring is the main defense is a baseline to beat), **InjecAgent** (matches
  AcMAS, enabling head-to-head).
- Judge: LLM/rubric — our **HarmBench classifier** is fine; hold one judge fixed across all conditions;
  validate the keyword fast-path against it and report agreement (CLAUDE.md §2.3).
- Propagation metrics: network-level ASR/compromise rate, **infection rate over rounds**, **hops-to-
  saturation** (Agent Smith's 27–31-round curve is the reference), across **line/star/dense** topologies.

### 7.5 Three reviewer objections + preemption
1. *"AcMAS already does activation MAS defense."* → reframe as circuit-grounded refusal restoration vs
   generic manifold; **run AcMAS as an in-process baseline** (same seeds/judge) and show it fails on
   framing-conditioned harmful content that isn't an activation anomaly; show our intervention is a
   **no-op at identity strength** (their manifold steering has no such guarantee + collapse cost).
2. *"Just Arditi / continuation-head papers applied to agents — incremental."* → the **causal propagation**
   result: steer agent B only, intercept A→B→C spread — a network counterfactual no single-model paper
   can produce; cite 2608.08212's negative steering result to argue the mechanism was non-obvious.
3. *"Correlation not causation"* (our own scar, RESEARCH_OVERVIEW §2.3). → mandatory **patch/ablate**
   (path patching à la 2603.08234), **identity-strength no-op** unit test, **matched random-direction
   control**, greedy reported separately from n sampled + bootstrap CIs; show refusal is *restored*, not
   just norm-shifted.

### 7.6 Sharpest surviving framing
> Safety training binds refusal to request-framing rather than content, so harmful material arriving as a
> peer agent's contribution rides the model's continuation circuit past refusal — and steering that
> circuit at inference time, in one node, restores content-based refusal and halts jailbreak propagation
> across the agent network: the first circuit-level (rather than behavioral or generic-anomaly) account of
> and defense against multi-agent contagion.
