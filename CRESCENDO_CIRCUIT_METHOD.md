# Finding the circuit for multi-turn attacks (why GCG patching doesn't transfer)

Design note answering: *GCG has clean refuse-vs-comply contrast pairs, so activation patching
works. Multi-turn (Crescendo) attacks don't — in a successful escalation the model complies at
every turn. So how do we find the circuit, and where does CCA fit?*

## The problem, precisely

Harm-contrastive activation patching needs two runs that differ minimally, one refusing and one
complying, so patching a head's clean activation into the corrupted run isolates that head's
causal contribution to refusal.

- **GCG:** natural pair — harmful prompt (refuses) vs harmful prompt + adversarial suffix
  (complies). Works.
- **Crescendo:** by the terminal turn the model complies, and there is no minimally-different
  refusing run to patch *from*. The legacy `crescendo_circuit_discovery_v3.py` built a
  cross-trajectory contrast (benign-filler history vs escalation history, same terminal request)
  and gated on "benign refuses AND escalation complies" — only **2 of 3** candidate pairs
  passed. Every published Crescendo circuit number rests on n=2.

## Why CCA is not, by itself, an answer

CCA (Canonical Correlation Analysis) finds a shared linear subspace maximizing correlation
between two activation sets — e.g. refuse-trajectory vs comply-trajectory representations. It is
**correlational / representational, not causal**. It can surface *a direction along which the two
trajectories differ*, but a direction being correlated with the behavior does not make it a
*cause* of it. Reporting a CCA direction as "the circuit" is the same category error the audit
flagged elsewhere: a description mistaken for a mechanism.

(If "CCA" meant **CAA — Contrastive Activation Addition** — that is a *steering* method built from
a difference-of-means direction. It proposes and applies a direction; it still needs causal
validation, and we test it directly in the `steering_defense` experiment.)

## The approach this repo takes

**Correlational discovery → causal validation.** Two stages, never one without the other.

### 1. Propose (representational)
Use a cheap representational method to nominate candidate directions/heads:
- difference-of-means between refuse- and comply-trajectory residuals (what `steering_defense`
  extracts), or
- CCA/SVCCA between the two trajectory sets, or a linear probe.

These are fast and need no clean patching pair. Their output is a *hypothesis*.

### 2. Validate (causal) — the part that actually localizes the circuit
The Crescendo mechanism is **KV-cache poisoning**: refusal heads attend to the model's own
prior-turn compliant tokens. So the causal probe is not head-activation patching (no clean run
exists) — it is **intervening on attention to source positions within the single complying run**:

> For each refusal head, down-weight (or zero) its attention mass on the *untrusted* token
> positions — the assistant-generated, prior-turn tokens — renormalize, and measure whether
> refusal is restored at the terminal turn.

This needs no refusing baseline. If down-weighting a head's attention to prior-turn tokens
restores refusal, that head is causally implicated, and the circuit is localized to
**(head × source-position)** — the multi-turn analog of the GCG per-head circuit, but indexed by
*where the head attends* rather than *what it writes*. The trust-tiered KV-partitioning PoC in the
legacy repo sketches the mechanism; turning it into a scored causal map is the planned
`crescendo_kv_circuit` experiment.

### 3. Fix the contrast, don't dodge it
Independently, scale the contrast set: the harvester generates 50+ Crescendo scenarios, so many
more pass the "benign refuses AND escalation complies" gate → real n instead of 2. A robust n=30
cross-trajectory patching result and the KV-attention causal map should agree; if they don't,
that disagreement is itself a finding.

## Summary

| | GCG (single-turn) | Crescendo (multi-turn) |
|---|---|---|
| Clean refusing baseline? | yes (harmful prompt) | no (complies every turn) |
| Circuit probe | head-activation patching | attention-to-source-position ablation |
| Indexed by | head (what it writes) | head × source-position (where it attends) |
| CCA/CAA role | — | *propose* directions; never the final circuit |
| Causal step | patch clean→corrupted | down-weight attention to untrusted tokens |

The one-line answer to "how do you ablate using CCA?": you don't ablate *with* CCA — CCA (or a
probe, or difference-of-means) **proposes** a direction, and you then **ablate attention to the
model's own prior-turn tokens** to test it causally. Correlational tools nominate; causal
interventions confirm.
