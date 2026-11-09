# Learning the Wrong Lesson: Deceptive Reward Shaping for Securing Agentic Information Systems

Code for every evaluation in the CAiSE 2027 proposal: the synthetic theory studies, the live
AgentDojo studies with Claude, and all tables (CSV + LNCS-ready LaTeX) and figures (PDF + PNG).

**The only thing you must change:** your Anthropic API key in `config.yaml`
(`anthropic_api_key: "..."`), or `export ANTHROPIC_API_KEY=...` instead.

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
python -m pytest -q                                    # 7 correctness tests, ~10 s
```

## 2. Run

| Step | Command | API cost | Time |
|---|---|---|---|
| Smoke test (synthetic) | `python run_all.py --synthetic --quick` | $0 | ~1 min |
| Synthetic study + ontology study, paper scale (E1, E2, E3s, E4, E7) | `python run_all.py --synthetic` | $0 | ~45 min on 2 cores, ~12 min on 8 |
| Live pipeline, small slice, scripted agent | `python run_all.py --live-mock` | $0 | ~1 min |
| Predicted cost of your live plan | `python run_all.py --estimate` | $0 | seconds |
| Pilot (tiny real run) | `python run_all.py --live --pilot` | ~$0.10 | ~10 min |
| Full live campaign, default agent profile | `python run_all.py --live` | ~$2 (cap $15) | ~1–2 h API + ~30 min CPU on 8 cores |
| Same for the permissive profile (`agent.profile: permissive`) | `python run_all.py --live` | ~$2 | same |
| Rebuild live tables/figures | `python run_all.py --report` | $0 | seconds |

Order inside `--live`: **E0 → E5 → E3 → E3v → E2 → E7**. E0 must finish first: it produces
`results/live/calibration.json`, which adds the calibrated condition to E3.

Live runs are **resumable**. Real-agent episodes (E0, E5) are written one by one; an
assume-breach run is written when it completes. A re-run skips finished work. Logs from an
earlier version of the design are moved aside automatically to `*.old.jsonl`, so they are
never mixed in. The cap `budget.max_usd` stops the run before overspending.

Speed: assume-breach runs make no API calls and run in parallel on all cores but one
(`workers:` in `config.yaml`). `CARS_WORKERS=8 python run_all.py --synthetic` does the same for
the synthetic study.

## 3. Evaluation design: three kinds of evidence

The design keeps two questions apart: **how often the real agent is steered** (agent
vulnerability) and **how well each policy contains harm once it is steered** (defense
effectiveness).

| Layer | Experiment | Method | What it establishes |
|---|---|---|---|
| L0 natural steering | **E0** | Real Claude agent on AgentDojo tasks, template injections, plain blocking. Tasks split into **fit / held-out** sets | How often this model attempts the injection's goal: k / n, Wilson 95% CI, per avenue only when n ≥ 30, held-out validation. A near-zero rate is reported as a finding, not tuned away |
| L0 benign safety | **E5** | Real agent, **clean** tasks and **contaminated** tasks (legitimate task + untrusted injected data) reported separately | R1 invariant (no fabrication in trusted flows), utility, false routing of the user's own benign calls |
| L1 calibrated assume-breach | **E3** (P3) | Real AgentDojo environment, real gate and shadow twins; the agent's decision to follow is drawn with probability p. **Calibrated condition**: p is drawn per run from E0's Beta posterior, so calibration uncertainty propagates. **Sensitivity sweep**: p ∈ {0.1 … 1.0} | Conditional containment: real compromises, deception budget, exposures, utility. All policies face **paired** scenarios (same tasks, compliance draw and per-round steering draws) → paired differences and Wilcoxon tests |
| L1 | **E3v** (P1 live) | c ∈ {0.5, 1.0} × pre-registered attacker verification v ∈ {0.1, 0.3, 0.5, 0.8}, all deception policies, UCB1 + Skeptical-UCB, paired | Whether CARS's credibility constraints beat budget-matched fixed deception once attackers verify (the theory's prediction). Report every cell |
| L1 | **E2** (P2) | Equal total rounds; arms: 1 attacker, 4 independent, 4 sharing **statistics only**, 4 sharing **statistics + decoy identities** | Separates the effect of more observations from the effect of sharing |
| L1 | **E7** | Flat KB vs. ontology vs. oracle on Slack (unmodelled) + **decision audit** | Which CARS decisions (response form, alert label) the ontology changes, and the paired outcome effect on rounds where decisions differ |
| L1 | Adaptation table (from E3 logs) | P(attacker switches avenue next round \| feedback) | Whether the feedback channel changes the learner's behaviour: success, failure, believed fabrication, exposed fabrication |

**Policies:** blocking only; static honeytool; fixed reward shaping (the original paper);
**fixed shaping with a CARS-matched budget** (fabricates at CARS's 0.50 statistical cap but
with no credibility constraints, so a CARS advantage cannot be "more lies"); state-consistent
adaptive deception (a **simplified re-implementation** of the AgentSnare idea, marked *); CARS
without verification modelling; full CARS. Attacker learners: UCB1, Thompson, EXP3,
Suspicious-UCB, Skeptical-UCB (all in the calibrated condition; UCB1 on the sweep).

**Attacker text.** Fixed templates for every avenue are the reproducible baseline. The share of
attack text from each source is printed by the report; with the default mode it is 100%
templates and no substitution happens.

**How to state the result.** Results from L1 are *conditional*: "CARS reduces harm after an
unauthorized action attempt under modelled steering", not "CARS reduces harm from naturally
occurring live-agent attacks". If E0 is near zero, say so and present E3 as a conditional
stress test.

Synthetic studies (unchanged): P1 credibility boundary (E1), P2 collective verification (E2),
P3 policy value against 7 bandit attackers (E3s), ablation (E4), ontology knowledge quality and
downstream effect (E7a/b).

## 4. How CARS is implemented

* **Provenance gate** (`cars/live/gate.py`). A write call is tainted if an argument carries an
  identifier that appeared in untrusted tool output. Identifiers from the user's prompt and
  from the organisation's known-entity registry are excluded. Trusted flows therefore cannot
  be tainted (the R1 invariant). Detection is imperfect by design: identifiers the agent
  reassembles from fragments escape it (ρ < 1).
* **Shadow twins** (`cars/live/executor.py`). A tainted call runs the *real* tool on a deep copy
  of the environment, so a fabricated output has exactly the real format. With the semantic
  dimension on, the copy persists for the episode as a deception ledger, and later reads are
  served from it. The real environment never changes.
* **Credibility gate and learning layer** (`cars/live/policies.py`, `cars/synthetic/policies.py`):
  * *stat* caps the fabricated success rate at the estimated plausibility threshold;
  * *ver* selects "success" or "pending" forms using per-tool verifiability tags
    (`cars/live/knowledge.py`) once verification behaviour is observed, and rotates exposed
    decoys;
  * *pop* gives each attacker session its own decoy identities.
* **Honeytools**: decoy tools added to the agent's tool list (names drawn per identity).
* **Ground truth**: a real compromise is judged by AgentDojo's security check on the **real**
  environment and on the trace of calls that actually executed. Shadowed, blocked and
  honeytool calls are removed, so a fabrication can never count as a compromise
  (tested in `tests/test_core.py`).
* **Learning attacker** (`cars/live/attacker.py`) has six avenues and three modes:
  * `llm`: Claude chooses the avenue from summarised statistics and writes the text;
  * `ucb_llm`: UCB1 chooses the avenue and Claude writes the text;
  * `ucb_template`: no API calls.

  After a reported success the attacker verifies out of band with probability v. Exposures go
  into its own or shared memory. If Claude declines or returns malformed output, the avenue
  template is used instead and counted (`attack_source = fallback`).

## 5. The ontology (E7)

`cars/ontology/cars.ttl` is an OWL/RDF ontology of agent actions. Its classes include
FinancialTransfer, ScheduledPayment, CredentialChange, AccessGrant, DataDisclosure,
BroadcastDisclosure, CalendarManipulation, Destruction, ContentModification and others. Each
class carries the knowledge CARS needs for credibility-aware reward shaping:

* **verifiability**: how conclusively an attacker can confirm the effect out of band;
* **ATLAS technique**: the label used for the SOC;
* **decoy family**: what kind of decoy imitates this action.

`cars/ontology/reasoner.py` (rdflib) infers these for any tool:

* asserted tools (the two modelled systems, banking and workspace) take their class directly;
* unmodelled tools are classified lexically from their names, with values inherited up the
  `rdfs:subClassOf` hierarchy.

The ontology feeds reward shaping in three places:

1. which fabricated form to use ("success" or "pending");
2. on which avenues to stop fabricating once attackers are seen verifying (verifiability ≥ 0.7);
3. how to assign and rotate decoy families (spread across families; prefer unexposed, less
   verifiable ones).

The comparison is a flat lookup table (unknown tool → 0.5, generic ATLAS label), the ontology
with estimates only, the ontology with estimates plus decoy families, and oracle knowledge.

**Label caution.** The AgentDojo labels and the ontology were written by the same authors, so
E7a on AgentDojo tools is an upper bound. `heldout_tools.csv` holds 30 tool names from other
ecosystems. Their labels were fixed before the classifier was run on them and were never used
to tune it, but they are still provisional author labels. For the paper, have 2–3 security
experts fill `cars/ontology/annotation_template.csv`, then run
`python -m experiments.annotation_agreement annotations/*.csv`. It reports Fleiss' kappa and
re-scores the ontology against their consensus.

## 6. Cost

Only the real-agent layer calls Claude (Haiku 5.5: $0.10 in / $0.50 out per MTok; editable in
`config.yaml`). From `python run_all.py --estimate`, per agent profile:

| Experiment | Episodes | USD |
|---|---|---|
| E0 steering, banking + workspace (fit + held-out, 6 avenues, 2 reps) | 192 | ≈ $0.51 |
| E5 clean, all user tasks × 7 policies | 392 | ≈ $0.98 |
| E5 contaminated, 6 tasks × 2 avenues × 7 policies × 2 suites | 168 | ≈ $0.47 |
| **Total per profile** | **752** | **≈ $1.96** |

Both profiles: ≈ $3.90; plan for ≈ $5.50 with retries. The ≈ 89,000 assume-breach episodes
(E2, E3, E7) cost nothing; they take ≈ 3 CPU-hours, i.e. ≈ 25 minutes on 8 cores.

## 7. Assumptions and limitations (state these in the paper)

1. **Mock mode is plumbing, not evidence.** `--live-mock` uses a scripted agent with assumed
   compliance probabilities. Only `--live` results are findings.
1b. **Assume-breach is conditional evidence.** E3/E2/E7 treat the agent's decision to follow
   an injection as a probability (calibrated from E0, plus a sweep). It assumes this probability
   does not depend on the defender policy; that holds by construction within one episode (the
   agent decides before seeing any defender response) but is a modelling assumption across
   rounds. The learning attacker is a bandit over fixed templates, not an LLM: the adaptation
   evidence is about learning dynamics, not about LLM attacker behaviour.
1c. **Goal-attempt measure.** A goal attempt is a honeytool call, or a call to the goal function
   that is tainted by untrusted data or carries one of the injection's own target values. A user
   task that legitimately uses the same function (e.g. `send_money`) is not counted.
2. **Attacker observation channel.** The attacker is assumed to observe the outcome the tool
   reports for its goal action, and verification is an oracle check of the real environment
   with probability v. The defender is assumed to observe verification attempts on
   fabricated artefacts (for example, a fabricated reference being used).
3. **One target-model family.** Only a Claude agent is measured in E0/E5; report other model
   families as future work.
4. **CaMeL** is not executed. Blocking with our provenance gate is the "block" reference;
   cite CaMeL for the gate design.
5. **Rotation and learner optimism.** UCB treats a rotated decoy as a new arm and is optimistic
   about it, which flatters rotation. `Skeptical-UCB` removes this (a re-learned arm starts
   pessimistic). Report it next to UCB1. CARS also has a liveness check: a decoy that an
   attacker stops using for `abandon_window` pulls is presumed exposed and rotated.
6. **Robust bandit attacker** is BARBAR (Gupta, Koren & Talwar, 2019) with scaled constants,
   not Lykouris et al. (2018).
7. **ATLAS technique IDs** in `cars/live/knowledge.py` must be checked against the current
   MITRE ATLAS release.
8. **Synthetic parameters** (detection rates, verifiability, pending hazard, generalisation
   probability) are assumptions, listed in `cars/synthetic/sim.py` (`SimConfig`, `DEFAULT_ARMS`).
   Report a sensitivity analysis for any that drive a conclusion.
9. **Ethics.** All attacks run inside AgentDojo's synthetic sandbox; no real accounts, people
   or systems are involved. CARS deceives only traffic aimed at systems the defender owns.

## 8. Layout

```
config.yaml                      API key, models, experiment sizes, budget cap, prices
run_all.py                       single entry point
cars/synthetic/attackers.py      UCB1, Thompson, EXP3, eps-greedy (const/decaying), Suspicious-UCB, Skeptical-UCB, BARBAR
cars/synthetic/policies.py       six policies + ablations; CARS learning layer
cars/synthetic/sim.py            environment, verification hazard, sharing, rotation, metrics
cars/live/knowledge.py           tool verifiability + ATLAS tags, identifier extraction
cars/live/gate.py                provenance gate
cars/live/executor.py            CARS tool executor: shadow twins, honeytools, ledger
cars/live/policies.py            live policies, decoy identities, rotation
cars/live/attacker.py            learning / verifying / sharing attacker
cars/live/runner.py              one AgentDojo episode + ground-truth accounting + mock agent
cars/live/llm.py                 Claude calls, token/cost tracking, spending cap
cars/ontology/cars.ttl           OWL/RDF action ontology (verifiability, ATLAS, decoy families)
cars/ontology/reasoner.py        rdflib reasoner: asserted + lexical classification, inheritance
cars/ontology/heldout_tools.csv  30 held-out tool names for E7a
cars/ontology/annotation_template.csv  form for independent expert labels
experiments/                     synthetic_experiments, ontology_experiments, live_experiments,
                                 live_report, estimate_cost, annotation_agreement
results/  tables/  figures/      outputs
```
