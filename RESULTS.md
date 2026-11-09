# Results of the synthetic and ontology studies (paper scale, 30 seeds per condition)

These come from `python run_all.py --synthetic` (budget 20,000 attacker interactions per run
unless stated). Every number is in `tables/`, and every figure is in `figures/` as PDF and PNG.
The live AgentDojo results appear once you run `python run_all.py --live` with your API key.
The `*_mock` tables only test the pipeline and are **not findings**.

## P1: the credibility boundary is predictable (E1)
* Decoy lifetime follows the prediction 1/h, h = f·v·v_b, across two orders of magnitude for
  UCB1 and Thompson attackers. Observed/predicted ratios range from 0.75 to 1.26 over 24
  conditions, with confidence intervals covering the prediction in most of them
  (`table_e1_lifetime`, `fig_e1a_lifetime`).
* Without verification-aware rotation, deception loses its value at very low verification
  rates: v* = 0.005 for every attacker, at which point the reduction vs. blocking falls below 2x.
* Full CARS keeps about a 2.5x reduction up to v = 0.5 against the UCB-family attackers
  (UCB1, Suspicious-UCB, Skeptical-UCB). It does **not** keep it against EXP3 or the
  corruption-robust BARBAR attacker, where v* ≤ 0.005 (`table_e1_boundary`, `fig_e1b_boundary`).
  Report this limit.

## P2: collective verification matters beyond the budget (E2)
* At an equal total budget, the first exposure is budget-driven. It stays near 1/h whatever N
  and whatever the sharing mode (`fig_e2_collective` a).
* With a shared decoy identity, attacker sharing significantly increases real compromises:
  +1,292 to +3,769 at N = 2–8, all with 95% CIs excluding 0 (`table_e2_sharing_effect`).
  So the null hypothesis "sharing only adds verification opportunities" is rejected in this model.
* Session-unique decoys (the population dimension) reduce the sharing penalty, but not
  uniformly. With shared memory it remains significant for Thompson (+2,230 to +3,120), and
  one condition reverses (UCB1, N = 8, exposure sharing: −260).
* In the parallel schedule, wall-clock time to first exposure scales as 1/N. That is a rate
  effect, not collective learning (`fig_e2_parallel_rate`).

## P3: policy value (E3, synthetic)
Real compromises, mean over 8 attacker types, each cell a mean over 30 runs (`table_e3_objective`):

| Scenario | Block | Static honeytool | Fixed shaping | State-consistent* | CARS w/o verif. | CARS |
|---|---|---|---|---|---|---|
| Single attacker, v = 0 | 5,908 | 5,847 | 907 | 3,440 | 606 | **571** |
| Single attacker, v = 0.05 | 5,908 | 5,847 | 5,829 | 5,620 | 5,559 | **3,481** |
| 4 sharing attackers, v = 0.05 | 5,908 | 5,847 | 5,829 | 5,621 | 5,448 | **3,470** |

Once attackers verify even 5% of successes, every policy without verification-aware rotation
falls back to roughly blocking-level harm. CARS keeps about a 41% reduction.
*Simplified re-implementation, not equivalent to AgentSnare.

## Ablation (E4): not every credibility dimension earns its place
4 sharing attackers, v = 0.05, templated inconsistency 0.003 (`table_e4_ablation`):
* **Verifiability**: removing it raises compromises by +2,034 to +4,178. Clearly needed.
* **Semantic consistency**: removing it raises compromises by +357 to +485. Needed.
* **Population**: removing it raises compromises by about +1,600 for the two UCB attackers but
  lowers them for Thompson (−279). Mixed.
* **Statistical cap**: removing it *lowers* compromises for every attacker, including the
  suspicious one (e.g. UCB1 1,506 → 20). When rotation exists, a suspicious attacker's flag
  just triggers a rotation, so capping the fabricated rate only costs engagement. Report this
  honestly. Either drop the statistical dimension from the claims, or restrict it to settings
  where rotation is costly or unavailable.

## E7: does an ontology help reward shaping?
* **Knowledge quality: yes** (`table_e7a_knowledge`):

| Tool set | Verifiability error, flat → ontology | Correct ATLAS label, flat → ontology | Correct block/fabricate decision, flat → ontology |
|---|---|---|---|
| AgentDojo systems never modelled (Slack, travel) | 0.33 → 0.02 | 42% → 92% | 17% → 100% |
| 30 held-out tools from other ecosystems | 0.36 → 0.16 | 43% → 83% | 27% → 73% |

  The held-out labels are provisional author labels; replace them with expert annotations
  (`experiments/annotation_agreement.py`).
* **Downstream effect on compromises: no** (`table_e7b_downstream`, `fig_e7b_downstream`).
  None of the 54 comparisons with v > 0 against the flat KB is significant, and oracle
  knowledge does not beat the flat KB either. So in this model, verifiability knowledge is not
  the bottleneck for reward shaping. The ontology's demonstrated value is in labelling and
  decision quality for the SOC, not in fewer compromises. Ontology-guided decoy-family design
  also showed no benefit.
* An unexplained observation for follow-up: family-level generalisation by attackers
  (g = 0.5) *lowers* compromises under CARS. This is likely through more frequent
  verification, faster defender learning and more rotations. Check this before interpreting it.

## What this means for the paper
The strongest claims are P1 (a predictable credibility boundary), P2 (a collective effect
beyond the budget, mitigated but not removed by session-unique decoys), and verification-aware
rotation as the dimension that matters. The statistical cap and the ontology's effect on
outcomes did not hold up and should be framed as negative results. The live runs decide
whether these synthetic findings transfer to LLM attackers.
