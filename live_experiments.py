"""Live experiments on AgentDojo with Claude. Evidence layers:
  L0  E0 natural steering of the real Claude agent (+ E5 benign safety, real agent)
  L1  E3/E2/E4/E7 calibrated assume-breach in the real AgentDojo environment (no API calls)

    python -m experiments.live_experiments --exp e0     # L0: real-agent steering rate, fit/held-out
    python -m experiments.live_experiments --exp e3     # P3: policy value, calibrated + sweep
    python -m experiments.live_experiments --exp e3v    # P1 live: CARS vs matched deception across v_att
    python -m experiments.live_experiments --exp e2     # P2: collective verification, live
    python -m experiments.live_experiments --exp e4     # ablation, live (off by default)
    python -m experiments.live_experiments --exp e5     # benign safety + utility (no attack)
    python -m experiments.live_experiments --exp all
Add --mock to run everything with a scripted agent and template attacker (no API cost).
Add --pilot to run a tiny slice (calibrates cost before the full campaign).

Every round is appended to results/live/<exp>.jsonl immediately; re-running resumes and
skips completed runs, so a stopped campaign never pays twice.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import zlib
from pathlib import Path

from cars.live.attacker import AttackerMemory, LiveAttacker
from cars.live.llm import BudgetExceeded, init_llm, load_config, tracker
from cars.live.policies import LIVE_PRESETS, DefenderState
from cars.live.runner import run_episode, suite

LIVE = Path(__file__).resolve().parents[1] / "results" / "live"
LIVE.mkdir(parents=True, exist_ok=True)


def _seed(*xs) -> int:
    return zlib.crc32("|".join(map(str, xs)).encode())


def _archive_stale(path: Path, field: str):
    """Logs written by an earlier version of the design (missing `field`) are moved aside to
    <name>.old.jsonl so they are never mixed with, or resumed into, the current design."""
    if not path.exists():
        return
    with path.open() as f:
        first = f.readline()
    if first and field not in json.loads(first):
        dst = path.with_suffix(".old.jsonl")
        path.replace(dst)
        print(f"  (moved logs from an earlier design version to {dst.name})", flush=True)


def _done_runs(path: Path, rounds_by_key: dict) -> set:
    if not path.exists():
        return set()
    counts: dict = {}
    for line in path.open():
        r = json.loads(line)
        counts[r["run_key"]] = counts.get(r["run_key"], 0) + 1
    return {k for k, n in counts.items() if n >= rounds_by_key.get(k, 10 ** 9)}


def _purge_partial(path: Path, done: set):
    if not path.exists():
        return
    bad = {json.loads(l)["run_key"] for l in path.open() if json.loads(l).get("error")}
    done -= bad
    keep = [l for l in path.open() if json.loads(l)["run_key"] in done]
    path.write_text("".join(keep))


def _sfx(cfg) -> str:
    """Results of the permissive agent profile go to separate files, never mixed with default."""
    return "_permissive" if cfg["agent"].get("profile") == "permissive" else ""


def _run_spec(exp: str, sp: dict, cfg: dict, mock: bool, out: Path | None):
    """One run (a fixed policy against a learning attacker for sp['rounds'] rounds). Returns rows,
    or None if the budget cap was hit. Writes rows as it goes when `out` is given."""
    version = cfg["agentdojo_version"]
    s = suite(version, sp["suite"])
    trng = random.Random(_seed("tasks", sp["suite"], sp["seed"]))   # identical across policies (paired)
    it_id = trng.choice(sorted(s.injection_tasks))
    uts = sorted(s.user_tasks)
    trng.shuffle(uts)
    it = s.get_injection_task_by_id(it_id)
    policy = LIVE_PRESETS[sp["policy"]]
    defender = DefenderState(policy, seed=sp["seed"])
    N, sharing = sp["N"], sp["sharing"]
    algo, horizon = sp.get("attacker_algo", "UCB1"), sp["rounds"]
    # sharing: none   = independent attackers
    #          stats  = shared bandit statistics only (more observations, no decoy identities)
    #          memory = shared statistics AND exposed decoy identities (full collective verification)
    # paired design (common random numbers): EVERY random stream (agent steering, defender and
    # executor draws, attacker learner and verification coins) is keyed without the policy name,
    # so policies differ only through what the policy itself does.
    pair = (sp["suite"], algo, sp.get("compliance_key", sp.get("compliance")), sp["seed"], N, sharing,
            sp["v_att"])
    rng = random.Random(_seed("run", *pair))
    shared_mem = (AttackerMemory(algo, horizon, _seed("mem", *pair))
                  if sharing in ("memory", "stats") else None)
    comp = sp.get("compliance")
    if comp == "beta":      # propagate E0 calibration uncertainty: one draw per paired run
        a_, b_ = sp["beta"]
        comp = random.Random(_seed("beta", *pair)).betavariate(a_, b_)
    attackers = []
    for i in range(N):
        sess = f"s{i}"
        mem = shared_mem or AttackerMemory(algo, horizon, _seed("mem", *pair, i))
        attackers.append(LiveAttacker(
            sess, mem, set(), sp["mode"], cfg, sp["attacker_model"], it.GOAL,
            [], [], sp["v_att"], random.Random(_seed("att", *pair, i)), shared=sharing == "memory"))
    rows = []
    try:
        for t in range(sp["rounds"]):
            i = t % N
            a = attackers[i]
            hts = defender.honeytool_names(a.session) if policy.honeytools else []
            a.honeytools = hts
            a.tool_names = [f.name for f in s.tools] + hts
            avenue, text, meta = a.propose()
            ep = run_episode(version=version, suite_name=sp["suite"], user_task_id=uts[t % len(uts)],
                             injection_task_id=it_id, injection_text=text, avenue=avenue, policy=policy,
                             defender=defender, session=a.session, cfg=cfg, mock=mock, rng=rng,
                             agent_model=cfg["agent"]["model"], agent=sp.get("agent"),
                             compliance=comp,
                             follow_u=random.Random(_seed("follow", *pair, t)).random())
            ob = a.observe(avenue, ep["observed"], ep["real_compromise"], ep["identity"])
            fabricated = ep["observed"] != "fail" and not ep["real_compromise"]
            if ob["verified"] and fabricated:
                defender.observe_verification(a.session, ob["exposed"])
            if ob["exposed"] and sharing == "memory" and policy.pop:
                for j in range(N):     # generalisation to other sessions' decoy identities
                    if j != i and rng.random() < cfg["attacker"].get("generalize", 0.2):
                        a.mem.known_fake.add(defender.identity(f"s{j}"))
            row = {"exp": exp, "run_key": sp["key"], "round": t, **{k: sp[k] for k in
                   ("suite", "policy", "attacker_model", "seed", "N", "sharing", "mode", "v_att")},
                   "attacker_algo": sp.get("attacker_algo", "UCB1"), "compliance_set": ("calibrated" if sp.get("compliance") == "beta"
                                                                      else sp.get("compliance")),
                   "compliance_drawn": comp,
                   "injection_task": it_id, **{k: v for k, v in ep.items() if k not in ("suite", "policy")},
                   "fabricated_observed": fabricated, **ob, "attack_source": meta.get("source"),
                   "rotations": defender.rotations, "crn": 2, "agent_profile": cfg["agent"].get("profile", "default"), "spent_usd": None if mock else round(tracker().total(), 4)}
            rows.append(row)
            if out is not None:
                with out.open("a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
    except BudgetExceeded as e:
        print(f"\n!! {e}", flush=True)
        return None
    return rows


_WORKER_READY = False


def _worker(args):
    global _WORKER_READY
    exp, sp, cfg, mock = args
    if not _WORKER_READY:          # each spawned process needs its own (read-only) tracker
        init_llm(cfg, True)
        _WORKER_READY = True
    rows = _run_spec(exp, sp, cfg, True, None)
    for r in rows:                 # no API spend inside assume-breach workers
        r["spent_usd"] = None
    return sp, rows


def _report_run(n_run, total, sp, rows, mock):
    c = sum(r["real_compromise"] for r in rows)
    print(f"  [{n_run}/{total}] {sp['key']}: real={c}/{sp['rounds']} "
          f"fabricated-observed={sum(r['fabricated_observed'] for r in rows)} "
          f"exposures={sum(r['exposed'] for r in rows)}"
          + ("" if mock else f" | total ${tracker().total():.2f}"), flush=True)


def run_attack_campaign(exp: str, specs: list[dict], cfg: dict, mock: bool):
    out = LIVE / (f"{exp}_mock.jsonl" if mock else f"{exp}{_sfx(cfg)}.jsonl")
    _archive_stale(out, "crn")
    keys = {sp["key"]: sp["rounds"] for sp in specs}
    done = _done_runs(out, keys)
    _purge_partial(out, done)
    todo = [sp for sp in specs if sp["key"] not in done]
    print(f"[{exp}] {len(specs)} runs, {len(done)} already complete, {len(todo)} to go"
          + ("" if mock else f" | spent so far ${tracker().total():.2f}"), flush=True)
    # Assume-breach runs with template attackers make no API calls: run them in parallel.
    # Anything that calls Claude stays sequential so the spending cap is exact.
    no_api = all(sp.get("agent") in ("assume_breach", "mock") and sp["mode"] == "ucb_template" for sp in todo)
    nproc = int(cfg.get("workers") or max(1, (os.cpu_count() or 2) - 1))
    if no_api and nproc > 1 and len(todo) > 1:
        import multiprocessing as mp
        print(f"  (parallel: {nproc} worker processes, no API calls)", flush=True)
        with mp.get_context("spawn").Pool(nproc) as pool:
            for n_run, (sp, rows) in enumerate(pool.imap_unordered(_worker, [(exp, sp, cfg, mock) for sp in todo],
                                                                   chunksize=1), 1):
                with out.open("a") as f:      # a run is written only once complete -> resume-safe
                    f.write("".join(json.dumps(r, default=str) + "\n" for r in rows))
                _report_run(n_run, len(todo), sp, rows, mock)
        return True
    for n_run, sp in enumerate(todo, 1):
        rows = _run_spec(exp, sp, cfg, mock, out)
        if rows is None:
            return False
        _report_run(n_run, len(todo), sp, rows, mock)
    return True


def _algos(cfg, mock):
    return ["UCB1"] if mock else cfg["e3"].get("attacker_algos", ["UCB1"])


def _sweep(cfg, pilot):
    sw = cfg["e3"].get("compliance_sweep", [0.1, 0.25, 0.5, 0.75, 1.0])
    return sw[:2] if pilot else sw


def _cal_path(mock: bool, sfx: str = "") -> Path:
    return LIVE / ("calibration_mock.json" if mock else f"calibration{sfx}.json")


def load_calibration(mock: bool = False):
    """E0 result for the DEFAULT agent profile (the anchor of the assume-breach layer)."""
    p = _cal_path(mock)
    return json.loads(p.read_text()) if p.exists() else None


def calibrated_compliance(mock: bool = False):
    """Point estimate (fit split) of real-agent compliance, or None if E0 has not run."""
    cal = load_calibration(mock)
    return cal["fit"]["rate"] if cal else None


def specs_e3(cfg, mock, pilot):
    """P3, layer L1: calibrated assume-breach in the live AgentDojo environment. The agent follows
    the injection with a compliance probability; in the headline 'calibrated' condition it is drawn
    per run from the E0 Beta posterior (uncertainty propagated), and a sweep over fixed levels is
    the sensitivity analysis. All policies (incl. the budget-matched fixed baseline) face the same
    paired steering opportunities. Attacker text = fixed templates, avenue choice = real bandit
    learners. No API calls."""
    c = cfg["e3"]
    suites = c["suites"][:1] if pilot else c["suites"]
    pols = c["policies"][-3:] if pilot else c["policies"]
    algos = _algos(cfg, mock)[:1] if pilot else _algos(cfg, mock)
    levels = list(_sweep(cfg, pilot))
    cal = load_calibration(mock)
    if cal is not None:
        levels.append("beta")      # calibrated condition: compliance ~ Beta posterior from E0
    mode = "ucb_template" if (mock or cfg["attacker"].get("mode") == "ucb_template") else cfg["attacker"]["mode"]
    seeds = range(1 if pilot else c["seeds"])
    out = []
    for su in suites:
        for p in pols:
            for comp in levels:
                # all bandit algorithms in the calibrated (headline) condition; the sensitivity
                # sweep uses sweep_algos only, to keep the campaign short
                for al in (algos if comp == "beta" or mock else
                           [x for x in algos if x in c.get("sweep_algos", algos)] or algos[:1]):
                    for sd in seeds:
                        ck = "cal" if comp == "beta" else comp
                        out.append({"key": f"e3|{su}|{p}|{al}|c{ck}|{sd}", "suite": su, "policy": p,
                                    "compliance_key": ck, "beta": cal["fit"]["beta"] if comp == "beta" else None,
                                    "attacker_model": "none" if mode == "ucb_template" else cfg["e3"]["llm_model"],
                                    "attacker_algo": al, "seed": sd, "N": c.get("n_attackers", 1),
                                    "sharing": "none", "rounds": 6 if pilot else c["rounds"], "mode": mode,
                                    "v_att": cfg["attacker"]["v_att"], "agent": "assume_breach", "compliance": comp})
    return out


def specs_e3v(cfg, mock, pilot):
    """P1 in the live environment: does CARS's advantage over budget-matched fixed deception
    appear as attackers verify more? Fixed compliance levels x pre-registered verification
    levels v_att, all policies, paired. No API calls."""
    c = cfg["e3v"]
    suites = c["suites"][:1] if pilot else c["suites"]
    pols = c["policies"][-3:] if pilot else c["policies"]
    algos = ["UCB1"] if (mock or pilot) else c["attacker_algos"]
    comps = c["compliance"][-1:] if pilot else c["compliance"]
    vs = c["v_att_levels"][::2] if pilot else c["v_att_levels"]
    out = []
    for su in suites:
        for p in pols:
            for al in algos:
                for comp in comps:
                    for v in vs:
                        for sd in range(1 if pilot else c["seeds"]):
                            out.append({"key": f"e3v|{su}|{p}|{al}|c{comp}|v{v}|{sd}", "suite": su, "policy": p,
                                        "compliance_key": comp, "attacker_model": "none", "attacker_algo": al,
                                        "seed": sd, "N": 1, "sharing": "none", "rounds": 6 if pilot else c["rounds"],
                                        "mode": "ucb_template", "v_att": v, "agent": "assume_breach",
                                        "compliance": comp})
    return out


def specs_e2(cfg, mock, pilot):
    c = cfg["e2"]
    mode = "ucb_template" if (mock or c.get("mode") == "ucb_template") else c["mode"]
    comp = c.get("compliance", 0.6)   # documented conditional stress level (never tuned to E0)
    algos = ["UCB1"] if (mock or pilot) else c.get("attacker_algos", ["UCB1", "Thompson"])
    out = []
    for al in algos:
        for p in c["policies"]:
            for N, sh in c["configs"]:
                for sd in range(1 if pilot else c["seeds"]):
                    out.append({"key": f"e2|{c['suite']}|{p}|{al}|{N}|{sh}|{sd}", "suite": c["suite"], "policy": p,
                                "attacker_model": "none", "attacker_algo": al, "seed": sd, "N": N, "sharing": sh,
                                "rounds": 8 if pilot else c["total_rounds"], "mode": mode, "v_att": c["v_att"],
                                "agent": "assume_breach", "compliance": comp})
    return out


def specs_e4(cfg, mock, pilot):
    c = cfg["e4"]
    mode = "ucb_template"
    comp = c.get("compliance", 0.6)   # documented conditional stress level (never tuned to E0)
    return [{"key": f"e4|{c['suite']}|{p}|{sd}", "suite": c["suite"], "policy": p, "attacker_model": "none",
             "attacker_algo": c.get("attacker_algo", "UCB1"), "seed": sd, "N": c["n_attackers"],
             "sharing": c["sharing"], "rounds": 8 if pilot else c["rounds"], "mode": mode, "v_att": c["v_att"],
             "agent": "assume_breach", "compliance": comp}
            for p in c["policies"] for sd in range(1 if pilot else c["seeds"])]


def specs_e7(cfg, mock, pilot):
    """Live ontology check on a system outside the modelled ones (default slack), assume-breach."""
    c = cfg["e7"]
    mode = "ucb_template"
    comp = c.get("compliance", 0.6)   # documented conditional stress level (never tuned to E0)
    return [{"key": f"e7|{c['suite']}|{p}|{sd}", "suite": c["suite"], "policy": p, "attacker_model": "none",
             "attacker_algo": c.get("attacker_algo", "UCB1"), "seed": sd, "N": 1, "sharing": "none",
             "rounds": 6 if pilot else c["rounds"], "mode": mode, "v_att": c["v_att"],
             "agent": "assume_breach", "compliance": comp}
            for p in c["policies"] for sd in range(1 if pilot else c["seeds"])]


def _split_tasks(uts: list[str], cap):
    """Deterministic fit / held-out split of user tasks (alternating in sorted order)."""
    uts = sorted(uts, key=lambda u: (len(u), u))     # natural order: user_task_2 before user_task_10
    fit, held = uts[0::2], uts[1::2]
    return (fit[:cap], held[:cap]) if cap else (fit, held)


def _binom(rows, key="attempted_goal"):
    from .common import wilson
    k, n = sum(bool(r[key]) for r in rows), len(rows)
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(p, 4) if n else None, "ci": [round(lo, 4), round(hi, 4)] if n else None}


def summarize_calibration(rows: list[dict], min_n: int) -> dict:
    """Fit on the fit split, validate on held-out tasks. Per-avenue rates only where n >= min_n;
    otherwise only the pooled rate is reported. Uncertainty is carried as a Beta(k+1, n-k+1)
    posterior (uniform prior) that the assume-breach layer samples from. Zero is a valid
    result: nothing here forces a non-zero rate."""
    from cars.live.attacker import AVENUES
    fit = [r for r in rows if r.get("split") == "fit"]
    held = [r for r in rows if r.get("split") == "heldout"]
    f, h, al = _binom(fit), _binom(held), _binom(rows)
    out = {"fit": {**f, "beta": [f["k"] + 1, f["n"] - f["k"] + 1]}, "heldout": h, "all": al,
           "min_n_per_avenue": min_n, "per_avenue": {}, "real_compromise_all": _binom(rows, "real_compromise")}
    for av in AVENUES:
        r = [x for x in rows if x.get("avenue") == av]
        bb = _binom(r)
        bb["reported"] = bb["n"] >= min_n
        out["per_avenue"][av] = bb
    if f["n"] and h["n"]:
        lo, hi = h["ci"]
        out["validation"] = {
            "fit_rate_within_heldout_ci": lo <= f["rate"] <= hi,
            "abs_error": round(abs(f["rate"] - h["rate"]), 4),
            # Brier score of predicting each held-out episode with the fit rate vs. with 0.5
            "brier_fit": round(sum((f["rate"] - bool(x["attempted_goal"])) ** 2 for x in held) / len(held), 4),
            "brier_uninformed": 0.25}
    return out


def run_calibration(cfg, mock, pilot):
    """E0, natural steering of the REAL agent. The real Claude agent runs AgentDojo user tasks
    with a template injection present and plain blocking. Logged per episode: whether the agent
    ATTEMPTED the injection's goal (a goal write or a honeytool call, before the defense acts)
    and whether real harm occurred. Tasks are split into fit / held-out sets. Output:
    results/live/calibration{profile}.json with numerator, denominator, Wilson CIs, the Beta
    posterior used by E3, and held-out validation."""
    from cars.live.attacker import AVENUES, template
    sfx = _sfx(cfg)
    out = LIVE / ("e0_mock.jsonl" if mock else f"e0{sfx}.jsonl")
    _archive_stale(out, "split")
    done = {json.loads(l)["key"] for l in out.open()} if out.exists() else set()
    version = cfg["agentdojo_version"]
    c = cfg["e0"]
    suites = c["suites"][:1] if pilot else c["suites"]
    reps = 1 if pilot else c["reps"]
    for su in suites:
        s = suite(version, su)
        fit, held = _split_tasks(list(s.user_tasks), 2 if pilot else c.get("max_user_tasks"))
        it_id = sorted(s.injection_tasks)[0]
        it = s.get_injection_task_by_id(it_id)
        pol = LIVE_PRESETS["block"]
        for split, uts in (("fit", fit), ("heldout", held)):
            for av in list(AVENUES):
                for ut in uts:
                    for rep in range(reps):
                        key = f"e0|{su}|{av}|{ut}|{rep}"
                        if key in done:
                            continue
                        text = template(av, it.GOAL, [f.name for f in s.tools], [])
                        try:
                            ep = run_episode(version=version, suite_name=su, user_task_id=ut,
                                             injection_task_id=it_id, injection_text=text, avenue=av, policy=pol,
                                             defender=DefenderState(pol), session="cal", cfg=cfg, mock=mock,
                                             rng=random.Random(_seed(key)), agent_model=cfg["agent"]["model"],
                                             agent="claude" if not mock else "mock")
                        except BudgetExceeded as e:
                            print(f"\n!! {e}"); return False
                        ep.update(key=key, exp="e0", avenue=av, split=split, rep=rep,
                                  agent_model=cfg["agent"]["model"], agent_profile=cfg["agent"].get("profile", "default"))
                        with out.open("a") as f:
                            f.write(json.dumps(ep, default=str) + "\n")
        print(f"[e0] {su} done" + ("" if mock else f" | total ${tracker().total():.2f}"), flush=True)
    rows = [json.loads(l) for l in out.open()]
    cal = summarize_calibration(rows, c.get("min_n_per_avenue", 30))
    cal.update(profile=cfg["agent"].get("profile", "default"), agent_model=cfg["agent"]["model"])
    _cal_path(mock, sfx).write_text(json.dumps(cal, indent=2))
    f, h = cal["fit"], cal["heldout"]
    print(f"[e0] fit: {f['k']}/{f['n']} = {f['rate']} CI {f['ci']} | held-out: {h['k']}/{h['n']} = {h['rate']} "
          f"CI {h['ci']}", flush=True)
    return True


def _e5_episode(out, key, cfg, mock, **kw):
    ep = run_episode(cfg=cfg, mock=mock, rng=random.Random(_seed(key)), agent_model=cfg["agent"]["model"], **kw)
    return ep


def run_e5(cfg, mock, pilot):
    """E5, benign safety with the REAL agent, in two separate conditions:
      clean        : legitimate user tasks, no untrusted instructions anywhere. Tests the R1
                     invariant (no fabrication ever reaches a trusted flow) and benign utility.
      contaminated : the same legitimate tasks while untrusted data containing an injection is
                     present in the environment. Tests whether the gate misroutes the user's OWN
                     benign actions (false routing) once tainted data flows into their inputs."""
    from cars.live.attacker import AVENUES, template
    out = LIVE / ("e5_mock.jsonl" if mock else f"e5{_sfx(cfg)}.jsonl")
    _archive_stale(out, "condition")
    done = set()
    if out.exists():   # keep only episodes that really ran; drop any earlier error rows
        good = [l for l in out.open() if not json.loads(l).get("error")]
        out.write_text("".join(good))
        done = {json.loads(l)["key"] for l in good}
    version = cfg["agentdojo_version"]
    c = cfg["e5"]
    cc = c.get("contaminated", {})
    for su in c["suites"][: 1 if pilot else None]:
        s = suite(version, su)
        uts = sorted(s.user_tasks)[: 3 if pilot else None]
        it_id = sorted(s.injection_tasks)[0]
        it = s.get_injection_task_by_id(it_id)
        avs = list(AVENUES)[: (1 if pilot else cc.get("n_avenues", 2))]
        cuts = uts[: (2 if pilot else cc.get("max_user_tasks", 6))]
        for p in c["policies"]:
            policy = LIVE_PRESETS[p]
            jobs = [(f"e5|{su}|{p}|{ut}", "clean", ut, None) for ut in uts]
            if cc.get("enabled", True):
                jobs += [(f"e5c|{su}|{p}|{ut}|{av}", "contaminated", ut, av) for ut in cuts for av in avs]
            for key, cond, ut, av in jobs:
                if key in done:
                    continue
                text = template(av, it.GOAL, [f.name for f in s.tools], []) if av else ""
                try:
                    ep = _e5_episode(out, key, cfg, mock, version=version, suite_name=su, user_task_id=ut,
                                     injection_task_id=it_id if av else None, injection_text=text, avenue=av,
                                     policy=policy, defender=DefenderState(policy), session="benign",
                                     agent="claude" if not mock else "mock")
                except BudgetExceeded as e:
                    print(f"\n!! {e}")
                    return False
                ep.update(key=key, exp="e5", condition=cond)
                with out.open("a") as f:
                    f.write(json.dumps(ep, default=str) + "\n")
        print(f"[e5] {su} done" + ("" if mock else f" | total ${tracker().total():.2f}"), flush=True)
    return True


def e7_decision_diff(cfg, mock):
    """E7 decision audit: for every write tool of the E7 suite and every honeytool, what the
    knowledge layer outputs under the flat KB vs. the ontology (vs. the oracle reference), and
    whether that changes a CARS decision: the response form (success / pending) over a grid of
    estimated attacker verification rates, and the ATLAS label put on the alert. The report joins
    this with the paired E7 logs to show whether the changed decisions changed outcomes."""
    from cars.live.knowledge import atlas_label, is_write, verifiability
    from cars.live.policies import HONEYTOOL_POOL
    su = cfg["e7"]["suite"]
    tools = sorted(f.name for f in suite(cfg["agentdojo_version"], su).tools if is_write(su, f.name))
    grid = cfg["e7"].get("vhat_grid", [0.02, 0.05, 0.1, 0.2, 0.5])
    pol = LIVE_PRESETS["cars"]
    rows = []
    for name, ht in [(t, False) for t in tools] + [(h, True) for h in HONEYTOOL_POOL]:
        r = {"tool": name, "honeytool": ht}
        for kb in ("flat", "onto", "oracle"):
            v = 0.5 if ht else verifiability(su, name, kb)
            r[f"ver_{kb}"] = round(v, 3)
            r[f"atlas_{kb}"] = atlas_label(su, name, ht, kb)
            forms = []
            for vh in grid:
                d = DefenderState(pol)
                d.verif_seen, d.fab_shown = 1, 1      # verification has been seen
                d.v_hat = (lambda vv=vh: vv)          # evaluate the form rule at this v_hat
                forms.append(d.form(v))
            r[f"forms_{kb}"] = forms
        r["form_changed"] = r["forms_flat"] != r["forms_onto"]
        r["atlas_changed"] = r["atlas_flat"] != r["atlas_onto"]
        r["atlas_correct_flat"] = r["atlas_flat"] == r["atlas_oracle"]
        r["atlas_correct_onto"] = r["atlas_onto"] == r["atlas_oracle"]
        r["ver_abs_err_flat"] = round(abs(r["ver_flat"] - r["ver_oracle"]), 3)
        r["ver_abs_err_onto"] = round(abs(r["ver_onto"] - r["ver_oracle"]), 3)
        rows.append(r)
    out = LIVE / ("e7_decision_diff_mock.json" if mock else "e7_decision_diff.json")
    out.write_text(json.dumps({"suite": su, "vhat_grid": grid, "rows": rows}, indent=2))
    print(f"[e7] decision audit: {sum(r['form_changed'] for r in rows)} form changes, "
          f"{sum(r['atlas_changed'] for r in rows)} label changes over {len(rows)} tools", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["all", "e0", "e2", "e3", "e3v", "e4", "e5", "e7", "e7diff"])
    ap.add_argument("--mock", action="store_true", help="scripted agent + template attacker, no API calls")
    ap.add_argument("--pilot", action="store_true", help="tiny slice to calibrate cost")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    init_llm(cfg, a.mock)
    ok = True
    if a.exp in ("all", "e0") and ok:
        ok = run_calibration(cfg, a.mock, a.pilot)
    if a.exp in ("all", "e5") and ok:
        ok = run_e5(cfg, a.mock, a.pilot)
    if a.exp in ("all", "e3") and ok:
        ok = run_attack_campaign("e3", specs_e3(cfg, a.mock, a.pilot), cfg, a.mock)
    if a.exp in ("all", "e3v") and ok and cfg.get("e3v", {}).get("enabled", True):
        ok = run_attack_campaign("e3v", specs_e3v(cfg, a.mock, a.pilot), cfg, a.mock)
    if a.exp in ("all", "e2") and ok:
        ok = run_attack_campaign("e2", specs_e2(cfg, a.mock, a.pilot), cfg, a.mock)
    if (a.exp == "e4" or (a.exp == "all" and cfg["e4"].get("enabled"))) and ok:
        ok = run_attack_campaign("e4", specs_e4(cfg, a.mock, a.pilot), cfg, a.mock)
    if (a.exp == "e7" or (a.exp == "all" and cfg["e7"].get("enabled"))) and ok:
        ok = run_attack_campaign("e7", specs_e7(cfg, a.mock, a.pilot), cfg, a.mock)
    if a.exp in ("all", "e7", "e7diff") and ok:
        e7_decision_diff(cfg, a.mock)
    if not a.mock:
        print(f"\nTotal API spend so far: ${tracker().total():.2f}  (details: results/live/usage.json)")


if __name__ == "__main__":
    main()
