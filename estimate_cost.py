"""Estimate the Claude API cost of the live campaign defined in config.yaml.

    python -m experiments.estimate_cost

Before any real run it uses a token model built from the measured size of each AgentDojo
suite's tool schema. After a --pilot run it re-calibrates from results/live/usage.json and
the episode logs, so the estimate reflects your actual token use.
"""
from __future__ import annotations

import json
from pathlib import Path

from agentdojo.agent_pipeline.llms.anthropic_llm import _function_to_anthropic
from agentdojo.functions_runtime import FunctionsRuntime

from cars.live.llm import load_config
from cars.live.runner import suite

LIVE = Path(__file__).resolve().parents[1] / "results" / "live"
CHARS_PER_TOKEN = 3.6          # conservative for JSON schemas / English
SYSTEM_TOKENS = 150
TURN_HISTORY_TOKENS = 450      # tool call + tool result (+ injection) added per agent turn
OUTPUT_PER_CALL = 220
CALLS_PER_EPISODE = {"banking": 4, "workspace": 5, "slack": 5, "travel": 6}
ATTACKER_IN, ATTACKER_OUT = 1100, 350
HONEYTOOL_TOKENS = 160


def agent_tokens(cfg, suite_name: str, honeytools: bool, attacked: bool) -> tuple[float, float]:
    s = suite(cfg["agentdojo_version"], suite_name)
    schema = len(json.dumps([_function_to_anthropic(t) for t in FunctionsRuntime(s.tools).functions.values()]))
    schema_tok = schema / CHARS_PER_TOKEN + (HONEYTOOL_TOKENS if honeytools else 0)
    calls = CALLS_PER_EPISODE.get(suite_name, 5) + (1 if attacked else 0)
    hist = (calls - 1) / 2 * TURN_HISTORY_TOKENS
    return calls * (SYSTEM_TOKENS + schema_tok + 60 + hist), calls * OUTPUT_PER_CALL


def usd(cfg, model, tin, tout):
    p = next(v for k, v in cfg["pricing"].items() if model.startswith(k))
    return (tin * p["input"] + tout * p["output"]) / 1e6


def plan(cfg):
    """(experiment, suite, episodes that call the real Claude agent). The assume-breach
    experiments (E2, E3, E4, E7) use a scripted agent and template attackers: no API cost."""
    from cars.live.attacker import AVENUES
    v = cfg["agentdojo_version"]
    out = []
    e0 = cfg["e0"]
    for su in e0["suites"]:
        n_ut = len(suite(v, su).user_tasks)
        per_split = min(e0.get("max_user_tasks") or n_ut, (n_ut + 1) // 2)
        out.append(("E0 steering", su, 2 * per_split * len(AVENUES) * e0["reps"]))
    e5 = cfg["e5"]
    cc = e5.get("contaminated", {})
    for su in e5["suites"]:
        n_ut = len(suite(v, su).user_tasks)
        out.append(("E5 clean", su, n_ut * len(e5["policies"])))
        if cc.get("enabled", True):
            out.append(("E5 contaminated", su, min(cc.get("max_user_tasks", 6), n_ut) * cc.get("n_avenues", 2)
                        * len(e5["policies"])))
    return out


def free_plan(cfg):
    e3 = cfg["e3"]
    lv = len(e3["compliance_sweep"])
    sweep_al = len([a for a in e3["attacker_algos"] if a in e3.get("sweep_algos", e3["attacker_algos"])]) or 1
    n3 = len(e3["suites"]) * len(e3["policies"]) * e3["seeds"] * e3["rounds"] * (lv * sweep_al + len(e3["attacker_algos"]))
    e2 = cfg["e2"]
    n2 = len(e2["policies"]) * len(e2["configs"]) * e2["seeds"] * e2["total_rounds"] * len(e2.get("attacker_algos", [1]))
    e7 = cfg["e7"]
    n7 = len(e7["policies"]) * e7["seeds"] * e7["rounds"] if e7.get("enabled") else 0
    v = cfg.get("e3v", {})
    nv = (len(v["suites"]) * len(v["policies"]) * len(v["attacker_algos"]) * len(v["compliance"])
          * len(v["v_att_levels"]) * v["seeds"] * v["rounds"]) if v.get("enabled", True) and v else 0
    return n3 + n2 + n7 + nv


def measured(cfg):
    """Average cost per episode from real runs, if a pilot has been done."""
    u = LIVE / "usage.json"
    eps = sum(1 for n in ("e0", "e5", "e0_permissive", "e5_permissive") if (LIVE / f"{n}.jsonl").exists()
              for _ in (LIVE / f"{n}.jsonl").open())
    if not u.exists() or eps == 0:
        return None
    data = json.loads(u.read_text())
    agent = sum(v["by_role"].get("agent", {}).get("usd", 0) for v in data["by_model"].values())
    return {"agent_usd_per_episode": agent / eps, "episodes": eps, "total": data["total_usd"]}


def main():
    cfg = load_config()
    agent_model = cfg["agent"]["model"]
    rows, tot = [], 0.0
    for exp, su, n in plan(cfg):
        tin, tout = agent_tokens(cfg, su, exp != "E0 steering", exp != "E5 clean")
        c = n * usd(cfg, agent_model, tin, tout)
        rows.append((exp, su, n, n * tin / 1e6, n * tout / 1e6, c))
        tot += c
    print(f"Agent model: {agent_model}  (profile files are separate: run once per profile)\n")
    print(f"{'Exp':16s} {'Suite':10s} {'Episodes':>8s} {'Mtok in':>8s} {'Mtok out':>8s} {'USD':>7s}")
    for r in rows:
        print(f"{r[0]:16s} {r[1]:10s} {r[2]:8d} {r[3]:8.2f} {r[4]:8.2f} {r[5]:7.2f}")
    print(f"\nEstimated API cost per agent profile: ${tot:.2f}   (both profiles: ${2 * tot:.2f})")
    print(f"Suggested budget with 40% margin: ${1.4 * 2 * tot:.2f}")
    n = free_plan(cfg)
    print(f"\nAssume-breach episodes (E2/E3/E3v/E7, no API cost): {n:,} "
          f"(~{n * 0.12 / 3600:.1f} CPU-hours; runs in parallel on {max(1, (__import__('os').cpu_count() or 2) - 1)} cores)")
    m = measured(cfg)
    if m:
        print(f"\nCalibration from your runs: {m['episodes']} episodes so far cost ${m['total']:.2f}; "
              f"agent ${m['agent_usd_per_episode']:.4f}/episode on average.")


if __name__ == "__main__":
    main()
