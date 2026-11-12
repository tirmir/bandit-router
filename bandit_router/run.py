"""Run all policies on the synthetic stream over several seeds; write plot and table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import env
from .policies import Policy, make_policies


def simulate(policy: Policy, stream: env.Stream) -> dict:
    exp = stream.expected_reward()
    best = exp.max(axis=1)
    T = len(stream)
    regret = np.zeros(T)
    quality = np.zeros(T)
    cost = np.zeros(T)
    reward = np.zeros(T)
    arms = np.zeros(T, dtype=int)
    for t in range(T):
        x = stream.X[t]
        a = policy.select(x)
        q, r = stream.outcome(t, a)
        policy.update(x, a, r)
        regret[t] = best[t] - exp[t, a]
        quality[t], cost[t], reward[t], arms[t] = q, env.COST[a], r, a
    return {"regret": np.cumsum(regret), "quality": quality.mean(), "cost": cost.mean(),
            "reward": reward.mean(), "large_frac": float((arms == 1).mean())}


def run_all(seeds: int, steps: int, lam: float) -> dict:
    out: dict[str, list[dict]] = {}
    for seed in range(seeds):
        stream = env.make_stream(steps, seed, lam)
        for pol in make_policies(len(env.ARMS), env.D, seed):
            out.setdefault(pol.name, []).append(simulate(pol, stream))
    return out


def table(results: dict, steps: int) -> str:
    rows = ["| policy | quality | cost / request | reward / request | final cum. regret | share sent to large |",
            "|---|---|---|---|---|---|"]
    for name, runs in results.items():
        m = lambda k: np.mean([r[k] for r in runs])
        fr = [r["regret"][-1] for r in runs]
        rows.append(f"| {name} | {m('quality'):.3f} | {m('cost'):.3f} | {m('reward'):.3f} | "
                    f"{np.mean(fr):.1f} +/- {np.std(fr):.1f} | {m('large_frac'):.2f} |")
    return "\n".join(rows)


def plot(results: dict, path: str, steps: int, seeds: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, runs in results.items():
        curves = np.stack([r["regret"] for r in runs])
        mean, sd = curves.mean(0), curves.std(0)
        x = np.arange(1, steps + 1)
        ax.plot(x, mean, label=name, lw=2 if name in ("linucb", "thompson") else 1.2)
        ax.fill_between(x, mean - sd, mean + sd, alpha=0.12)
    ax.set_xlabel("requests routed")
    ax.set_ylabel("cumulative regret (expected reward)")
    ax.set_title(f"Synthetic router simulation, mean +/- sd over {seeds} seeds")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def main(argv=None):
    p = argparse.ArgumentParser(description="Contextual bandit router simulation (synthetic data)")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lam", type=float, default=0.3, help="cost penalty in reward = quality - lam * cost")
    p.add_argument("--out-dir", default="assets")
    a = p.parse_args(argv)

    results = run_all(a.seeds, a.steps, a.lam)
    tbl = table(results, a.steps)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text(tbl + "\n")
    summary = {n: {"quality": float(np.mean([r["quality"] for r in rs])),
                   "cost": float(np.mean([r["cost"] for r in rs])),
                   "reward": float(np.mean([r["reward"] for r in rs])),
                   "final_regret_mean": float(np.mean([r["regret"][-1] for r in rs])),
                   "final_regret_std": float(np.std([r["regret"][-1] for r in rs])),
                   "large_frac": float(np.mean([r["large_frac"] for r in rs]))}
               for n, rs in results.items()}
    (out / "results.json").write_text(json.dumps(
        {"seeds": a.seeds, "steps": a.steps, "lam": a.lam, "policies": summary}, indent=2))
    plot(results, str(out / "regret.png"), a.steps, a.seeds)
    print(f"synthetic data, {a.seeds} seeds, {a.steps} steps, lam={a.lam}")
    print(tbl)
    print(f"wrote {out}/regret.png, results.md, results.json")


if __name__ == "__main__":
    main()
