"""Aggregate blind predictions into per-item flags and per-category tables.

An item is `blind_pass` for a model iff it is answered correctly under EVERY
rotation. With --rule any (default) an item is EXCLUDED if any model passes;
with --rule all, only if every model passes.

Outputs (in --out-dir):
  flags.parquet     item_id, benchmark, category, n_options, chance, per-model pass/plain, blind_pass, excluded
  summary.csv/.md   per benchmark x category: n, plain acc, pass rate, chance, binomial p-value, kept n
  kept_ids.txt / excluded_ids.txt
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def per_model(pred: pd.DataFrame) -> pd.DataFrame:
    g = pred.groupby("item_id")
    out = pd.DataFrame({
        "benchmark": g["benchmark"].first(),
        "category": g["category"].first(),
        "n_options": g["n_options"].first(),
        "n_rot": g["shift"].nunique(),
        "pass": g["correct"].min().astype(int),           # all rotations correct
        "plain": pred[pred["shift"] == 0].set_index("item_id")["correct"],
        "parsed": g["parsed"].mean(),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="prediction parquets (one per model)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rule", choices=["any", "all"], default="any")
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tables, names = [], []
    for p in args.preds:
        pred = pd.read_parquet(p)
        name = os.path.splitext(os.path.basename(p))[0]
        names.append(name)
        t = per_model(pred)
        t = t.rename(columns={"pass": f"pass__{name}", "plain": f"plain__{name}", "parsed": f"parsed__{name}"})
        tables.append(t)
    flags = tables[0][["benchmark", "category", "n_options", "n_rot"]].copy()
    for t in tables:
        flags = flags.join(t.drop(columns=["benchmark", "category", "n_options", "n_rot"]), how="outer")
    pass_cols = [c for c in flags.columns if c.startswith("pass__")]
    plain_cols = [c for c in flags.columns if c.startswith("plain__")]
    flags["blind_pass"] = flags[pass_cols].max(axis=1) if args.rule == "any" else flags[pass_cols].min(axis=1)
    flags["blind_pass"] = flags["blind_pass"].fillna(0).astype(int)
    flags["plain_mean"] = flags[plain_cols].mean(axis=1)
    flags["chance"] = 1.0 / flags["n_options"]
    flags["chance_pass"] = flags["chance"] ** flags["n_rot"]
    flags["excluded"] = flags["blind_pass"]
    flags.index.name = "item_id"
    flags.reset_index().to_parquet(os.path.join(args.out_dir, "flags.parquet"), index=False)

    rows = []
    for (b, c), g in flags.groupby(["benchmark", "category"]):
        n = len(g)
        chance = g["chance"].mean()
        k = int(round(g["plain_mean"].sum()))
        pval = binomtest(k, n, chance, alternative="greater").pvalue if n > 0 else np.nan
        rows.append({"benchmark": b, "category": c, "n": n, "chance": round(chance, 3),
                     "blind_plain_acc": round(g["plain_mean"].mean(), 3), "p_vs_chance": f"{pval:.1e}",
                     "text_prior": "YES" if pval < args.alpha else "no",
                     "blind_pass_rate": round(g["blind_pass"].mean(), 3), "expected_lucky_pass": round(g["chance_pass"].mean(), 4),
                     "excluded": int(g["excluded"].sum()), "kept": int(n - g["excluded"].sum())})
    for b, g in flags.groupby("benchmark"):
        n = len(g)
        rows.append({"benchmark": b, "category": "__ALL__", "n": n, "chance": round(g["chance"].mean(), 3),
                     "blind_plain_acc": round(g["plain_mean"].mean(), 3), "p_vs_chance": "",
                     "text_prior": "", "blind_pass_rate": round(g["blind_pass"].mean(), 3),
                     "expected_lucky_pass": round(g["chance_pass"].mean(), 4),
                     "excluded": int(g["excluded"].sum()), "kept": int(n - g["excluded"].sum())})
    summ = pd.DataFrame(rows).sort_values(["benchmark", "category"])
    summ.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(f"models: {', '.join(names)}  rule: {args.rule}  alpha: {args.alpha}\n\n")
        f.write(summ.to_markdown(index=False))
    with open(os.path.join(args.out_dir, "kept_ids.txt"), "w") as f:
        f.write("\n".join(flags.index[flags["excluded"] == 0]))
    with open(os.path.join(args.out_dir, "excluded_ids.txt"), "w") as f:
        f.write("\n".join(flags.index[flags["excluded"] == 1]))
    print(summ.to_string(index=False))
    print(f"\ntotal items {len(flags)}  excluded {int(flags['excluded'].sum())}  kept {int((flags['excluded'] == 0).sum())}")


if __name__ == "__main__":
    main()
