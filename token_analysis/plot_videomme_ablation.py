#!/usr/bin/env python3
"""Video-MME 150샘플 ablation 결과 → 비디오 길이별 곡선.

실행: python token_analysis/plot_videomme_ablation.py
"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_ablation import COLORS, GRID, INK, INK_MUTED  # 같은 팔레트 재사용

BASE = os.path.dirname(os.path.abspath(__file__))
DURATIONS = ["short", "medium", "long"]


def main():
    data = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(BASE, "results_videomme", "*_keep*.json"))):
        m = re.match(r"(.+)_keep([\d.]+)\.json$", os.path.basename(f))
        if not m:
            continue
        r = json.load(open(f))
        data[m.group(1)][float(m.group(2))] = r.get("per_duration", {})
    if not data:
        print("results_videomme/ 에 결과가 없습니다.")
        return

    baseline = data.get("none", {}).get(1.0, {})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, dur in zip(axes, DURATIONS):
        for method, by_keep in sorted(data.items()):
            if method == "none":
                continue
            pts = sorted((k, v[dur]["acc"]) for k, v in by_keep.items() if dur in v)
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=2,
                    color=COLORS.get(method, "#5F6368"), label=method)
        if dur in baseline:
            ax.axhline(baseline[dur]["acc"], color=INK_MUTED, linewidth=1.2, linestyle="--")
            ax.text(0.03, baseline[dur]["acc"], "no compression", fontsize=7.5, color=INK_MUTED, va="bottom")
        n = baseline.get(dur, {}).get("n", "?")
        ax.set_title(f"{dur} (n={n})", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("token keep ratio", fontsize=9, color=INK_MUTED)
        ax.set_xscale("log")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK_MUTED)
    axes[0].set_ylabel("accuracy", fontsize=9, color=INK_MUTED)
    axes[-1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Token compression vs accuracy by video length (Video-MME 150, LLaVA-OneVision-7B)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = os.path.join(BASE, "results_videomme", "videomme_ablation.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved: {out}\n")

    for method, by_keep in sorted(data.items()):
        for keep, vals in sorted(by_keep.items()):
            row = "  ".join(f"{d}={vals[d]['acc']:.3f}" for d in DURATIONS if d in vals)
            print(f"{method:14s} keep={keep:<6}  {row}")


if __name__ == "__main__":
    main()
