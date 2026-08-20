#!/usr/bin/env python3
"""토큰 압축 ablation 결과 → 곡선 그래프.

token_analysis/results/<method>_keep<keep>/ 의 lmms-eval 결과를 모아
(유지 비율 → 정확도) 곡선을 temporal / spatial 태스크 그룹별로 그린다.

실행: python token_analysis/plot_ablation.py
"""

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# MVBench 서브태스크의 temporal/spatial 구분 (우리 정의 — README 참고)
TEMPORAL = {
    "action_sequence", "action_prediction", "action_localization", "action_count",
    "moving_count", "moving_direction", "moving_attribute", "object_shuffle",
    "character_order", "scene_transition", "episodic_reasoning", "state_change",
    "counterfactual_inference", "egocentric_navigation",
}

# 방법별 색 — 구글 팔레트 (plot_latency.py와 동일 계열, CVD 검증 완료 순서)
COLORS = {
    "none": "#5F6368", "pool_avg": "#4285F4", "pool_max": "#8AB4F8",
    "random": "#B06000", "pca_select": "#EA4335", "pca_recon": "#C5221F",
    "tome": "#188038", "kmeans": "#5BB974", "temporal_pool": "#F29900",
    "framediff": "#1967D2", "scribe_tf": "#202124",
    "shuffle": "#9AA0A6", "reverse": "#DADCE0",
}
INK, INK_MUTED, GRID = "#202124", "#5F6368", "#E8EAED"

BASE = os.path.dirname(os.path.abspath(__file__))


def collect():
    """{method: {keep: {"temporal": acc, "spatial": acc, "overall": acc}}}"""
    data = defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(BASE, "results", "*_keep*"))):
        m = re.match(r"(.+)_keep([\d.]+)$", os.path.basename(d))
        if not m:
            continue
        method, keep = m.group(1), float(m.group(2))
        files = sorted(glob.glob(os.path.join(d, "**", "*results.json"), recursive=True))
        if not files:
            continue
        r = json.load(open(files[-1]))["results"]
        groups = defaultdict(list)
        for task, metrics in r.items():
            if not task.startswith("mvbench_"):
                continue
            val = next((v for k, v in metrics.items() if "stderr" not in k and isinstance(v, (int, float))), None)
            if val is None:
                continue
            sub = task.replace("mvbench_", "")
            groups["temporal" if sub in TEMPORAL else "spatial"].append(val)
            groups["overall"].append(val)
        data[method][keep] = {g: sum(v) / len(v) for g, v in groups.items() if v}
    return data


def main():
    data = collect()
    if not data:
        print("results/ 에 결과가 없습니다. run_token_ablation.py 먼저 실행하세요.")
        return

    baseline = data.get("none", {}).get(1.0, {})
    panels = ["overall", "temporal", "spatial"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, panel in zip(axes, panels):
        for method, by_keep in sorted(data.items()):
            if method == "none":
                continue
            pts = sorted((k, v[panel]) for k, v in by_keep.items() if panel in v)
            if not pts:
                continue
            xs, ys = zip(*pts)
            color = COLORS.get(method, "#5F6368")
            ax.plot(xs, ys, marker="o", markersize=5, linewidth=2, color=color, label=method)
        if panel in baseline:
            ax.axhline(baseline[panel], color=INK_MUTED, linewidth=1.2, linestyle="--")
            ax.text(0.03, baseline[panel], "no compression", fontsize=7.5, color=INK_MUTED, va="bottom")
        ax.set_title(panel, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("token keep ratio", fontsize=9, color=INK_MUTED)
        ax.set_xscale("log")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK_MUTED)
    axes[0].set_ylabel("accuracy", fontsize=9, color=INK_MUTED)
    axes[-1].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Token compression vs accuracy (MVBench, LLaVA-OneVision-7B)", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = os.path.join(BASE, "results", "token_ablation.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved: {out}\n")

    # 콘솔 표
    for method, by_keep in sorted(data.items()):
        for keep, vals in sorted(by_keep.items()):
            row = "  ".join(f"{g}={vals[g]:.3f}" for g in panels if g in vals)
            print(f"{method:14s} keep={keep:<6}  {row}")


if __name__ == "__main__":
    main()
