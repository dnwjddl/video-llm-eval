#!/usr/bin/env python3
"""[분석 2] 모델 × MVBench 서브태스크 히트맵.

GPU 불필요. 실행: python analysis/heatmap_subtasks.py
출력: analysis/out/subtask_heatmap.png + CSV + 콘솔 표
"""

import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "analysis", "out")

# temporal 서브태스크 (token_analysis/plot_ablation.py와 동일 정의)
TEMPORAL = {
    "action_sequence", "action_prediction", "action_localization", "action_count",
    "moving_count", "moving_direction", "moving_attribute", "object_shuffle",
    "character_order", "scene_transition", "episodic_reasoning", "state_change",
    "counterfactual_inference", "egocentric_navigation",
}
INK, INK_MUTED = "#202124", "#5F6368"


def load():
    """{model: {subtask: acc}} — 모델 폴더별로 최신 results.json이 서브태스크를 덮어씀."""
    data = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(BASE, "logs", "*", "**", "*results.json"), recursive=True)):
        model = os.path.relpath(f, os.path.join(BASE, "logs")).split(os.sep)[0]
        try:
            r = json.load(open(f))["results"]
        except Exception:
            continue
        for task, m in r.items():
            if task.startswith("mvbench_"):
                v = next((x for k, x in m.items() if "stderr" not in k and isinstance(x, (int, float))), None)
                if v is not None:
                    data[model][task.replace("mvbench_", "")] = v * 100 if v <= 1 else v
    return {m: s for m, s in data.items() if len(s) >= 20}  # 전체 20개 완주 모델만


def main():
    data = load()
    if not data:
        print("20개 서브태스크를 완주한 모델이 없습니다.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    models = sorted(data, key=lambda m: -sum(data[m].values()) / len(data[m]))
    # 열: temporal 그룹 먼저, 각 그룹 안에서는 전 모델 평균 낮은(어려운) 순
    subs = sorted({s for d in data.values() for s in d},
                  key=lambda s: (s not in TEMPORAL, sum(d.get(s, 0) for d in data.values())))
    grid = [[data[m].get(s, float("nan")) for s in subs] for m in models]

    # 단일 색상(구글 블루) sequential ramp — 밝음(낮음) → 진함(높음)
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#FFFFFF", "#8AB4F8", "#1967D2", "#174EA6"])
    n_t = sum(1 for s in subs if s in TEMPORAL)

    fig_w = 1.0 + 0.62 * len(subs)
    fig_h = 1.8 + 0.42 * len(models)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    im = ax.imshow(grid, cmap=cmap, vmin=20, vmax=90, aspect="auto")
    for i in range(len(models)):
        for j in range(len(subs)):
            v = grid[i][j]
            if v == v:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 62 else INK)
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels(subs, rotation=45, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([f"{m}  ({sum(data[m].values()) / len(data[m]):.1f})" for m in models], fontsize=8.5, color=INK)
    ax.axvline(n_t - 0.5, color=INK, linewidth=1.6)
    ax.text((n_t - 1) / 2, -0.9, "temporal", ha="center", fontsize=9, color=INK)
    ax.text(n_t + (len(subs) - n_t - 1) / 2, -0.9, "spatial/static", ha="center", fontsize=9, color=INK)
    ax.set_title("MVBench subtask accuracy by model", fontsize=12, color=INK, loc="left", pad=26)
    fig.colorbar(im, ax=ax, shrink=0.7, label="accuracy (%)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "subtask_heatmap.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved: {out}")

    with open(os.path.join(OUT_DIR, "subtask_heatmap.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + subs + ["temporal_mean", "spatial_mean"])
        for m in models:
            t = [data[m][s] for s in subs if s in TEMPORAL and s in data[m]]
            sp = [data[m][s] for s in subs if s not in TEMPORAL and s in data[m]]
            w.writerow([m] + [round(data[m].get(s, float("nan")), 2) for s in subs]
                       + [round(sum(t) / len(t), 2), round(sum(sp) / len(sp), 2)])
    print(f"saved: {os.path.join(OUT_DIR, 'subtask_heatmap.csv')}")

    print(f"\n{'model':38s} {'temporal':>9s} {'spatial':>9s} {'격차':>7s}")
    for m in models:
        t = [data[m][s] for s in subs if s in TEMPORAL and s in data[m]]
        sp = [data[m][s] for s in subs if s not in TEMPORAL and s in data[m]]
        tm, sm = sum(t) / len(t), sum(sp) / len(sp)
        print(f"{m:38s} {tm:>9.1f} {sm:>9.1f} {tm - sm:>+7.1f}")


if __name__ == "__main__":
    main()
