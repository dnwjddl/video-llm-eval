#!/usr/bin/env python3
"""latency_results/*.json → 단계별 latency breakdown 차트 (PNG).

profile_latency.py가 저장한 JSON들을 전부 읽어 short/medium/long 패널에
모델별 스택 가로 막대로 그립니다. 색상은 Google 브랜드 팔레트 기반
(색약 안전성 검증 완료된 순서).

사용:
  python latency/plot_latency.py                        # latency_results/ 전체
  python latency/plot_latency.py --out my_plot.png
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 스택(파이프라인) 순서 고정 — 인접 색 조합은 CVD 검증을 통과한 순서
STAGES = [
    ("video_io_frame_extract", "Video I/O + frame extract", "#0D652D"),
    ("video_preprocess", "Video preprocess", "#4285F4"),
    ("vision_encoder", "Vision encoder", "#EA4335"),
    ("projector", "Projector", "#174EA6"),
    ("text_tokenization", "Text tokenization", "#F29900"),
    ("llm_prefill", "LLM prefill", "#A50E0E"),
    ("autoregressive_decode", "Autoregressive decode", "#34A853"),
    ("detokenization", "Detokenization", "#B06000"),
]
DURATIONS = ["short", "medium", "long"]
INK = "#202124"        # 본문 텍스트 (Google grey 900)
INK_MUTED = "#5F6368"  # 보조 텍스트 (Google grey 700)
GRID = "#E8EAED"       # 그리드 (Google grey 200)


def load_reports(results_dir):
    reports = []
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            r = json.load(open(f))
            if "per_duration" in r and r["per_duration"]:
                reports.append(r)
        except Exception as e:
            print(f"[skip] {f}: {e}")
    return reports


def model_label(r):
    return os.path.basename(r["pretrained"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="latency_results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    reports = load_reports(args.results_dir)
    if not reports:
        print(f"{args.results_dir}/ 에 결과 JSON이 없습니다. profile_latency.py를 먼저 실행하세요.")
        return

    durations = [d for d in DURATIONS if any(d in r["per_duration"] for r in reports)]
    # 전체 합(느린 순)으로 모델 정렬 — 색은 단계(entity)에 고정, 순서만 정렬
    def total_of(r):
        return sum(sum(r["per_duration"][d].get(k, 0.0) for k, _, _ in STAGES) for d in durations if d in r["per_duration"])

    reports.sort(key=total_of)
    labels = [model_label(r) for r in reports]

    fig_h = max(2.2, 0.62 * len(reports) + 1.6)
    fig, axes = plt.subplots(1, len(durations), figsize=(4.6 * len(durations), fig_h), sharey=True)
    if len(durations) == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, dur in zip(axes, durations):
        ax.set_facecolor("white")
        y = range(len(reports))
        left = [0.0] * len(reports)
        for key, _, color in STAGES:
            vals = [r["per_duration"].get(dur, {}).get(key, 0.0) for r in reports]
            ax.barh(y, vals, left=left, height=0.62, color=color,
                    edgecolor="white", linewidth=1.5)
            left = [l + v for l, v in zip(left, vals)]
        # 막대 끝 합계 직접 라벨 (텍스트는 잉크 색 — 시리즈 색 아님)
        xmax = max(left) if left else 1.0
        for yi, total in zip(y, left):
            ax.text(total + xmax * 0.015, yi, f"{total:.2f}s",
                    va="center", ha="left", fontsize=9, color=INK)
        n = next((r["per_duration"][dur].get("n") for r in reports if dur in r["per_duration"]), "?")
        ax.set_title(f"{dur}  (n={n}/model)", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("seconds per QA (mean)", fontsize=9, color=INK_MUTED)
        ax.set_xlim(0, xmax * 1.14)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=9, color=INK)
        ax.tick_params(axis="x", labelsize=8, colors=INK_MUTED)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    handles = [Patch(facecolor=c, edgecolor="white", label=lbl) for _, lbl, c in STAGES]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02), labelcolor=INK)
    loads = " · ".join(f"{model_label(r)}: {r.get('checkpoint_load_s', '?')}s" for r in reports)
    fig.suptitle("Video QA latency breakdown (Video-MME)", fontsize=13, color=INK, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.915, f"checkpoint load (once): {loads}", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 0.86))

    out = args.out or os.path.join(args.results_dir, "latency_breakdown.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved: {out}")

    # 수치 표 (콘솔) — 저채도 색 대비 보완용 정확값
    for dur in durations:
        print(f"\n[{dur}] seconds per QA (mean)")
        header = "model".ljust(34) + "".join(k.split("_")[0][:10].rjust(11) for k, _, _ in STAGES) + "total".rjust(11)
        print(header)
        for r in reports:
            d = r["per_duration"].get(dur, {})
            vals = [d.get(k, 0.0) for k, _, _ in STAGES]
            print(model_label(r).ljust(34) + "".join(f"{v:.3f}".rjust(11) for v in vals) + f"{sum(vals):.3f}".rjust(11))


if __name__ == "__main__":
    main()
