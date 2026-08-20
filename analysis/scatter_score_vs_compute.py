#!/usr/bin/env python3
"""[분석 3] 문항당 모델 연산 시간 vs MVBench 점수 산점도.

GPU 불필요. latency_results/(breakdown)와 logs/(점수)를 결합.
실행: python analysis/scatter_score_vs_compute.py
출력: analysis/out/score_vs_compute.png
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "analysis", "out")
INK, INK_MUTED, GRID = "#202124", "#5F6368", "#E8EAED"

FAMILY_COLOR = {  # 계열별 구글 색 (검증된 조합)
    "llava": "#4285F4", "LLaVA": "#4285F4",
    "Qwen": "#EA4335",
    "gemma": "#F29900",
    "InternVL": "#188038",
}
STAGES_COMPUTE = ["video_preprocess", "vision_encoder", "projector", "text_tokenization",
                  "llm_prefill", "autoregressive_decode", "detokenization"]


def family_color(name):
    for k, c in FAMILY_COLOR.items():
        if k.lower() in name.lower():
            return c
    return INK_MUTED


def load_scores():
    scores = {}
    for f in sorted(glob.glob(os.path.join(BASE, "logs", "*", "**", "*results.json"), recursive=True)):
        model = os.path.relpath(f, os.path.join(BASE, "logs")).split(os.sep)[0]
        try:
            r = json.load(open(f))["results"]
        except Exception:
            continue
        subs = {}
        for task, m in r.items():
            if task.startswith("mvbench_"):
                v = next((x for k, x in m.items() if "stderr" not in k and isinstance(x, (int, float))), None)
                if v is not None:
                    subs[task] = v * 100 if v <= 1 else v
        if len(subs) >= 20:
            scores[model] = sum(subs.values()) / len(subs)  # 최신 완주본이 덮어씀
    return scores


def load_compute():
    compute = {}
    for f in sorted(glob.glob(os.path.join(BASE, "latency_results", "*_videomme.json"))):
        r = json.load(open(f))
        model = os.path.basename(r["pretrained"])
        per = r.get("per_duration", {})
        vals = [sum(d.get(s, 0.0) for s in STAGES_COMPUTE) for d in per.values()]
        if vals:
            compute[model] = sum(vals) / len(vals)
    return compute


# 초기 실행분의 일반 폴더명 → 체크포인트 이름 별칭 (로그 체계 변경 전 실행분 대응)
ALIAS = {
    "llava_onevision": "llava-onevision-qwen2-7b-ov",
    "llava_video": "LLaVA-Video-7B-Qwen2",
    "qwen2_vl": "Qwen2-VL-7B-Instruct",
    "qwen2_5_vl": "Qwen2.5-VL-7B-Instruct",
    "qwen3_vl": "Qwen3-VL-8B-Instruct",
}


def norm(name):  # logs 폴더명 ↔ latency pretrained basename 매칭
    return ALIAS.get(name, name).lower().replace("_", "-")


def main():
    scores, compute = load_scores(), load_compute()
    os.makedirs(OUT_DIR, exist_ok=True)
    comp_by_norm = {norm(k): (k, v) for k, v in compute.items()}
    pts = []
    for m, s in scores.items():
        hit = comp_by_norm.get(norm(m))
        if hit:
            pts.append((hit[0], hit[1], s))
        else:
            print(f"[skip] breakdown 없음: {m}")
    if not pts:
        print("결합 가능한 모델이 없습니다.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 6))
    fig.patch.set_facecolor("white")
    for name, x, y in pts:
        c = family_color(name)
        ax.scatter(x, y, s=90, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=8, color=INK)
    ax.set_xlabel("model compute per QA (s, mean over durations, excl. video I/O)", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("MVBench accuracy (%)", fontsize=9, color=INK_MUTED)
    ax.set_title("Accuracy vs compute — recipe beats FLOPs", fontsize=12, color=INK, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK_MUTED)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "score_vs_compute.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved: {out}")
    for name, x, y in sorted(pts, key=lambda p: -p[2]):
        print(f"{name:38s} compute={x:5.2f}s  acc={y:5.2f}  (점수/초={y / x:5.1f})")


if __name__ == "__main__":
    main()
