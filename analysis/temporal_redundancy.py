#!/usr/bin/env python3
"""[분석 4] 프레임 간 토큰 중복도 — temporal predictability 정량화.

SigLIP(so400m/384)으로 Video-MME 150개 비디오의 프레임 토큰을 뽑아,
인접 프레임 간 같은 위치 토큰의 코사인 유사도를 잰다.
"long 비디오 토큰의 X%는 이전 프레임과 사실상 동일" → 압축/prune 여지의 상한.

실행 (videollm 환경, GPU): python analysis/temporal_redundancy.py --video_dir ~/videomme_videos
출력: analysis/out/temporal_redundancy.{json,png}
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "latency"))
OUT_DIR = os.path.join(BASE, "analysis", "out")
INK, INK_MUTED, GRID = "#202124", "#5F6368", "#E8EAED"
BLUE, DBLUE, LBLUE = "#4285F4", "#174EA6", "#8AB4F8"

SIGLIP = "google/siglip-so400m-patch14-384"


def load_siglip(device="cuda"):
    from transformers import AutoImageProcessor, SiglipVisionModel

    model = SiglipVisionModel.from_pretrained(SIGLIP, torch_dtype=torch.bfloat16).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(SIGLIP)
    return model, proc


@torch.no_grad()
def video_tokens(model, proc, frames, batch=16):
    """frames (T,H,W,3) uint8 → (T, 729, D) fp32 (L2 정규화)."""
    outs = []
    for i in range(0, len(frames), batch):
        inp = proc(images=list(frames[i:i + batch]), return_tensors="pt").to(model.device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        h = model(**inp).last_hidden_state.float()
        outs.append(torch.nn.functional.normalize(h, dim=-1).cpu())
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default=os.path.expanduser("~/videomme_videos"))
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--n_per_duration", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thresholds", default="0.8,0.9,0.95")
    args = ap.parse_args()
    ths = [float(t) for t in args.thresholds.split(",")]

    from profile_latency import index_videos, load_samples, read_frames

    samples = load_samples("videomme", args.n_per_duration, args.seed)
    vindex = index_videos(args.video_dir)
    model, proc = load_siglip()
    os.makedirs(OUT_DIR, exist_ok=True)

    stats = {}
    for dur, rows in samples.items():
        sims_mean, frac = [], {t: [] for t in ths}
        seen = set()
        for i, row in enumerate(rows):
            vid = row["videoID"]
            if vid in seen:
                continue
            seen.add(vid)
            path = vindex.get(vid)
            if path is None:
                continue
            try:
                frames = read_frames(path, args.num_frames)
                tok = video_tokens(model, proc, frames)          # (T, 729, D)
                sim = (tok[1:] * tok[:-1]).sum(-1)                # (T-1, 729) 같은 위치 코사인
                sims_mean.append(sim.mean().item())
                for t in ths:
                    frac[t].append((sim > t).float().mean().item())
            except Exception as e:
                print(f"[err] {dur} #{i} {vid}: {type(e).__name__}: {e}", flush=True)
                continue
            if (i + 1) % 10 == 0:
                print(f"{dur} {i + 1}/{len(rows)}: mean_sim so far {np.mean(sims_mean):.3f}", flush=True)
        if sims_mean:
            stats[dur] = {"n_videos": len(sims_mean),
                          "mean_adjacent_cosine": round(float(np.mean(sims_mean)), 4),
                          **{f"frac_sim>{t}": round(float(np.mean(frac[t])), 4) for t in ths}}
            print(f"[{dur}] {stats[dur]}", flush=True)
        json.dump({"num_frames": args.num_frames, "encoder": SIGLIP, "per_duration": stats},
                  open(os.path.join(OUT_DIR, "temporal_redundancy.json"), "w"), indent=2)

    # 그림: duration별 임계값 초과 토큰 비율 (그룹 막대)
    durs = [d for d in ("short", "medium", "long") if d in stats]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("white")
    w = 0.8 / len(ths)
    colors = [LBLUE, BLUE, DBLUE]
    for j, t in enumerate(ths):
        xs = [i + j * w for i in range(len(durs))]
        ys = [stats[d][f"frac_sim>{t}"] * 100 for d in durs]
        ax.bar(xs, ys, width=w * 0.92, color=colors[j % 3], edgecolor="white", linewidth=1.2, label=f"cos > {t}")
        for x, y in zip(xs, ys):
            ax.text(x, y + 1, f"{y:.0f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks([i + w for i in range(len(durs))])
    ax.set_xticklabels([f"{d}\n(n={stats[d]['n_videos']})" for d in durs], fontsize=9, color=INK)
    ax.set_ylabel("tokens ~unchanged from previous frame (%)", fontsize=9, color=INK_MUTED)
    ax.set_title("Temporal redundancy of frame tokens (SigLIP, 32f uniform)", fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=INK_MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "temporal_redundancy.png"), dpi=200, bbox_inches="tight")
    print(f"saved: {os.path.join(OUT_DIR, 'temporal_redundancy.png')}")


if __name__ == "__main__":
    main()
