#!/usr/bin/env python3
"""[분석 5] 살아남는 토큰의 시공간 분포 — pca_select가 keep 0.05에서 남기는 토큰이
화면 어디에 몰리는지(27×27 히트맵), 그리고 움직임(프레임 간 변화) 큰 토큰과
얼마나 겹치는지(motion alignment)를 잰다.

실행 (videollm 환경, GPU): python analysis/token_selection_map.py --video_dir ~/videomme_videos
출력: analysis/out/token_selection_map.{json,png}
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
from matplotlib.colors import LinearSegmentedColormap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "latency"))
sys.path.insert(0, os.path.join(BASE, "token_analysis"))
sys.path.insert(0, os.path.join(BASE, "analysis"))
OUT_DIR = os.path.join(BASE, "analysis", "out")
INK, INK_MUTED = "#202124", "#5F6368"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default=os.path.expanduser("~/videomme_videos"))
    ap.add_argument("--keep", type=float, default=0.05)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--n_videos", type=int, default=30, help="duration당 사용할 비디오 수 (기본 각 10)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from compress import pca_select
    from profile_latency import index_videos, load_samples, read_frames
    from temporal_redundancy import load_siglip, video_tokens

    samples = load_samples("videomme", 50, args.seed)
    vindex = index_videos(args.video_dir)
    model, proc = load_siglip()
    os.makedirs(OUT_DIR, exist_ok=True)

    per_dur = max(1, args.n_videos // 3)
    grid = 27
    heat = np.zeros((grid, grid))
    motion_align, motion_align_rand = [], []
    used = 0

    for dur in ("short", "medium", "long"):
        cnt = 0
        seen = set()
        for row in samples.get(dur, []):
            if cnt >= per_dur:
                break
            vid = row["videoID"]
            if vid in seen or vindex.get(vid) is None:
                continue
            seen.add(vid)
            try:
                frames = read_frames(vindex[vid], args.num_frames)
                tok = video_tokens(model, proc, frames)  # (T, 729, D) 정규화됨
            except Exception as e:
                print(f"[err] {vid}: {type(e).__name__}: {e}", flush=True)
                continue
            T, N, D = tok.shape
            k = max(1, int(N * args.keep))

            # pca_select와 동일한 선택 인덱스 재현 (compress.pca_select는 인덱스를 안 돌려주므로 재계산)
            sel_mask = torch.zeros(T, N, dtype=torch.bool)
            for f in range(T):
                feat = tok[f]
                c = feat - feat.mean(0, keepdim=True)
                _, _, v = torch.pca_lowrank(c, q=max(2, min(k, min(c.shape) - 1, 32)))
                lever = (c @ v).pow(2).sum(-1)
                sel_mask[f, lever.topk(k).indices] = True
            heat += sel_mask.float().mean(0).view(grid, grid).numpy()

            # motion alignment: 프레임 간 변화량 상위 k 토큰과의 겹침 (기대값 = keep)
            diff = 1 - (tok[1:] * tok[:-1]).sum(-1)          # (T-1, N) 변화량
            for f in range(1, T):
                top_motion = diff[f - 1].topk(k).indices
                inter = sel_mask[f][top_motion].float().mean().item()
                motion_align.append(inter)
                motion_align_rand.append(args.keep)           # 무작위 선택의 기대 겹침
            cnt += 1
            used += 1
            print(f"{dur}: {cnt}/{per_dur} ({vid})", flush=True)

    heat /= max(used, 1)
    align = float(np.mean(motion_align)) if motion_align else 0.0
    report = {"keep": args.keep, "n_videos": used,
              "motion_alignment": round(align, 4),
              "random_expectation": args.keep,
              "alignment_ratio_vs_random": round(align / args.keep, 2)}
    json.dump(report, open(os.path.join(OUT_DIR, "token_selection_map.json"), "w"), indent=2)
    print(f"\nmotion alignment: {align:.3f} (무작위 기대 {args.keep}) → {align / args.keep:.1f}배")

    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#FFFFFF", "#8AB4F8", "#1967D2", "#174EA6"])
    fig, ax = plt.subplots(figsize=(5.6, 5))
    fig.patch.set_facecolor("white")
    im = ax.imshow(heat * 100, cmap=cmap)
    ax.set_title(f"pca_select keep={args.keep}: spatial distribution of selected tokens\n"
                 f"(n={used} videos · motion alignment {align / args.keep:.1f}× random)",
                 fontsize=10, color=INK, loc="left")
    ax.set_xticks([]), ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="selection probability (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "token_selection_map.png"), dpi=200, bbox_inches="tight")
    print(f"saved: {os.path.join(OUT_DIR, 'token_selection_map.png')}")


if __name__ == "__main__":
    main()
