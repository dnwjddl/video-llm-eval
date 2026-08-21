#!/usr/bin/env python3
"""토큰 스윕 진행 상황 — 어떤 (방법 × keep) 조합이 끝났고 뭐가 남았는지.

실행: python token_analysis/sweep_progress.py
      ✓ 완료 · ~ 진행중(폴더는 있으나 결과 없음) · · 미시작
남은 조합을 GPU별로 나누는 명령도 함께 출력한다.
"""

import glob
import os

BASE = os.path.dirname(os.path.abspath(__file__))
METHODS = ["pool_avg", "pool_max", "random", "pca_select", "tome", "kmeans",
           "temporal_pool", "framediff", "scribe_tf", "pca_recon"]
KEEPS = ["0.05", "0.125", "0.25", "0.5"]


def state(method, keep):
    d = os.path.join(BASE, "results", f"{method}_keep{keep}")
    if not os.path.isdir(d):
        return "·"
    if glob.glob(os.path.join(d, "**", "*results.json"), recursive=True):
        return "✓"
    return "~"


def main():
    print(f"{'방법':16s}" + "".join(f"{('keep ' + k):>12s}" for k in KEEPS))
    print("-" * (16 + 12 * len(KEEPS)))
    pending = []
    for m in METHODS:
        cells = []
        for k in KEEPS:
            s = state(m, k)
            cells.append(f"{s:>12s}")
            if s == "·":
                pending.append((m, k))
        row = "".join(cells)
        if row.strip().replace("·", "") == "" and m in ("scribe_tf", "pca_recon"):
            continue  # 아직 안 돌린 선택 항목은 목록에서만
        print(f"{m:16s}{row}")

    done = sum(1 for m in METHODS for k in KEEPS if state(m, k) == "✓")
    print(f"\n완료 {done}개 · 미시작 {len(pending)}개")
    if not pending:
        print("모두 완료 — python token_analysis/plot_ablation.py 로 곡선 생성")
        return

    print("\n[남은 조합]")
    by_method = {}
    for m, k in pending:
        by_method.setdefault(m, []).append(k)
    for m, ks in by_method.items():
        print(f"  {m:16s} keep {' '.join(ks)}")

    ms = sorted(by_method)
    half = (len(ms) + 1) // 2
    print("\n[GPU 2장으로 나눠 돌리기]")
    for i, part in enumerate([ms[:half], ms[half:]]):
        if part:
            print(f'  GPU{i}: METHODS="{" ".join(part)}" CUDA_VISIBLE_DEVICES=<번호> bash token_analysis/run_sweep.sh mvbench')
    print("\n  (완료된 조합은 자동 스킵 · 한 방법을 두 GPU에서 동시에 돌리지 말 것)")


if __name__ == "__main__":
    main()
