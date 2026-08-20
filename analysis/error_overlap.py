#!/usr/bin/env python3
"""[분석 1] 모델 간 오답 중첩 — MVBench 문항별 정오를 12개 모델에 걸쳐 겹친다.

GPU 불필요. 실행: python analysis/error_overlap.py

출력:
  - 모델별 정확도 (검증용)
  - 전 모델 정답/오답 문항 비율 ("진짜 어려운 문항")
  - oracle ensemble 상한 (하나라도 맞히면 정답)
  - 상보성 행렬: P(B 정답 | A 오답)
  - cascade 상한: 작은 모델이 맞히는 문항은 작은 모델로, 나머지만 큰 모델로 보냈을 때
    (oracle 라우팅 기준의 정확도와 큰 모델 호출 절감률)
  - analysis/out/error_overlap.json 저장
"""

import glob
import json
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "analysis", "out")

SMALL_CANDIDATES = ["gemma-4-E2B-it", "llava-onevision-qwen2-0.5b-ov", "gemma-4-E4B-it"]


def load_scores():
    """{model: {(subtask, doc_id): 0/1}} — 모델별 최신 샘플 로그 우선."""
    scores = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(BASE, "logs", "*", "**", "*samples*mvbench_*.jsonl"), recursive=True)):
        model = os.path.relpath(f, os.path.join(BASE, "logs")).split(os.sep)[0]
        sub = os.path.basename(f).split("mvbench_")[-1].split(".")[0]
        for line in open(f):
            try:
                row = json.loads(line)
                s = (row.get("mvbench_accuracy") or {}).get("score")
                did = row.get("doc_id")
                if s is not None and did is not None:
                    scores[model][(sub, did)] = int(s)  # 나중 파일이 이전 실행을 덮어씀
            except Exception:
                continue
    return scores


def main():
    scores = load_scores()
    if not scores:
        print("logs/ 에서 mvbench 샘플 로그를 찾지 못했습니다.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"모델 {len(scores)}개 로드:")
    for m, qs in sorted(scores.items(), key=lambda x: -sum(x[1].values()) / max(len(x[1]), 1)):
        print(f"  {m:38s} n={len(qs):5d}  acc={sum(qs.values()) / len(qs):.4f}")

    # 전 모델이 공통으로 푼 문항만으로 비교 (부분 실행 모델은 자동으로 교집합이 줄어듦)
    common = set.intersection(*[set(q) for q in scores.values()])
    print(f"\n공통 문항: {len(common)}개 (이후 분석은 이 집합 기준)")
    if len(common) < 100:
        print("[warn] 공통 문항이 적습니다 — 일부 모델이 부분 실행본만 있는지 확인하세요.")

    models = sorted(scores, key=lambda m: -sum(scores[m][q] for q in common) / max(len(common), 1))
    mat = {m: [scores[m][q] for q in sorted(common)] for m in models}
    n = len(common)

    all_right = sum(all(mat[m][i] for m in models) for i in range(n))
    all_wrong = sum(not any(mat[m][i] for m in models) for i in range(n))
    oracle = sum(any(mat[m][i] for m in models) for i in range(n))
    print(f"\n전 모델 정답: {all_right / n:.3f} | 전 모델 오답(진짜 어려움): {all_wrong / n:.3f} | oracle ensemble: {oracle / n:.3f}")

    # 상보성: P(B 정답 | A 오답)
    print("\n상보성 P(열 모델 정답 | 행 모델 오답) — 값이 높을수록 열 모델이 행 모델의 구멍을 메움")
    short = [m[:14] for m in models]
    print(" " * 16 + "".join(f"{s:>15s}" for s in short))
    for a in models:
        row = []
        a_wrong = [i for i in range(n) if not mat[a][i]]
        for b in models:
            row.append(sum(mat[b][i] for i in a_wrong) / max(len(a_wrong), 1))
        print(f"{a[:14]:>15s} " + "".join(f"{v:>15.3f}" for v in row))

    # cascade 상한 (oracle 라우팅)
    print("\ncascade 상한 (oracle 라우팅: 작은 모델이 맞히는 문항은 작은 모델이 답)")
    print(f"{'small':30s} {'big':30s} {'cascade acc':>12s} {'big단독':>8s} {'big호출율':>10s}")
    results = {"models": {m: sum(mat[m]) / n for m in models}, "all_wrong": all_wrong / n,
               "oracle_ensemble": oracle / n, "cascade": []}
    for small in SMALL_CANDIDATES:
        if small not in mat:
            continue
        for big in models:
            if big == small or sum(mat[big]) <= sum(mat[small]):
                continue
            acc = sum(1 for i in range(n) if mat[small][i] or mat[big][i]) / n
            big_calls = sum(1 for i in range(n) if not mat[small][i]) / n
            print(f"{small:30s} {big:30s} {acc:>12.3f} {sum(mat[big]) / n:>8.3f} {big_calls:>10.3f}")
            results["cascade"].append({"small": small, "big": big, "cascade_acc": round(acc, 4),
                                       "big_only_acc": round(sum(mat[big]) / n, 4),
                                       "big_call_fraction": round(big_calls, 4)})

    json.dump(results, open(os.path.join(OUT_DIR, "error_overlap.json"), "w"), indent=2)
    print(f"\nsaved: {os.path.join(OUT_DIR, 'error_overlap.json')}")


if __name__ == "__main__":
    main()
