#!/usr/bin/env python3
"""Gemma 4 E2B mvbench 로그 전수 분석 — 서브태스크별 답 패턴 (GPU 불필요, 몇 초).

실행: python debug/gemma4_debug4.py
출력은 debug/gemma4_debug4_output.txt 에도 저장됩니다.

각 서브태스크에 대해: 평균 점수, 모델이 낸 답(letter 추출 후)의 분포,
정답 letter 분포, 그리고 원문 답변 예시 2개를 출력합니다.
"""

import glob
import json
import os
import sys
from collections import Counter


class _Tee:
    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma4_debug4_output.txt")
sys.stdout = _Tee(OUT)
sys.stderr = sys.stdout

files = sorted(glob.glob("logs/gemma-4-E2B-it/**/*mvbench_*.jsonl", recursive=True))
assert files, "샘플 로그 없음: logs/gemma-4-E2B-it/**/*mvbench_*.jsonl"
print(f"{len(files)}개 서브태스크 로그 발견\n")

overall = []
for f in files:
    sub = os.path.basename(f).split("mvbench_")[-1].split(".")[0].replace("_test", "")
    rows = [json.loads(l) for l in open(f)]
    metrics = [r.get("mvbench_accuracy") or {} for r in rows]
    scores = [m.get("score") for m in metrics if m.get("score") is not None]
    preds = Counter(str(m.get("pred_answer", ""))[:40] for m in metrics)
    gts = Counter(str(m.get("gt_answer", "")) for m in metrics)
    mean = sum(scores) / len(scores) if scores else float("nan")
    overall.append((sub, mean))
    print(f"== {sub}: n={len(rows)}, 평균={mean:.3f}")
    print(f"   정답 letter 분포: {dict(gts.most_common(6))}")
    print(f"   모델 답 상위 5개: {preds.most_common(5)}")
    raws = [str((r.get('filtered_resps') or r.get('resps') or [''])) for r in rows[:2]]
    for raw in raws:
        print(f"   원문 예: {raw[:120]}")
    print()

print("=" * 60)
for sub, mean in sorted(overall, key=lambda x: x[1]):
    print(f"{sub:35s} {mean:.3f}")
print("\n완료 — bash scripts/send_debug.sh 로 공유하거나 패턴을 읽어주세요.")
