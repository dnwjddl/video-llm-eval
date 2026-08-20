#!/usr/bin/env bash
# 결과 점수 요약 — logs/ 아래 모든 *results.json에서 벤치마크 점수를 뽑아 출력
#
# 사용법:
#   bash scripts/score.sh              # 모든 모델/벤치마크 결과 요약
#   bash scripts/score.sh qwen2_vl     # 경로에 qwen2_vl이 들어간 결과만
#   bash scripts/score.sh E2B          # Gemma E2B 결과만
#
# MVBench는 서브태스크 20개의 평균(= 최종 점수)을 자동 계산해서 >> 줄로 보여줍니다.
cd "$(dirname "$0")/.."

python3 - "${1:-}" <<'EOF'
import glob
import json
import sys

filt = sys.argv[1] if len(sys.argv) > 1 else ""
files = sorted(glob.glob("logs/**/*results.json", recursive=True))
files = [f for f in files if filt.lower() in f.lower()]
if not files:
    print("결과 파일이 없습니다. logs/ 아래에 *results.json이 있는지, 필터가 맞는지 확인하세요.")
    sys.exit(1)

def first_numeric(metrics):
    for k, v in metrics.items():
        if "stderr" not in k and isinstance(v, (int, float)):
            return v
    return None

for f in files:
    try:
        results = json.load(open(f))["results"]
    except Exception as e:
        print(f"[skip] {f}: {e}")
        continue
    print(f"\n=== {f}")
    mvbench_vals = []
    for task, metrics in sorted(results.items()):
        val = first_numeric(metrics)
        if val is None:
            continue
        print(f"  {task:45s} {val:.4f}")
        if task.startswith("mvbench_"):
            mvbench_vals.append(val)
    if mvbench_vals:
        avg = sum(mvbench_vals) / len(mvbench_vals)
        print(f"  >> MVBench 최종 점수 ({len(mvbench_vals)} subtasks 평균): {avg:.4f}")
EOF
