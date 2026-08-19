#!/usr/bin/env bash
# 전체 모델 x 전체 벤치마크 일괄 평가 (매우 오래 걸립니다 — tmux/nohup 권장)
# 사용법: bash scripts/run_all.sh [TASKS]
set -e
cd "$(dirname "$0")/.."

TASKS="${1:-}"

bash scripts/run_llava_onevision.sh "$TASKS"
bash scripts/run_llava_video.sh     "$TASKS"
bash scripts/run_qwen2_vl.sh        "$TASKS"
bash scripts/run_qwen2_5_vl.sh      "$TASKS"
bash scripts/run_gemma4.sh          "$TASKS" google/gemma-4-E2B-it
bash scripts/run_gemma4.sh          "$TASKS" google/gemma-4-E4B-it

echo "모든 평가 완료 — 결과는 logs/ 아래에 있습니다."
