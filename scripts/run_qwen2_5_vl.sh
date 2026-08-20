#!/usr/bin/env bash
# Qwen2.5-VL 평가
# 사용법: bash scripts/run_qwen2_5_vl.sh [TASKS] [CHECKPOINT]
#   기본 체크포인트: Qwen/Qwen2.5-VL-7B-Instruct
set -e
source "$(dirname "$0")/common.sh"

TASKS="${1:-$ALL_TASKS}"
CKPT="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"

# 빈 40GB GPU에서 동작 확인된 설정. OOM이 나면 먼저 nvidia-smi로 GPU를 다른
# 작업과 공유하고 있지 않은지 확인하고, 그래도 나면 max_pixels=602112로 낮추세요
run_eval qwen2_5_vl \
  "pretrained=${CKPT},max_pixels=1605632,max_num_frames=32" \
  "$TASKS" "logs/qwen2_5_vl"
