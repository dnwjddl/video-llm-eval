#!/usr/bin/env bash
# Qwen2-VL 평가
# 사용법: bash scripts/run_qwen2_vl.sh [TASKS] [CHECKPOINT]
#   기본 체크포인트: Qwen/Qwen2-VL-7B-Instruct
set -e
source "$(dirname "$0")/common.sh"

TASKS="${1:-$ALL_TASKS}"
CKPT="${2:-Qwen/Qwen2-VL-7B-Instruct}"

# max_pixels=602112: 40GB GPU에서 장시간 비디오 OOM 방지용 (OOM 시 더 낮추세요)
run_eval qwen2_vl \
  "pretrained=${CKPT},max_pixels=602112,max_num_frames=32" \
  "$TASKS" "logs/qwen2_vl"
