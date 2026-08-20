#!/usr/bin/env bash
# Qwen3-VL 평가 (videollm 환경)
# 사용법: bash scripts/run_qwen3_vl.sh [TASKS] [CHECKPOINT]
#   기본 체크포인트: Qwen/Qwen3-VL-8B-Instruct
set -e
source "$(dirname "$0")/common.sh"

TASKS="${1:-$ALL_TASKS}"
CKPT="${2:-Qwen/Qwen3-VL-8B-Instruct}"

run_eval qwen3_vl \
  "pretrained=${CKPT},max_pixels=1605632,max_num_frames=32" \
  "$TASKS" "logs/$(basename "$CKPT")"
