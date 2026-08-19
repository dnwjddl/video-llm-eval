#!/usr/bin/env bash
# Qwen2.5-VL 평가
# 사용법: bash scripts/run_qwen2_5_vl.sh [TASKS] [CHECKPOINT]
#   기본 체크포인트: Qwen/Qwen2.5-VL-7B-Instruct
set -e
source "$(dirname "$0")/common.sh"

TASKS="${1:-$ALL_TASKS}"
CKPT="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"

# OOM 시 max_pixels를 602112 이하로, max_num_frames를 16으로 낮추세요
run_eval qwen2_5_vl \
  "pretrained=${CKPT},max_pixels=1605632,max_num_frames=32" \
  "$TASKS" "logs/qwen2_5_vl"
