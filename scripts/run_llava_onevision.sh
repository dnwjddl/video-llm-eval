#!/usr/bin/env bash
# LLaVA-OneVision 평가
# 사용법: bash scripts/run_llava_onevision.sh [TASKS] [CHECKPOINT]
#   TASKS      쉼표 구분 task 목록 (기본: 전체 8개 벤치마크)
#   CHECKPOINT HF 체크포인트 (기본: lmms-lab/llava-onevision-qwen2-7b-ov)
set -e
source "$(dirname "$0")/common.sh"

TASKS="${1:-$ALL_TASKS}"
CKPT="${2:-lmms-lab/llava-onevision-qwen2-7b-ov}"

run_eval llava_onevision \
  "pretrained=${CKPT},conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32" \
  "$TASKS" "logs/llava_onevision"
