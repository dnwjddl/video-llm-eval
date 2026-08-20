#!/usr/bin/env bash
# LLaVA-Video 평가
# 사용법: bash scripts/run_llava_video.sh [TASKS] [CHECKPOINT]
#   기본 체크포인트: lmms-lab/LLaVA-Video-7B-Qwen2
set -e
source "$(dirname "$0")/common.sh"

TASKS="${1:-$ALL_TASKS}"
CKPT="${2:-lmms-lab/LLaVA-Video-7B-Qwen2}"

# 공식 평가 세팅: 64프레임 + average spatial pooling
run_eval llava_vid \
  "pretrained=${CKPT},conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average,video_decode_backend=decord" \
  "$TASKS" "logs/llava_video"
