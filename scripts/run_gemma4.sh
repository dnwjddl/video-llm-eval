#!/usr/bin/env bash
# Gemma 4 (E2B / E4B) 평가 — 이 레포의 gemma4 플러그인 필요 (setup.sh가 설치)
# 사용법: bash scripts/run_gemma4.sh [TASKS] [CHECKPOINT]
#   기본 체크포인트: google/gemma-4-E2B-it
#   E4B로 돌리려면: bash scripts/run_gemma4.sh "" google/gemma-4-E4B-it
set -e
source "$(dirname "$0")/common.sh"

TASKS="${1:-$ALL_TASKS}"
CKPT="${2:-google/gemma-4-E2B-it}"
NAME="$(basename "$CKPT")"

run_eval gemma4 \
  "pretrained=${CKPT},max_num_frames=32" \
  "$TASKS" "logs/${NAME}"
