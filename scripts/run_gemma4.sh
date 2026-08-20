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

# system_prompt: Gemma는 객관식에서 장황한 설명/거절로 답해 letter 추출이 실패하므로
# (MVBench 27.5점 사건) letter-only 답변을 강제. 쉼표 금지 — model_args 파싱이 깨짐.
SYS="You are a helpful assistant. For multiple-choice questions you must answer with only the letter of the best option (e.g. A or B). Never explain and never refuse - always pick the single best option even if you are unsure."

# 주의: Gemma 4는 head dim이 256을 넘어 flash-attn 미지원
# ("Flash Attention forward only supports head dimension at most 256") → sdpa 고정
run_eval gemma4 \
  "pretrained=${CKPT},max_num_frames=32,system_prompt=${SYS},attn_implementation=sdpa" \
  "$TASKS" "logs/${NAME}"
