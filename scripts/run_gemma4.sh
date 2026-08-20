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

# flash-attn이 설치돼 있으면 자동으로 사용 (없으면 transformers 기본값)
ATTN=""
if python -c "import flash_attn" 2>/dev/null; then
  echo "flash-attn 감지 → flash_attention_2 사용"
  ATTN=",attn_implementation=flash_attention_2"
fi

run_eval gemma4 \
  "pretrained=${CKPT},max_num_frames=32,system_prompt=${SYS}${ATTN}" \
  "$TASKS" "logs/${NAME}"
