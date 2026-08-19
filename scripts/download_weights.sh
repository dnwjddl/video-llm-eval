#!/usr/bin/env bash
# 모델 weight 일괄 선다운로드 (선택 사항 — 첫 평가 실행 시 자동으로도 받아짐)
# 총 약 80~90GB. HF_HOME 환경변수로 캐시 위치 지정 가능.
set -e

MODELS=(
  lmms-lab/llava-onevision-qwen2-7b-ov
  lmms-lab/LLaVA-Video-7B-Qwen2
  Qwen/Qwen2-VL-7B-Instruct
  Qwen/Qwen2.5-VL-7B-Instruct
  google/gemma-4-E2B-it
  google/gemma-4-E4B-it
)

for m in "${MODELS[@]}"; do
  echo "== downloading $m =="
  hf download "$m" || huggingface-cli download "$m"
done
