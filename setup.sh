#!/usr/bin/env bash
# video-llm-eval 환경 세팅 스크립트
# 사용법: bash setup.sh
# 사전 준비: python>=3.10 가상환경 (conda 또는 venv) 활성화 상태에서 실행
set -e

echo "[1/4] lmms-eval 설치 (GitHub main — Video-MME v2, 플러그인 지원에 필요)"
pip install -U pip
pip install "git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git"

echo "[2/4] 공통 의존성 설치"
pip install -U torch torchvision accelerate transformers
pip install decord qwen-vl-utils "huggingface_hub[cli]"

echo "[3/4] LLaVA-NeXT 설치 (LLaVA-OneVision / LLaVA-Video 용)"
# 의존성 충돌이 생기면 이 단계만 별도 환경에서 수행해도 됩니다.
pip install "git+https://github.com/LLaVA-VL/LLaVA-NeXT.git" || \
  echo "!! LLaVA-NeXT 설치 실패 — LLaVA 계열 모델을 쓸 경우 별도 환경에서 재시도하세요."

echo "[4/4] Gemma 4 플러그인 설치"
pip install -e "$(dirname "$0")/gemma4_plugin"

echo ""
echo "완료! 다음으로 HF 로그인을 해두세요 (gated 데이터셋/모델용):"
echo "  hf auth login"
