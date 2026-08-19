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

echo "[3/4] LLaVA-NeXT는 이 환경에 설치하지 않습니다"
# LLaVA-NeXT는 구버전 transformers가 필요해서 Gemma 4(최신 transformers 필요)와
# 한 환경에 공존할 수 없습니다. LLaVA 계열을 쓰려면 README 1번 섹션대로
# 별도 conda 환경(llava)을 만들어 transformers==4.40.0으로 고정하세요.

echo "[4/4] Gemma 4 플러그인 설치"
pip install -e "$(dirname "$0")/gemma4_plugin"

echo ""
echo "완료! 다음으로 HF 로그인을 해두세요 (gated 데이터셋/모델용):"
echo "  hf auth login"
