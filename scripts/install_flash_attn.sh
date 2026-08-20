#!/usr/bin/env bash
# flash-attn 설치 — 현재 환경의 torch/python 조합에 맞는 공식 릴리스 wheel을 받아 설치.
# (PyPI 기본 설치본이 GLIBC_2.32를 요구해 구형 OS에서 ImportError가 나는 문제의 해법)
#
# 사용법: (videollm 등 대상 conda 환경 활성화 후) bash scripts/install_flash_attn.sh
set -e

FA_VER="2.7.4.post1"

PYTAG=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
TORCH_MM=$(python -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'{v[0]}.{v[1]}')")
ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")

WHEEL="flash_attn-${FA_VER}+cu12torch${TORCH_MM}cxx11abi${ABI}-${PYTAG}-${PYTAG}-linux_x86_64.whl"
URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/${WHEEL}"

echo "python: ${PYTAG}, torch: ${TORCH_MM}, cxx11abi: ${ABI}"
echo "다운로드: ${URL}"

pip uninstall -y flash-attn 2>/dev/null || true

cd /tmp
wget -q --show-progress "$URL"
pip install "./${WHEEL}"
rm -f "./${WHEEL}"

echo ""
python -c "import flash_attn; print('flash-attn OK:', flash_attn.__version__)" && \
  echo "설치 성공 — 이후 새로 시작하는 실행부터 자동 적용됩니다." || \
  echo "!! import 실패 — 에러 메시지를 Claude에게 알려주세요."
