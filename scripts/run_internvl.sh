#!/usr/bin/env bash
# InternVL 계열 평가 (videollm 환경) — 체크포인트 이름으로 세대별 래퍼 자동 선택
# 사용법: bash scripts/run_internvl.sh [TASKS] [CHECKPOINT]
#   기본 체크포인트: OpenGVLab/InternVL3-8B
# 예:
#   bash scripts/run_internvl.sh mvbench OpenGVLab/InternVL2-8B
#   bash scripts/run_internvl.sh mvbench OpenGVLab/InternVL2_5-8B
#   bash scripts/run_internvl.sh mvbench OpenGVLab/InternVL3-8B
#   bash scripts/run_internvl.sh mvbench OpenGVLab/InternVL3_5-8B
set -e
source "$(dirname "$0")/common.sh"

TASKS="${1:-$ALL_TASKS}"
CKPT="${2:-OpenGVLab/InternVL3-8B}"

case "$(basename "$CKPT")" in
  InternVL3_5*) MODEL=internvl3_5 ;;
  InternVL3*)   MODEL=internvl3 ;;
  InternVL2*)   MODEL=internvl2 ;;   # InternVL2와 InternVL2.5(InternVL2_5-*) 둘 다 이 래퍼
  *) echo "체크포인트 이름에서 InternVL 세대를 인식하지 못했습니다: $CKPT"; exit 1 ;;
esac

# modality=video: 비디오 벤치마크 필수 설정
# InternVL3/3.5는 flash-attn이 기본 활성인데, 미설치 환경을 위해 꺼둠
#   (flash-attn을 설치했다면 use_flash_attn=True로 바꾸면 더 빠릅니다)
ARGS="pretrained=${CKPT},modality=video,num_frame=32"
if [ "$MODEL" != "internvl2" ]; then
  ARGS="${ARGS},use_flash_attn=False"
fi

run_eval "$MODEL" "$ARGS" "$TASKS" "logs/$(basename "$CKPT")"
