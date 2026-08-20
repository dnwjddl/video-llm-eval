#!/usr/bin/env bash
# 전체 baseline latency breakdown 일괄 실행 (Qwen2.5-VL 제외 — 이미 측정했다는 전제)
# conda 환경 전환은 conda run으로 자동 처리. 한 모델이 실패해도 다음으로 계속 진행.
#
# 사용법: bash scripts/run_latency_all.sh [VIDEO_DIR]
#   VIDEO_DIR 기본값: ~/videomme_videos
#
# 주의: GPU 한 장에서 순차 실행입니다. 다른 평가가 돌고 있는 GPU라면
#       CUDA_VISIBLE_DEVICES=<빈 번호> bash scripts/run_latency_all.sh 로 지정하세요.
cd "$(dirname "$0")/.."

VIDEO_DIR="${1:-$HOME/videomme_videos}"

run_one() {
  local env="$1" family="$2" ckpt="$3"
  echo ""
  echo "############ [$env] $family — $ckpt ############"
  if conda run -n "$env" --no-capture-output python latency/profile_latency.py \
      --family "$family" --pretrained "$ckpt" \
      --dataset videomme --video_dir "$VIDEO_DIR" --n_per_duration 50; then
    echo "############ 완료: $ckpt"
  else
    echo "############ !! 실패: $ckpt — 건너뛰고 계속"
  fi
}

# llava 환경
run_one llava    llava_onevision lmms-lab/llava-onevision-qwen2-7b-ov
run_one llava    llava_onevision lmms-lab/llava-onevision-qwen2-0.5b-ov
run_one llava    llava_vid       lmms-lab/LLaVA-Video-7B-Qwen2

# videollm 환경
run_one videollm qwen2_vl        Qwen/Qwen2-VL-7B-Instruct
run_one videollm qwen3_vl        Qwen/Qwen3-VL-8B-Instruct
run_one videollm gemma4          google/gemma-4-E2B-it
run_one videollm gemma4          google/gemma-4-E4B-it
run_one videollm internvl2       OpenGVLab/InternVL2-8B
run_one videollm internvl2_5     OpenGVLab/InternVL2_5-8B
run_one videollm internvl3       OpenGVLab/InternVL3-8B
run_one videollm internvl3_5     OpenGVLab/InternVL3_5-8B

echo ""
echo "모두 끝. 차트 생성: python latency/plot_latency.py  (결과: latency_results/)"
