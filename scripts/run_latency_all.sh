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

# 사전 체크: 비디오 폴더
N_MP4=$(find "$VIDEO_DIR" -name "*.mp4" 2>/dev/null | wc -l)
if [ "$N_MP4" -lt 1 ]; then
  echo "!! $VIDEO_DIR 에 mp4가 없습니다. 먼저 비디오를 풀어주세요:"
  echo "   python latency/extract_videos_subset.py --dataset videomme --out_dir $VIDEO_DIR"
  exit 1
fi
echo "video_dir OK: $VIDEO_DIR ($N_MP4 mp4)"
mkdir -p debug

# conda 없이 각 환경의 python 바이너리를 직접 사용 (비대화형 셸에서 conda가 없어도 동작)
env_python() {
  local env="$1"
  for base in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "$HOME/conda" "${CONDA_PREFIX%%/envs/*}"; do
    if [ -n "$base" ] && [ -x "$base/envs/$env/bin/python" ]; then
      echo "$base/envs/$env/bin/python"
      return 0
    fi
  done
  return 1
}

run_one() {
  local env="$1" family="$2" ckpt="$3"
  local log="debug/latency_$(basename "$ckpt")_output.txt"
  local json="latency_results/$(basename "$ckpt")_videomme.json"
  if [ -f "$json" ] && grep -q '"complete": true' "$json"; then
    echo ""
    echo "############ 스킵 (이미 완료): $ckpt"
    return
  fi
  echo ""
  echo "############ [$env] $family — $ckpt ############"
  local py
  py=$(env_python "$env") || { echo "!! conda 환경 '$env'의 python을 찾지 못했습니다 — 건너뜀"; return; }
  "$py" -u latency/profile_latency.py \
      --family "$family" --pretrained "$ckpt" \
      --dataset videomme --video_dir "$VIDEO_DIR" --n_per_duration 50 2>&1 | tee "$log"
  if [ "${PIPESTATUS[0]}" -eq 0 ]; then
    echo "############ 완료: $ckpt"
  else
    echo "############ !! 실패: $ckpt — 에러 마지막 부분:"
    tail -8 "$log" | sed 's/^/    /'
    echo "############ (전체 로그: $log — scripts/send_debug.sh로 공유 가능)"
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
# InternVL remote code는 최신 transformers와 충돌 ('all_tied_weights_keys' AttributeError)
# → 2/2.5/3은 llava 환경(4.40), 3.5는 Qwen3 LM이라 중간 버전 환경(internvl) 필요 (README 참고)
run_one llava    internvl2       OpenGVLab/InternVL2-8B
run_one llava    internvl2_5     OpenGVLab/InternVL2_5-8B
run_one llava    internvl3       OpenGVLab/InternVL3-8B
run_one internvl internvl3_5     OpenGVLab/InternVL3_5-8B

echo ""
echo "모두 끝. 차트 생성: python latency/plot_latency.py  (결과: latency_results/)"
