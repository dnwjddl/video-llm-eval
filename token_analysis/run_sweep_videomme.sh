#!/usr/bin/env bash
# Video-MME 150샘플(길이별 50개)로 방법 × 유지 비율 스윕. llava 환경에서 실행.
# 사용법: bash token_analysis/run_sweep_videomme.sh [VIDEO_DIR]
set -e
cd "$(dirname "$0")/.."

VIDEO_DIR="${1:-$HOME/videomme_videos}"

# METHODS 환경변수로 방법 목록 오버라이드 가능 → GPU 여러 장에 절반씩 분배
#   예) GPU A: METHODS="pool_avg pool_max random pca_select" CUDA_VISIBLE_DEVICES=0 bash ...
#       GPU B: METHODS="tome kmeans temporal_pool framediff" RECON=0 CUDA_VISIBLE_DEVICES=1 bash ...
METHODS="${METHODS:-pool_avg pool_max random pca_select tome kmeans temporal_pool framediff}"
KEEPS="0.5 0.25 0.125 0.05"

if grep -q '"complete": true' "token_analysis/results_videomme/none_keep1.0.json" 2>/dev/null; then
  echo "스킵 (완료): 기준선 none"
else
  python token_analysis/run_videomme_ablation.py --method none --keep 1.0 --video_dir "$VIDEO_DIR"
fi

for m in $METHODS; do
  for k in $KEEPS; do
    # 이미 완료된 조합은 스킵
    if grep -q '"complete": true' "token_analysis/results_videomme/${m}_keep${k}.json" 2>/dev/null; then
      echo "스킵 (완료): $m keep=$k"
      continue
    fi
    echo ""
    echo "########## $m keep=$k ##########"
    python token_analysis/run_videomme_ablation.py --method "$m" --keep "$k" --video_dir "$VIDEO_DIR" || \
      echo "!! 실패: $m keep=$k — 건너뜀"
  done
done

# rank ablation은 한 GPU에서만 (병렬 시 다른 쪽은 RECON=0)
if [ "${RECON:-1}" = "1" ]; then
  for k in 0.5 0.25 0.1; do
    if grep -q '"complete": true' "token_analysis/results_videomme/pca_recon_keep${k}.json" 2>/dev/null; then
      echo "스킵 (완료): pca_recon keep=$k"
      continue
    fi
    python token_analysis/run_videomme_ablation.py --method pca_recon --keep "$k" --video_dir "$VIDEO_DIR" || true
  done
fi

echo "스윕 완료 — python token_analysis/plot_videomme_ablation.py 로 그래프 생성"
