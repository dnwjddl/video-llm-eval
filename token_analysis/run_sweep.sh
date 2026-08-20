#!/usr/bin/env bash
# 전체 스윕: 방법 × 유지 비율. llava 환경에서 실행.
# 사용법: bash token_analysis/run_sweep.sh [TASKS]
#   TASKS 생략 시 대표 8개 서브태스크 (run_token_ablation.py의 기본값)
set -e
cd "$(dirname "$0")/.."

TASKS_ARG=()
[ -n "$1" ] && TASKS_ARG=(--tasks "$1")

METHODS="pool_avg pool_max random pca_select tome kmeans temporal_pool framediff"
KEEPS="0.5 0.25 0.125 0.05"

# 기준선: 압축 없음 (원래 pooling, 196토큰/프레임)
python token_analysis/run_token_ablation.py --method none --keep 1.0 "${TASKS_ARG[@]}"

for m in $METHODS; do
  for k in $KEEPS; do
    echo ""
    echo "########## $m keep=$k ##########"
    python token_analysis/run_token_ablation.py --method "$m" --keep "$k" "${TASKS_ARG[@]}" || \
      echo "!! 실패: $m keep=$k — 건너뜀"
  done
done

# rank ablation 축 (토큰 수 유지, 정보량만 축소)
for k in 0.5 0.25 0.1; do
  python token_analysis/run_token_ablation.py --method pca_recon --keep "$k" "${TASKS_ARG[@]}" || true
done

echo "스윕 완료 — python token_analysis/plot_ablation.py 로 그래프 생성"
