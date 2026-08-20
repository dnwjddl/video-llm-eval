#!/usr/bin/env bash
# 전체 스윕: 방법 × 유지 비율. llava 환경에서 실행.
# 사용법: bash token_analysis/run_sweep.sh [TASKS]
#   TASKS 생략 시 대표 8개 서브태스크 (run_token_ablation.py의 기본값)
set -e
cd "$(dirname "$0")/.."

TASKS_ARG=()
[ -n "$1" ] && TASKS_ARG=(--tasks "$1")

# METHODS/RECON 오버라이드로 GPU 분할 (run_sweep_videomme.sh와 동일 패턴)
#   GPU A: METHODS="pool_avg pool_max random pca_select" CUDA_VISIBLE_DEVICES=0 bash ...
#   GPU B: METHODS="tome kmeans temporal_pool framediff" RECON=0 CUDA_VISIBLE_DEVICES=1 bash ...
METHODS="${METHODS:-pool_avg pool_max random pca_select tome kmeans temporal_pool framediff}"
KEEPS="0.5 0.25 0.125 0.05"

done_already() {  # 해당 조합 결과 폴더에 results.json이 있으면 완료로 간주
  ls token_analysis/results/"$1"_keep"$2"/*/*results.json >/dev/null 2>&1 || \
  ls token_analysis/results/"$1"_keep"$2"/*results.json >/dev/null 2>&1
}

# 기준선: 압축 없음 (원래 pooling, 196토큰/프레임)
if done_already none 1.0; then
  echo "스킵 (완료): 기준선 none"
else
  python token_analysis/run_token_ablation.py --method none --keep 1.0 "${TASKS_ARG[@]}"
fi

for m in $METHODS; do
  for k in $KEEPS; do
    if done_already "$m" "$k"; then
      echo "스킵 (완료): $m keep=$k"
      continue
    fi
    echo ""
    echo "########## $m keep=$k ##########"
    python token_analysis/run_token_ablation.py --method "$m" --keep "$k" "${TASKS_ARG[@]}" || \
      echo "!! 실패: $m keep=$k — 건너뜀"
  done
done

# rank ablation 축 (토큰 수 유지, 정보량만 축소) — 병렬 시 한쪽만 (RECON=0으로 끔)
if [ "${RECON:-1}" = "1" ]; then
  for k in 0.5 0.25 0.1; do
    if done_already pca_recon "$k"; then
      echo "스킵 (완료): pca_recon keep=$k"
      continue
    fi
    python token_analysis/run_token_ablation.py --method pca_recon --keep "$k" "${TASKS_ARG[@]}" || true
  done
fi

echo "스윕 완료 — python token_analysis/plot_ablation.py 로 그래프 생성"
