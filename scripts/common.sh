#!/usr/bin/env bash
# 공통 설정 — 각 run_*.sh 에서 source 해서 사용
# ALL_TASKS: 이 레포가 커버하는 8개 벤치마크 task
ALL_TASKS="videomme,videomme_w_subtitle,videomme_v2,videomme_v2_w_subtitle,mvbench,longvideobench_val_v,lvbench,mlvu_dev"

# GPU 수 (기본 1장). 여러 장이면 NUM_GPUS=4 bash scripts/run_xxx.sh ... 처럼 오버라이드
NUM_GPUS="${NUM_GPUS:-1}"

run_eval() {
  local model="$1" model_args="$2" tasks="$3" outdir="$4"
  mkdir -p "$outdir"
  accelerate launch --num_processes="$NUM_GPUS" -m lmms_eval \
    --model "$model" \
    --model_args "$model_args" \
    --tasks "$tasks" \
    --batch_size 1 \
    --log_samples \
    --output_path "$outdir"
}
