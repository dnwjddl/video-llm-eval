#!/usr/bin/env bash
# Stage-1 blind filter, end to end. Run inside the `videollm` conda env with HF_HOME set.
#   bash scripts/blind_filter.sh                       # HF-hosted benchmarks, Qwen2.5-7B-Instruct
#   MODELS="Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct" ENGINE=vllm bash scripts/blind_filter.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BENCH=${BENCH:-all}
MODELS=${MODELS:-Qwen/Qwen2.5-7B-Instruct}
ENGINE=${ENGINE:-hf}
ROT=${ROT:-4}
OUT=${OUT:-encoder_study}
mkdir -p "$OUT/items" "$OUT/preds"

python -m encoder_study.blind.build_items --benchmarks "$BENCH" --out "$OUT/items" ${STAR_JSON:+--star-json "$STAR_JSON"} ${CLEVRER_JSON:+--clevrer-json "$CLEVRER_JSON"} ${VIDEO_ROOT:+--video-root "$VIDEO_ROOT"}
for M in $MODELS; do
  TAG=$(basename "$M" | tr '[:upper:]' '[:lower:]')
  python -m encoder_study.blind.run_blind --items "$OUT/items/*.parquet" --model "$M" --engine "$ENGINE" --rotations "$ROT" --out "$OUT/preds/$TAG.parquet"
done
python -m encoder_study.blind.report --preds "$OUT"/preds/*.parquet --out-dir "$OUT/report" --rule any
python -m encoder_study.blind.viewer --items "$OUT/items/*.parquet" --flags "$OUT/report/flags.parquet" --preds "$OUT"/preds/*.parquet --out-dir "$OUT/viewer"
echo "done. summary: $OUT/report/summary.md   viewer: cd $OUT/viewer && python -m http.server 8000"
