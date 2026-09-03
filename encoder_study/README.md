# encoder_study

Frozen-LLM encoder-information study. Stage 1 = **blind filter**: find questions a
text-only LLM answers correctly under every option rotation (no video), and remove
them from the evaluation set before any projector is trained.

## Stage 1: blind filter

```bash
conda activate videollm          # needs: datasets pandas pyarrow scipy tabulate tqdm transformers (vllm optional)
pip install -r encoder_study/requirements.txt
export HF_HOME=/data/hf_cache    # same HF_HOME lmms-eval uses (videos are found there)

# 0) look at raw columns before trusting a loader on a new machine
python -m encoder_study.blind.build_items --inspect tomato

# 1) unified item tables (one parquet per benchmark)
python -m encoder_study.blind.build_items --benchmarks all --out encoder_study/items
python -m encoder_study.blind.build_items --benchmarks star    --star-json    /data/STAR/STAR_val.json      --video-root /data/Charades_v1_480 --out encoder_study/items
python -m encoder_study.blind.build_items --benchmarks clevrer --clevrer-json /data/CLEVRER/validation.json --video-root /data/CLEVRER/video_validation --out encoder_study/items

# 2) text-only inference, 4 option rotations, per model (GPU). ~20K items x 4 = 80K short generations
python -m encoder_study.blind.run_blind --items "encoder_study/items/*.parquet" --model Qwen/Qwen2.5-7B-Instruct  --out encoder_study/preds/qwen2.5-7b.parquet
python -m encoder_study.blind.run_blind --items "encoder_study/items/*.parquet" --model Qwen/Qwen2.5-32B-Instruct --engine vllm --tensor-parallel 2 --out encoder_study/preds/qwen2.5-32b.parquet

# 3) flags + per-category table + kept/excluded id lists
python -m encoder_study.blind.report --preds encoder_study/preds/*.parquet --out-dir encoder_study/report --rule any

# 4) look at examples with their videos
python -m encoder_study.blind.viewer --items "encoder_study/items/*.parquet" --flags encoder_study/report/flags.parquet \
    --preds encoder_study/preds/*.parquet --out-dir encoder_study/viewer --n-excluded 30 --n-kept 10
cd encoder_study/viewer && python -m http.server 8000     # open http://<server>:8000  (ssh -L 8000:localhost:8000 <server>)
```

Decision rule: an item is **excluded** if the LLM answers correctly under all K=4 cyclic
option rotations (lucky-pass probability (1/n)^4 = 0.4% for 4 options). `--rule any`
excludes if any listed model passes. `summary.md` also reports, per category, the plain
blind accuracy, a one-sided binomial p-value against chance (is there a text prior at all?),
and the expected lucky-pass rate.

Everything downstream evaluates on `report/kept_ids.txt` only. The blind LLM is the same
frozen LLM used later, so this filter is identical for every encoder and every condition.

## Layout
```
encoder_study/blind/schema.py       Item schema, rotation, letter parsing
encoder_study/blind/loaders.py      per-benchmark loaders (mirror lmms-eval dataset ids / cache dirs)
encoder_study/blind/build_items.py  CLI: items parquet, --inspect
encoder_study/blind/run_blind.py    CLI: text-only LLM inference (hf | vllm), resume-able
encoder_study/blind/report.py       CLI: flags, summary, kept/excluded ids
encoder_study/blind/viewer.py       CLI: static HTML viewer with videos
```

## Benchmarks covered
| name | source | note |
|---|---|---|
| mvbench | OpenGVLab/MVBench (20 subtasks) | videos under $HF_HOME/mvbench_video |
| tvbench | FunAILab/TVBench (10 subtasks) | |
| tomato | lmms-eval/TOMATO | |
| vsibench | nyu-visionx/VSI-Bench | MC question types only (numeric types are not rotatable) |
| perceptiontest | lmms-eval/PerceptionTest_Val mc_question_val | category = area, subcategory = reasoning |
| motionbench | zai-org/MotionBench meta jsonl | set MOTIONBENCH_VIDEO_DIR for videos |
| mme_videoocr | DogNeverSleep/MME-VideoOCR_Dataset | MC rows only; pass --video-root |
| star | STAR_val.json (--star-json) | videos = Charades mp4 (--video-root) |
| clevrer | validation.json (--clevrer-json) | each MC choice -> yes/no item (chance 0.5) |

Vinoground is intentionally absent: its blind score is 50% by construction (caption pairs).
