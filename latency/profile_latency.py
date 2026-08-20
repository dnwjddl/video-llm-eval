#!/usr/bin/env python3
"""Video QA latency breakdown profiler.

Video-MME / Video-MME v2에서 short/medium/long별로 N개씩 랜덤 샘플링해서
QA 1회당 단계별 latency를 측정합니다.

측정 단계:
  checkpoint_load        모델/프로세서 로딩 (모델당 1회)
  video_io_frame_extract 비디오 파일 열기 + 프레임 디코딩 (decord)
  video_preprocess       프레임 resize/normalize (processor)
  vision_encoder         비전 인코더 forward (hook 측정)
  projector              비전→LLM 프로젝터 forward (hook 측정)
  text_tokenization      텍스트 프롬프트 토크나이즈
  llm_prefill            첫 forward에서 vision/projector 중첩분을 뺀 시간
  autoregressive_decode  둘째 forward부터의 합
  detokenization         출력 토큰 → 텍스트

사용 예 (환경 주의: llava 계열은 llava env, 나머지는 videollm env):
  python latency/profile_latency.py --family qwen2_5_vl \
      --pretrained Qwen/Qwen2.5-VL-7B-Instruct \
      --dataset videomme --video_dir ~/videomme_videos --n_per_duration 50
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

STAGES = [
    "video_io_frame_extract",
    "video_preprocess",
    "vision_encoder",
    "projector",
    "text_tokenization",
    "llm_prefill",
    "autoregressive_decode",
    "detokenization",
]

HF_FAMILIES = ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "gemma4")
LLAVA_FAMILIES = ("llava_onevision", "llava_vid")

VISION_CANDIDATES = ["visual", "model.visual", "vision_tower", "model.vision_tower", "vision_model", "model.vision_model"]
PROJ_CANDIDATES = ["visual.merger", "model.visual.merger", "multi_modal_projector", "model.multi_modal_projector", "mm_projector", "model.mm_projector", "embed_vision", "model.embed_vision"]


def now():
    torch.cuda.synchronize()
    return time.perf_counter()


def find_module(model, candidates):
    for name in candidates:
        mod = model
        ok = True
        for part in name.split("."):
            if hasattr(mod, part):
                mod = getattr(mod, part)
            else:
                ok = False
                break
        if ok and isinstance(mod, torch.nn.Module):
            return name, mod
    return None, None


class HookTimer:
    """forward hook으로 특정 서브모듈의 누적 실행 시간을 잰다."""

    def __init__(self):
        self.total = defaultdict(float)
        self.handles = []

    def attach(self, name, module):
        def pre(mod, args):
            mod.__dict__["_lt_t0"] = now()

        def post(mod, args, out):
            self.total[name] += now() - mod.__dict__.get("_lt_t0", now())

        self.handles.append(module.register_forward_pre_hook(pre))
        self.handles.append(module.register_forward_hook(post))

    def reset(self):
        self.total = defaultdict(float)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class PrefillDecodeTimer:
    """모델 최상위 forward를 훅킹해 첫 호출(prefill)과 이후(decode)를 분리."""

    def __init__(self, model, hook_timer, nested_keys):
        self.ht = hook_timer
        self.nested_keys = nested_keys
        self.reset()

        def pre(mod, args, kwargs=None):
            self._t0 = now()
            self._nested0 = sum(self.ht.total[k] for k in self.nested_keys)

        def post(mod, args, out):
            dt = now() - self._t0
            nested = sum(self.ht.total[k] for k in self.nested_keys) - self._nested0
            if self.calls == 0:
                self.prefill = max(dt - nested, 0.0)
            else:
                self.decode += dt
            self.calls += 1

        self.h1 = model.register_forward_pre_hook(pre)
        self.h2 = model.register_forward_hook(post)

    def reset(self):
        self.calls = 0
        self.prefill = 0.0
        self.decode = 0.0

    def remove(self):
        self.h1.remove()
        self.h2.remove()


def read_frames(video_path, num_frames):
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    n = min(num_frames, total)
    idx = np.linspace(0, total - 1, n).astype(int)
    frames = vr.get_batch(idx).asnumpy()  # (T, H, W, 3) uint8
    return frames


# ---------------------------------------------------------------- HF models


def load_hf(family, pretrained):
    from transformers import AutoProcessor

    if family == "gemma4":
        try:
            from transformers import AutoModelForMultimodalLM as Cls
        except ImportError:
            from transformers import AutoModelForImageTextToText as Cls
    else:
        from transformers import AutoModelForImageTextToText as Cls
    t0 = time.perf_counter()
    model = Cls.from_pretrained(pretrained, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    processor = AutoProcessor.from_pretrained(pretrained)
    load_s = time.perf_counter() - t0
    return model, processor, load_s


def run_sample_hf(model, processor, frames, question, max_new_tokens, hook_timer, pd_timer, rec):
    from PIL import Image

    tokenizer = processor.tokenizer

    # 채팅 템플릿 텍스트 구성 (비디오 자리 + 질문)
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    t0 = time.perf_counter()
    tokenizer(text, return_tensors="pt")
    rec["text_tokenization"] = time.perf_counter() - t0

    # 전처리 (+ 내부 재토크나이즈 포함이므로 토크나이즈 추정치를 뺌)
    t0 = time.perf_counter()
    try:
        inputs = processor(text=[text], videos=[frames], return_tensors="pt")
    except Exception:
        pil = [Image.fromarray(f) for f in frames]
        messages = [{"role": "user", "content": [{"type": "image"} for _ in pil] + [{"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=pil, return_tensors="pt")
    rec["video_preprocess"] = max(time.perf_counter() - t0 - rec["text_tokenization"], 0.0)

    inputs = inputs.to(model.device)
    for k, v in inputs.items():
        if torch.is_floating_point(v) if torch.is_tensor(v) else False:
            inputs[k] = v.to(torch.bfloat16)

    hook_timer.reset()
    pd_timer.reset()
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

    rec["vision_encoder"] = hook_timer.total.get("vision", 0.0) - hook_timer.total.get("proj_nested", 0.0)
    rec["projector"] = hook_timer.total.get("proj_nested", 0.0) + hook_timer.total.get("proj", 0.0)
    rec["llm_prefill"] = pd_timer.prefill
    rec["autoregressive_decode"] = pd_timer.decode

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    t0 = time.perf_counter()
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    rec["detokenization"] = time.perf_counter() - t0
    rec["n_generated_tokens"] = int(new_tokens.shape[0])
    return answer


# ------------------------------------------------------------- LLaVA models


def load_llava(family, pretrained):
    from llava.model.builder import load_pretrained_model

    overwrite = {"mm_spatial_pool_mode": "average"} if family == "llava_vid" else None
    t0 = time.perf_counter()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        pretrained, None, "llava_qwen", device_map="cuda",
        torch_dtype="bfloat16", overwrite_config=overwrite,
    )
    model.eval()
    load_s = time.perf_counter() - t0
    return model, (tokenizer, image_processor), load_s


def run_sample_llava(model, proc, frames, question, max_new_tokens, hook_timer, pd_timer, rec):
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    tokenizer, image_processor = proc

    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    text = conv.get_prompt()

    t0 = time.perf_counter()
    input_ids = tokenizer_image_token(text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    rec["text_tokenization"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    rec["video_preprocess"] = time.perf_counter() - t0
    video = video.to(dtype=torch.bfloat16, device=model.device)

    hook_timer.reset()
    pd_timer.reset()
    out = model.generate(
        input_ids.unsqueeze(0).to(model.device),
        images=[video], modalities=["video"],
        do_sample=False, max_new_tokens=max_new_tokens,
    )

    rec["vision_encoder"] = hook_timer.total.get("vision", 0.0)
    rec["projector"] = hook_timer.total.get("proj", 0.0) + hook_timer.total.get("proj_nested", 0.0)
    rec["llm_prefill"] = pd_timer.prefill
    rec["autoregressive_decode"] = pd_timer.decode

    t0 = time.perf_counter()
    answer = tokenizer.decode(out[0], skip_special_tokens=True)
    rec["detokenization"] = time.perf_counter() - t0
    rec["n_generated_tokens"] = int(out.shape[1])
    return answer


# -------------------------------------------------------------------- data


def pick(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def load_samples(dataset, n_per_duration, seed):
    from datasets import load_dataset

    repo = {"videomme": "lmms-eval/Video-MME", "videomme_v2": "MME-Benchmarks/Video-MME-v2"}[dataset]
    ds = load_dataset(repo, split="test")
    cols = ds.column_names
    print(f"dataset columns: {cols}")

    by_dur = defaultdict(list)
    for row in ds:
        dur = pick(row, ["duration", "duration_category", "video_duration", "length"])
        vid = pick(row, ["videoID", "video_id", "video", "video_name", "videoID_path"])
        q = pick(row, ["question"])
        opts = pick(row, ["options", "choices"]) or []
        ans = pick(row, ["answer", "correct_answer"])
        if dur is None or vid is None or q is None:
            continue
        dur = str(dur).lower()
        prompt = q + "\n" + "\n".join(str(o) for o in opts) + "\nAnswer with the option's letter from the given choices directly."
        by_dur[dur].append({"videoID": str(vid), "prompt": prompt, "answer": ans})

    rng = random.Random(seed)
    picked = {}
    for dur in ("short", "medium", "long"):
        pool = by_dur.get(dur, [])
        if not pool:
            print(f"[warn] duration '{dur}' 샘플이 없습니다. (있는 값: {list(by_dur)})")
            continue
        rng.shuffle(pool)
        picked[dur] = pool[:n_per_duration]
        print(f"{dur}: {len(picked[dur])}개 샘플 선택 (전체 {len(pool)})")
    return picked


def index_videos(video_dir):
    index = {}
    for root, _, files in os.walk(os.path.expanduser(video_dir)):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                index[os.path.splitext(f)[0]] = os.path.join(root, f)
    print(f"video index: {len(index)}개 파일 ({video_dir})")
    return index


# -------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=HF_FAMILIES + LLAVA_FAMILIES)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--dataset", default="videomme", choices=["videomme", "videomme_v2"])
    ap.add_argument("--video_dir", required=True, help="비디오 파일들이 풀려 있는 디렉토리 (재귀 탐색)")
    ap.add_argument("--n_per_duration", type=int, default=50)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    samples = load_samples(args.dataset, args.n_per_duration, args.seed)
    vindex = index_videos(args.video_dir)

    print(f"\n== loading {args.pretrained} ==")
    if args.family in LLAVA_FAMILIES:
        model, proc, load_s = load_llava(args.family, args.pretrained)
        run_sample = run_sample_llava
    else:
        model, proc, load_s = load_hf(args.family, args.pretrained)
        run_sample = run_sample_hf
    print(f"checkpoint_load: {load_s:.1f}s")

    # vision/projector 훅 설치
    ht = HookTimer()
    if args.family in LLAVA_FAMILIES:
        vname, vmod = ("vision_tower", model.get_vision_tower()) if hasattr(model, "get_vision_tower") else (None, None)
        pmod = getattr(model.get_model(), "mm_projector", None) if hasattr(model, "get_model") else None
        pname = "mm_projector" if pmod is not None else None
    else:
        vname, vmod = find_module(model, VISION_CANDIDATES)
        pname, pmod = find_module(model, PROJ_CANDIDATES)
    if vmod is not None:
        ht.attach("vision", vmod)
        print(f"vision hook: {vname}")
    else:
        print("[warn] vision encoder 모듈을 찾지 못했습니다 — vision 시간은 prefill에 포함됩니다")
    if pmod is not None:
        nested = vname is not None and pname is not None and pname.startswith(vname)
        ht.attach("proj_nested" if nested else "proj", pmod)
        print(f"projector hook: {pname} (vision 내부: {nested})")
    else:
        print("[warn] projector 모듈을 찾지 못했습니다 — projector 시간은 vision/prefill에 포함됩니다")

    pd = PrefillDecodeTimer(model, ht, nested_keys=["vision", "proj", "proj_nested"])

    results = defaultdict(list)
    skipped = 0
    for dur, rows in samples.items():
        for i, row in enumerate(rows):
            path = vindex.get(row["videoID"])
            if path is None:
                skipped += 1
                continue
            rec = {}
            try:
                t0 = time.perf_counter()
                frames = read_frames(path, args.num_frames)
                rec["video_io_frame_extract"] = time.perf_counter() - t0
                run_sample(model, proc, frames, row["prompt"], args.max_new_tokens, ht, pd, rec)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[oom] {dur} #{i} {row['videoID']} — 건너뜀")
                skipped += 1
                continue
            except Exception as e:
                print(f"[err] {dur} #{i} {row['videoID']}: {type(e).__name__}: {e}")
                skipped += 1
                continue
            results[dur].append(rec)
            if (i + 1) % 10 == 0:
                print(f"{dur}: {i + 1}/{len(rows)}")

    if skipped:
        print(f"\n[warn] {skipped}개 샘플 건너뜀 (비디오 파일 없음/에러)")

    # 집계 + 출력
    report = {"family": args.family, "pretrained": args.pretrained, "dataset": args.dataset,
              "num_frames": args.num_frames, "max_new_tokens": args.max_new_tokens,
              "checkpoint_load_s": round(load_s, 2), "per_duration": {}}
    all_recs = []
    for dur in ("short", "medium", "long"):
        recs = results.get(dur, [])
        all_recs += recs
        if recs:
            report["per_duration"][dur] = {
                "n": len(recs),
                **{s: round(sum(r.get(s, 0.0) for r in recs) / len(recs), 4) for s in STAGES},
                "mean_generated_tokens": round(sum(r.get("n_generated_tokens", 0) for r in recs) / len(recs), 1),
            }
    if all_recs:
        report["overall"] = {
            "n": len(all_recs),
            **{s: round(sum(r.get(s, 0.0) for r in all_recs) / len(all_recs), 4) for s in STAGES},
        }

    print(f"\n===== latency breakdown (평균, 초/샘플) — {args.pretrained} =====")
    header = ["stage"] + [d for d in ("short", "medium", "long") if d in report["per_duration"]] + ["overall"]
    print("  ".join(f"{h:>24s}" for h in header))
    print(f"{'checkpoint_load(1회)':>24s}  " + f"{load_s:>24.2f}")
    for s in STAGES:
        row = [f"{s:>24s}"]
        for d in header[1:-1]:
            row.append(f"{report['per_duration'][d][s]:>24.4f}")
        row.append(f"{report.get('overall', {}).get(s, 0.0):>24.4f}")
        print("  ".join(row))

    out = args.out or f"latency_results/{os.path.basename(args.pretrained)}_{args.dataset}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"\nsaved: {out}")

    ht.remove()
    pd.remove()


if __name__ == "__main__":
    main()
