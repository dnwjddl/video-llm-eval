#!/usr/bin/env python3
"""토큰 압축 ablation을 Video-MME 150샘플(breakdown과 동일 seed)로 —
비디오 길이(short/medium/long)별 열화를 측정한다.

실행 (llava 환경):
  python token_analysis/run_videomme_ablation.py --method pool_avg --keep 0.25 \
      --video_dir ~/videomme_videos

- latency/extract_videos_subset.py 로 풀어둔 그 150개(seed 42)를 그대로 사용.
- Video-MME test는 정답 letter가 데이터셋에 있어 자체 채점 (MCQ letter 매칭).
- 결과: token_analysis/results_videomme/<method>_keep<keep>.json
"""

import argparse
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "latency"))

import torch  # noqa: E402


def extract_letter(pred):
    m = re.match(r"^\s*\(?([A-E])[).:\s]", pred + " ")
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-E])\b", pred)
    return m.group(1).upper() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--keep", type=float, required=True)
    ap.add_argument("--video_dir", default=os.path.expanduser("~/videomme_videos"))
    ap.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    ap.add_argument("--n_per_duration", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    args = ap.parse_args()

    from compress import METHODS, compress
    from profile_latency import index_videos, load_samples, read_frames

    assert args.method in METHODS, f"지원 방법: {list(METHODS)}"

    # --- llava pooling 지점 패치 (run_token_ablation.py와 동일 방식) ---
    import llava.model.llava_arch as llava_arch

    orig = llava_arch.LlavaMetaForCausalLM.get_2dPool
    method, keep = args.method, args.keep

    def patched(self, image_feature, stride=2):
        if method == "none":
            return orig(self, image_feature, stride)
        return compress(image_feature, method, keep)

    llava_arch.LlavaMetaForCausalLM.get_2dPool = patched
    print(f"[videomme_ablation] method={method}, keep={keep}", flush=True)

    samples = load_samples("videomme", args.n_per_duration, args.seed)
    vindex = index_videos(args.video_dir)

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.model.builder import load_pretrained_model

    print("loading model...", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.pretrained, None, "llava_qwen", device_map="cuda",
        torch_dtype="bfloat16", attn_implementation="sdpa",
    )
    model.eval()

    out_path = os.path.join(BASE, "results_videomme", f"{method}_keep{keep}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    stats = {}
    for dur, rows in samples.items():
        correct = total = 0
        for i, row in enumerate(rows):
            path = vindex.get(row["videoID"])
            if path is None or row.get("answer") is None:
                continue
            try:
                frames = read_frames(path, args.num_frames)
                conv = conv_templates["qwen_1_5"].copy()
                conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + row["prompt"])
                conv.append_message(conv.roles[1], None)
                input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
                video = video.to(dtype=torch.bfloat16, device=model.device)
                with torch.no_grad():
                    out = model.generate(
                        input_ids.unsqueeze(0).to(model.device),
                        images=[video], modalities=["video"],
                        do_sample=False, max_new_tokens=args.max_new_tokens,
                    )
                pred = tokenizer.decode(out[0], skip_special_tokens=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[oom] {dur} #{i} — 건너뜀", flush=True)
                continue
            except Exception as e:
                print(f"[err] {dur} #{i}: {type(e).__name__}: {e}", flush=True)
                continue
            letter = extract_letter(pred)
            gt = str(row["answer"]).strip().strip("()").upper()[:1]
            total += 1
            correct += int(letter == gt)
            if (i + 1) % 10 == 0:
                print(f"{dur} {i + 1}/{len(rows)}: acc so far {correct}/{total}", flush=True)
        if total:
            stats[dur] = {"n": total, "acc": round(correct / total, 4)}
            print(f"[{dur}] acc = {correct}/{total} = {correct / total:.3f}", flush=True)
        # duration마다 중간 저장
        json.dump({"method": method, "keep": keep, "pretrained": args.pretrained,
                   "num_frames": args.num_frames, "per_duration": stats}, open(out_path, "w"), indent=2)

    all_n = sum(v["n"] for v in stats.values())
    all_c = sum(round(v["acc"] * v["n"]) for v in stats.values())
    report = {"method": method, "keep": keep, "pretrained": args.pretrained,
              "num_frames": args.num_frames, "per_duration": stats,
              "overall": {"n": all_n, "acc": round(all_c / all_n, 4) if all_n else None},
              "complete": True}
    json.dump(report, open(out_path, "w"), indent=2)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
