"""Text-only (blind) inference with option rotation.

For each item and each cyclic shift s in [0, K), the LLM sees ONLY the question
and the rotated options. Output: one row per (item, shift) with the raw text,
parsed option index, rotated answer index and correctness.

Examples
  python -m encoder_study.blind.run_blind --items items/*.parquet --model Qwen/Qwen2.5-7B-Instruct --out preds/qwen7b.parquet
  python -m encoder_study.blind.run_blind --items items/*.parquet --model Qwen/Qwen2.5-32B-Instruct --engine vllm --out preds/qwen32b.parquet
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

from .schema import DEFAULT_POST_PROMPT, build_prompt, frame_to_items, parse_letter, rotate, rotation_shifts

SYSTEM_PROMPT = "You are a helpful assistant."


def pick_device() -> str:
    """Use the GPU with the most free memory (respects CUDA_VISIBLE_DEVICES; explicit --device overrides)."""
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"cuda:{i} free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
        if free > best_free:
            best, best_free = i, free
    print(f"-> using cuda:{best}")
    return f"cuda:{best}"


def load_items(patterns):
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    if not files:
        sys.exit(f"no item files match {patterns}")
    return frame_to_items(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))


def make_requests(items, k, post_prompt):
    reqs = []
    for it in items:
        for s in rotation_shifts(it.n_options(), k):
            opts, ans = rotate(it.options, it.answer_idx, s)
            reqs.append({"item_id": it.item_id, "benchmark": it.benchmark, "category": it.category, "shift": s,
                         "n_options": len(opts), "answer_idx": ans, "options": opts,
                         "prompt": build_prompt(it.question, opts, post_prompt)})
    return reqs


class HFEngine:
    def __init__(self, model, dtype="bfloat16", batch_size=16, max_new_tokens=8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.device = pick_device()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype), device_map=self.device).eval()
        except TypeError:  # older transformers
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=getattr(torch, dtype), device_map=self.device).eval()
        self.bs, self.mnt = batch_size, max_new_tokens

    def chat(self, prompt):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts):
        import torch

        texts = [self.chat(p) for p in prompts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        outs = [None] * len(texts)
        for b in tqdm(range(0, len(order), self.bs), desc="generate"):
            idx = order[b : b + self.bs]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.mnt, do_sample=False, temperature=None, top_p=None,
                                          pad_token_id=self.tok.pad_token_id)
            dec = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for i, d in zip(idx, dec):
                outs[i] = d
        return outs


class VLLMEngine:
    def __init__(self, model, dtype="bfloat16", max_new_tokens=8, tensor_parallel=1, **_):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, dtype=dtype, tensor_parallel_size=tensor_parallel)
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)

    def chat(self, prompt):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts):
        res = self.llm.generate([self.chat(p) for p in prompts], self.sp)
        return [r.outputs[0].text for r in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True, help="parquet globs from build_items")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tensor-parallel", type=int, default=1)
    ap.add_argument("--post-prompt", default=DEFAULT_POST_PROMPT)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N items")
    ap.add_argument("--device", default="", help="e.g. cuda:1; default = GPU with most free memory")
    args = ap.parse_args()

    items = load_items(args.items)
    if args.limit:
        items = items[: args.limit]
    reqs = make_requests(items, args.rotations, args.post_prompt)

    done = set()
    if os.path.exists(args.out):  # resume
        prev = pd.read_parquet(args.out)
        done = set(zip(prev["item_id"], prev["shift"]))
        reqs = [r for r in reqs if (r["item_id"], r["shift"]) not in done]
        print(f"resuming: {len(done)} rows done, {len(reqs)} to go")
    if not reqs:
        print("nothing to do")
        return

    if args.device:
        globals()["pick_device"] = lambda: args.device
    eng = (HFEngine(args.model, args.dtype, args.batch_size, args.max_new_tokens) if args.engine == "hf"
           else VLLMEngine(args.model, args.dtype, args.max_new_tokens, args.tensor_parallel))
    t0 = time.time()
    outs = eng.generate([r["prompt"] for r in reqs])
    rows = []
    for r, o in zip(reqs, outs):
        pred = parse_letter(o, r["n_options"], r["options"])
        rows.append({"item_id": r["item_id"], "benchmark": r["benchmark"], "category": r["category"], "shift": r["shift"],
                     "n_options": r["n_options"], "answer_idx": r["answer_idx"], "pred_idx": pred, "correct": int(pred == r["answer_idx"]),
                     "parsed": int(pred >= 0), "raw_output": o, "model": args.model})
    df = pd.DataFrame(rows)
    if done:
        df = pd.concat([pd.read_parquet(args.out), df], ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}  ({time.time() - t0:.0f}s)  parse-fail rate={1 - df['parsed'].mean():.3%}")


if __name__ == "__main__":
    main()
