"""Text-only (blind) inference with option rotation.

Scoring modes
  logits   (default) one forward pass per prompt; the answer is the option letter with the
           highest next-token logit. No parsing failures, no refusals; deterministic.
  generate greedy generation + lenient letter parsing (lmms-eval style). Refusals count as wrong.

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

from .schema import DEFAULT_POST_PROMPT, LETTERS, build_prompt, frame_to_items, parse_letter, rotate, rotation_shifts

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

    def _letter_ids(self, n):
        ids = []
        for L in LETTERS[:n]:
            cand = set()
            for v in (L, " " + L, L + ".", "(" + L):
                t = self.tok.encode(v, add_special_tokens=False)
                if len(t) >= 1:
                    cand.add(t[0])
            ids.append(sorted(cand))
        return ids

    def score_letters(self, prompts, n_options):
        """Return (pred_idx, raw) per prompt using next-token logits over option letters."""
        import torch

        texts = [self.chat(p) for p in prompts]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        preds = [None] * len(texts)
        cache = {}
        for b in tqdm(range(0, len(order), self.bs), desc="score"):
            idx = order[b : b + self.bs]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :].float()
            for row, i in enumerate(idx):
                n = n_options[i]
                if n not in cache:
                    cache[n] = self._letter_ids(n)
                scores = [logits[row, ids].max().item() if ids else float("-inf") for ids in cache[n]]
                k = max(range(n), key=lambda j: scores[j])
                srt = sorted(scores, reverse=True)
                preds[i] = (k, f"{LETTERS[k]} (margin {srt[0] - srt[1]:.2f})")
        return preds

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

    def score_letters(self, prompts, n_options):
        from vllm import SamplingParams

        res = self.llm.generate([self.chat(p) for p in prompts], SamplingParams(temperature=0, max_tokens=1, logprobs=30))
        out = []
        for r, n in zip(res, n_options):
            lp = r.outputs[0].logprobs[0] if r.outputs[0].logprobs else {}
            scores = [float("-inf")] * n
            for tid, info in lp.items():
                tok = (getattr(info, "decoded_token", None) or self.tok.decode([tid])).strip().strip("().:")
                if len(tok) == 1 and tok.upper() in LETTERS[:n]:
                    j = LETTERS.index(tok.upper())
                    scores[j] = max(scores[j], info.logprob)
            if max(scores) == float("-inf"):
                out.append((-1, "no letter in top-30 logprobs"))
            else:
                k = max(range(n), key=lambda j: scores[j])
                out.append((k, f"{LETTERS[k]} (logprob {scores[k]:.2f})"))
        return out


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
    ap.add_argument("--scoring", choices=["logits", "generate"], default="logits")
    ap.add_argument("--benchmarks", default="", help="comma list: only run these benchmarks")
    ap.add_argument("--purge-benchmarks", default="", help="comma list: delete existing rows of these benchmarks from --out before (re)running")
    args = ap.parse_args()

    items = load_items(args.items)
    if args.benchmarks:
        keep = set(b.strip() for b in args.benchmarks.split(",") if b.strip())
        items = [it for it in items if it.benchmark in keep]
    if args.limit:
        items = items[: args.limit]
    reqs = make_requests(items, args.rotations, args.post_prompt)

    done = set()
    if os.path.exists(args.out):  # resume
        prev = pd.read_parquet(args.out)
        if args.purge_benchmarks:
            purge = set(b.strip() for b in args.purge_benchmarks.split(",") if b.strip())
            n0 = len(prev)
            prev = prev[~prev["benchmark"].isin(purge)]
            print(f"purged {n0 - len(prev)} rows of {sorted(purge)}")
            prev.to_parquet(args.out, index=False)
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
    if args.scoring == "logits":
        scored = eng.score_letters([r["prompt"] for r in reqs], [r["n_options"] for r in reqs])
        outs = [raw for _, raw in scored]
        preds = [k for k, _ in scored]
    else:
        outs = eng.generate([r["prompt"] for r in reqs])
        preds = [parse_letter(o, r["n_options"], r["options"]) for r, o in zip(reqs, outs)]
    rows = []
    for r, o, pred in zip(reqs, outs, preds):
        rows.append({"item_id": r["item_id"], "benchmark": r["benchmark"], "category": r["category"], "shift": r["shift"],
                     "n_options": r["n_options"], "answer_idx": r["answer_idx"], "pred_idx": pred, "correct": int(pred == r["answer_idx"]),
                     "parsed": int(pred >= 0), "raw_output": o, "model": args.model, "scoring": args.scoring})
    df = pd.DataFrame(rows)
    bad = df[df["parsed"] == 0]["raw_output"].value_counts().head(10)
    if len(bad):
        print("most common unparsed outputs:\n" + bad.to_string())
    if done:
        df = pd.concat([pd.read_parquet(args.out), df], ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}  ({time.time() - t0:.0f}s)  parse-fail rate={1 - df['parsed'].mean():.3%}")


if __name__ == "__main__":
    main()
