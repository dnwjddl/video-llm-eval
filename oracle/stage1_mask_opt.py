#!/usr/bin/env python3
"""Stage 1 — 비디오(·질문)별 최소 충분 토큰 부분집합(oracle) 탐색.

토큰마다 θ_i 를 두고 m = sigmoid(θ/τ) 를 attention bias(log m)로 frozen LLM 에 넣어
    loss(θ) = mean_q KL(p_full,q ‖ p_m,q) + λ · mean(m)
를 θ 에 대해서만 Adam 으로 최적화한다. λ 를 작은 값부터 훑으며(warm start) λ 마다
    S_λ = {i : m_i > 0.5}
를 얻고, S_λ 만 남기고 **실제로 삭제**(position 유지)한 forward 로 질문별 KL 을 재검증한다.
결과는 비디오마다 (|S_λ|, KL_hard) 점 여섯 개 = rate-distortion 곡선.

모드
  agnostic : 한 비디오의 모든 질문 KL 을 함께 만족하는 마스크 하나 (질문을 모르는 encoder 의 상한)
  aware    : 질문마다 마스크 하나

기준선 (같은 |S_λ| 에서, 학습 없이 한 번의 forward)
  random        : 토큰 무작위
  frame_uniform : 프레임을 균등 간격으로 골라 통째로 유지
  grid          : 프레임마다 같은 공간 격자 위치만 유지 (균등 공간 pooling 의 '선택' 판)

실행 (llava 환경):
  python oracle/stage1_mask_opt.py --pretrained lmms-lab/llava-onevision-qwen2-0.5b-ov --mode agnostic --limit 5
  python oracle/stage1_mask_opt.py --pretrained lmms-lab/llava-onevision-qwen2-7b-ov  --mode agnostic --limit 100
  python oracle/stage1_mask_opt.py --pretrained lmms-lab/llava-onevision-qwen2-7b-ov  --mode aware    --limit 100

데이터: Video-MME 비디오 145개(~/videomme_videos) × 그 비디오의 질문 전부(3개, task_type 라벨 포함).
결과: oracle/results/stage1_<model>_<mode>/<videoID>.json  (+ masks_<videoID>.npz)
"""

import argparse
import contextlib
import json
import os
import sys
import time
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "latency"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from llava_hooks import (  # noqa: E402
    build_prompt_ids, clear_state, delete_tokens, encode_video_inputs, install_bias_patch,
    last_logits, letter_dist, letter_token_ids, letters_in_prompt, load_llava, make_bias, math_sdpa,
)


# ----------------------------------------------------------------------------
# 데이터: Video-MME 를 비디오 단위로 묶기
# ----------------------------------------------------------------------------
def load_videomme_grouped(video_index):
    """{videoID: [ {prompt, answer, task_type, duration, qid}, ... ]} — 로컬에 비디오가 있는 것만."""
    from datasets import load_dataset

    ds = load_dataset("lmms-eval/Video-MME", split="test")
    groups = defaultdict(list)
    for row in ds:
        vid = str(row["videoID"])
        if vid not in video_index:
            continue
        opts = row.get("options") or []
        prompt = row["question"] + "\n" + "\n".join(str(o) for o in opts) + \
            "\nAnswer with the option's letter from the given choices directly."
        groups[vid].append({
            "qid": str(row.get("question_id", "")), "prompt": prompt,
            "answer": str(row.get("answer", "")).strip().strip("()").upper()[:1],
            "task_type": str(row.get("task_type", "")), "duration": str(row.get("duration", "")),
        })
    return dict(groups)


# ----------------------------------------------------------------------------
# 기준선 subset 생성
# ----------------------------------------------------------------------------
def baseline_keep(name, n_keep, n_vis, n_frames, gen):
    per = n_vis // n_frames
    keep = torch.zeros(n_vis, dtype=torch.bool)
    if n_keep <= 0:
        return keep
    if name == "random":
        keep[torch.randperm(n_vis, generator=gen)[:n_keep]] = True
    elif name == "frame_uniform":
        k = max(1, min(n_frames, int(round(n_keep / per))))
        fidx = torch.linspace(0, n_frames - 1, k).round().long().unique()
        for f in fidx.tolist():
            keep[f * per:(f + 1) * per] = True
    elif name == "grid":
        # 프레임마다 같은 위치 집합: 14×14 격자에서 균등 간격으로 k_pos 개
        side = int(round(per ** 0.5))
        k_pos = max(1, min(per, int(round(n_keep / n_frames))))
        s = max(1, int(round((per / k_pos) ** 0.5)))
        pos = [r * side + c for r in range(0, side, s) for c in range(0, side, s)][:k_pos]
        if len(pos) < k_pos:  # 부족하면 순서대로 채움
            extra = [p for p in range(per) if p not in set(pos)][:k_pos - len(pos)]
            pos += extra
        for f in range(n_frames):
            for p in pos:
                keep[f * per + p] = True
    else:
        raise ValueError(name)
    return keep


# ----------------------------------------------------------------------------
# 평가: subset 을 실제로 삭제하고 질문별 KL
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_subset(model, qs, keep):
    """qs: [{vi, lid, p_full, letters, gt}], keep: (n_vis,) bool → 질문별 지표."""
    out = []
    for q in qs:
        emb, pos = delete_tokens(q["vi"], keep, renumber=False)
        logits = last_logits(model, emb, position_ids=pos)
        p = letter_dist(logits, q["lid"])
        pf = q["p_full"]
        kl = float((pf * (pf.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum())
        pred = q["letters"][int(p.argmax())]
        out.append({"kl": kl, "pred": pred, "same_as_full": pred == q["full_pred"], "correct": pred == q["gt"]})
    return out


def agg(evals):
    return {"kl_mean": float(np.mean([e["kl"] for e in evals])), "kl_max": float(np.max([e["kl"] for e in evals])),
            "same_as_full": float(np.mean([e["same_as_full"] for e in evals])),
            "acc": float(np.mean([e["correct"] for e in evals]))}


# ----------------------------------------------------------------------------
# 마스크 최적화
# ----------------------------------------------------------------------------
def optimize_mask(model, qs, args, log_prefix=""):
    """qs 전체의 KL 을 함께 낮추는 마스크 하나. λ sweep(warm start) → [(λ, m_soft, keep, steps_stats)]."""
    n_vis = qs[0]["vi"].n_vis
    theta = torch.zeros(n_vis, device=model.device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=args.lr)
    lambdas = [float(x) for x in args.lambdas.split(",")]
    results = []
    gctx = math_sdpa() if args.grad_kernel == "math" else contextlib.nullcontext()
    if not args.no_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    try:
        with gctx:
            for li, lam in enumerate(lambdas):
                steps = args.steps_first if li == 0 else args.steps_next
                t0 = time.perf_counter()
                step_ptr = 0
                for s in range(steps):
                    tau = args.tau_start + (args.tau_end - args.tau_start) * min(1.0, s / max(1, steps - 1)) \
                        if li == 0 else args.tau_end
                    opt.zero_grad(set_to_none=True)
                    # step 마다 질문 q_per_step 개를 돌아가며 사용 (agnostic 에서 비용 절약; 전부 쓰려면 --q_per_step 0)
                    k = len(qs) if args.q_per_step <= 0 else min(args.q_per_step, len(qs))
                    batch = [qs[(step_ptr + j) % len(qs)] for j in range(k)]
                    step_ptr = (step_ptr + k) % len(qs)
                    kl_sum = 0.0
                    for q in batch:  # 질문별 fwd+bwd 를 순차로 누적 (메모리 절약)
                        m = torch.sigmoid(theta / tau)          # 질문마다 그래프를 새로 (backward 가 그래프를 해제하므로)
                        bias = make_bias(q["vi"], m)
                        logits = last_logits(model, q["vi"].embeds, bias=bias, clear=False)
                        p = letter_dist(logits, q["lid"])
                        pf = q["p_full"]
                        kl = (pf * (pf.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum()
                        (kl / k).backward()
                        clear_state()
                        kl_sum += float(kl)
                    m2 = torch.sigmoid(theta / tau)
                    (lam * m2.mean()).backward()                 # λ 항 (rate)
                    opt.step()
                    if s % max(1, steps // 5) == 0 or s == steps - 1:
                        with torch.no_grad():
                            frac = float((torch.sigmoid(theta / tau) > 0.5).float().mean())
                        print(f"{log_prefix}  λ={lam:g} step {s:3d}/{steps} kl={kl_sum / k:.4f} "
                              f"keep={frac:.3f} τ={tau:.2f}", flush=True)
                with torch.no_grad():
                    m_soft = torch.sigmoid(theta / args.tau_end).detach()
                    keep = (m_soft > 0.5).cpu()
                results.append({"lambda": lam, "m_soft": m_soft.cpu().numpy().astype(np.float16),
                                "keep": keep, "sec": time.perf_counter() - t0, "steps": steps})
    finally:
        clear_state()
        model.eval()
        if not args.no_checkpoint:
            model.gradient_checkpointing_disable()
    return results


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    ap.add_argument("--video_dir", default=os.path.expanduser("~/videomme_videos"))
    ap.add_argument("--mode", default="agnostic", choices=["agnostic", "aware"])
    ap.add_argument("--limit", type=int, default=None, help="비디오 수 제한")
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--lambdas", default="0.03,0.1,0.3,1,3,10")
    ap.add_argument("--steps_first", type=int, default=150)
    ap.add_argument("--steps_next", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--q_per_step", type=int, default=1, help="step 당 질문 수 (round-robin). 0 = 전부")
    ap.add_argument("--tau_start", type=float, default=1.0)
    ap.add_argument("--tau_end", type=float, default=0.1)
    ap.add_argument("--baselines", default="random,frame_uniform,grid")
    ap.add_argument("--baseline_seeds", type=int, default=2)
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--grad_kernel", default="math", choices=["math", "default"])
    ap.add_argument("--no_checkpoint", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--resume", action="store_true", help="결과 JSON 이 있는 비디오는 건너뜀")
    args = ap.parse_args()

    from profile_latency import index_videos, read_frames

    tag = args.pretrained.split("/")[-1]
    out_dir = args.out_dir or os.path.join(BASE, "results", f"stage1_{tag}_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out_dir, "_args.json"), "w"), indent=2)

    vindex = index_videos(args.video_dir)
    groups = load_videomme_grouped(vindex)
    vids = sorted(groups)
    if args.limit:
        vids = vids[: args.limit]
    print(f"[data] {len(vids)} videos, {sum(len(groups[v]) for v in vids)} questions", flush=True)

    print("loading model...", flush=True)
    tokenizer, model, image_processor = load_llava(args.pretrained, args.attn_impl)
    install_bias_patch(model)
    baselines = [b for b in args.baselines.split(",") if b]

    for vi_i, vid in enumerate(vids):
        out_json = os.path.join(out_dir, f"{vid}.json")
        if args.resume and os.path.exists(out_json):
            continue
        t_video = time.perf_counter()
        frames = read_frames(vindex[vid], args.num_frames)

        # 질문별 입력·기준 분포 (encoder 는 질문마다 다시 돌지만 32프레임 SigLIP 이라 수 초)
        qs = []
        with torch.no_grad(), (math_sdpa() if args.grad_kernel == "math" else contextlib.nullcontext()):
            for q in groups[vid]:
                ids = build_prompt_ids(tokenizer, q["prompt"])
                vi = encode_video_inputs(model, image_processor, frames, ids)
                letters = letters_in_prompt(q["prompt"])
                lid = letter_token_ids(tokenizer, letters).to(model.device)
                p_full = letter_dist(last_logits(model, vi.embeds), lid)
                qs.append({"vi": vi, "lid": lid, "letters": letters, "p_full": p_full,
                           "full_pred": letters[int(p_full.argmax())], "gt": q["answer"],
                           "task_type": q["task_type"], "qid": q["qid"]})
        n_vis, n_frames = qs[0]["vi"].n_vis, qs[0]["vi"].n_frames

        groups_to_opt = [qs] if args.mode == "agnostic" else [[q] for q in qs]
        record = {"videoID": vid, "mode": args.mode, "n_vis": n_vis, "n_frames": n_frames,
                  "questions": [{"qid": q["qid"], "task_type": q["task_type"], "gt": q["gt"],
                                 "full_pred": q["full_pred"], "full_correct": q["full_pred"] == q["gt"]} for q in qs],
                  "runs": []}
        masks = {}
        gen = torch.Generator().manual_seed(args.seed)
        for gi, g in enumerate(groups_to_opt):
            prefix = f"[{vi_i + 1}/{len(vids)} {vid}" + (f" q{gi}" if args.mode == "aware" else "") + "]"
            sweep = optimize_mask(model, g, args, log_prefix=prefix)
            run = {"group": [q["qid"] for q in g], "points": []}
            for r in sweep:
                keep = r["keep"]
                n_keep = int(keep.sum())
                with (math_sdpa() if args.grad_kernel == "math" else contextlib.nullcontext()):
                    ev = eval_subset(model, g, keep)
                    pt = {"lambda": r["lambda"], "n_keep": n_keep, "keep_frac": n_keep / n_vis,
                          "oracle": agg(ev), "per_q": ev, "opt_sec": r["sec"], "baselines": {}}
                    for b in baselines:
                        evs = []
                        for sd in range(args.baseline_seeds if b == "random" else 1):
                            bk = baseline_keep(b, n_keep, n_vis, n_frames, gen)
                            evs += eval_subset(model, g, bk)
                        pt["baselines"][b] = agg(evs)
                run["points"].append(pt)
                masks[f"g{gi}_lam{r['lambda']:g}_soft"] = r["m_soft"]
                masks[f"g{gi}_lam{r['lambda']:g}_keep"] = keep.numpy()
                bl = " ".join(f"{b}={pt['baselines'][b]['kl_mean']:.3f}" for b in baselines)
                print(f"{prefix} λ={r['lambda']:g}: keep {n_keep}/{n_vis} ({n_keep / n_vis:.3f})  "
                      f"oracle KL={pt['oracle']['kl_mean']:.4f} same={pt['oracle']['same_as_full']:.2f}  | {bl}",
                      flush=True)
            record["runs"].append(run)
        record["sec"] = time.perf_counter() - t_video
        json.dump(record, open(out_json, "w"), indent=2)
        np.savez_compressed(os.path.join(out_dir, f"masks_{vid}.npz"), **masks)
        print(f"{'[' + vid + ']'} done in {record['sec']:.0f}s → {out_json}", flush=True)


if __name__ == "__main__":
    main()
