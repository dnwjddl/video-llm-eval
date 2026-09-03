#!/usr/bin/env python3
"""Stage 0 — "attention bias 로 가리기" ≈ "실제로 지우기" 등가성 검증 + fwd/bwd 비용 측정.

이후 모든 oracle 실험은 비주얼 토큰에 log m 을 attention bias 로 더하는 방식으로
마스킹한다. 그 전에 다음을 확인한다.

  (1) bias 로 가린 결과 == 토큰을 실제로 제거하되 position id 를 원래대로 둔 결과
      → 두 logit 이 수치 오차 안에서 같아야 한다. (같지 않으면 이후 단계 진행 불가)
  (2) 토큰 제거 후 position id 를 다시 매긴(renumber) 결과는 (1)과 얼마나 다른가
      → LLaVA-OV 는 1D RoPE 라 관례에 따라 결과가 달라질 수 있다. 차이의 크기를 기록.
  (3) (--timing) 연속 마스크 m=sigmoid(θ) 에 대해 KL 손실의 fwd+bwd 한 step 시간·메모리
      → 이후 단계의 비디오당 비용 추정. grad 가 θ 까지 실제로 흐르는지도 확인.

실행 (llava 환경, GPU 1장):
  python oracle/stage0_equivalence.py --pretrained lmms-lab/llava-onevision-qwen2-0.5b-ov --n 5 --timing
  python oracle/stage0_equivalence.py --pretrained lmms-lab/llava-onevision-qwen2-7b-ov  --n 20 --timing

비디오: latency/extract_videos_subset.py 로 풀어둔 Video-MME 150개 (기본 ~/videomme_videos).
결과: oracle/results/stage0_<model>.json
"""

import argparse
import contextlib
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(os.path.dirname(BASE), "latency"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from llava_hooks import (  # noqa: E402
    build_prompt_ids, clear_state, compare, delete_tokens, encode_video_inputs, install_bias_patch,
    last_logits, letter_dist, letter_token_ids, letters_in_prompt, load_llava, make_bias, math_sdpa, set_mask_impl,
)


def random_keep(n_vis, n_frames, ratio, gen, mode):
    """mode: 'token' = 토큰 단위 무작위, 'frame' = 프레임 단위 무작위 (프레임 전체를 지움)."""
    per = n_vis // n_frames
    if mode == "token":
        return torch.rand(n_vis, generator=gen) < ratio
    k = max(1, int(round(n_frames * ratio)))
    keep_f = torch.zeros(n_frames, dtype=torch.bool)
    keep_f[torch.randperm(n_frames, generator=gen)[:k]] = True
    return keep_f.repeat_interleave(per)


def summarize(rows, key_a, key_b):
    vals = [r[f"{key_a}_vs_{key_b}"] for r in rows if f"{key_a}_vs_{key_b}" in r]
    if not vals:
        return None
    return {
        "n": len(vals),
        "kl_mean": sum(v["kl_letters"] for v in vals) / len(vals),
        "kl_max": max(v["kl_letters"] for v in vals),
        "dlogit_vocab_max": max(v["max_abs_dlogit_vocab"] for v in vals),
        "dlogit_letters_max": max(v["max_abs_dlogit_letters"] for v in vals),
        "argmax_agree": sum(v["same_argmax"] for v in vals) / len(vals),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    ap.add_argument("--video_dir", default=os.path.expanduser("~/videomme_videos"))
    ap.add_argument("--n", type=int, default=20, help="검사할 (비디오, 질문) 수")
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--keeps", default="0.5,0.25,0.1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="float32 로 돌리면 (1)의 잔차가 bf16 반올림인지 확정할 수 있다 (0.5B 권장)")
    ap.add_argument("--grad_kernel", default="auto", choices=["auto", "math", "default"],
                    help="timing(fwd+bwd) 의 SDPA 백엔드. auto = default(mem-efficient, 시퀀스 padding 적용) 시도 후 실패하면 math")
    ap.add_argument("--pad_multiple", type=int, default=0, help="grad 경로에서 시퀀스 길이를 이 배수로 padding (bias 구현의 LSE 정렬 오류 회피용; keydim 에선 불필요)")
    ap.add_argument("--mask_impl", default="keydim", choices=["keydim", "bias"],
                    help="timing 에 쓸 마스크 구현. keydim = q/k 추가 차원 (mask 불필요, flash 가능), bias = 4D mask 에 log m")
    ap.add_argument("--timing", action="store_true", help="연속 마스크 fwd+bwd 비용 측정")
    ap.add_argument("--timing_steps", type=int, default=3)
    ap.add_argument("--kernel", default="math", choices=["math", "default"],
                    help="등가성 비교에 쓸 SDPA 백엔드. math = 모든 경로가 같은 커널을 쓰므로 순수 마스크 의미 차이만 남음")
    ap.add_argument("--no_checkpoint", action="store_true", help="timing 에서 gradient checkpointing 끄기 (메모리 여유 있을 때)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from profile_latency import index_videos, load_samples, read_frames

    tag = args.pretrained.split("/")[-1] + ("_fp32" if args.dtype == "float32" else "")
    out_path = args.out or os.path.join(BASE, "results", f"stage0_{tag}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("loading model...", flush=True)
    tokenizer, model, image_processor = load_llava(args.pretrained, args.attn_impl, args.dtype)
    attn_cls = install_bias_patch(model)
    set_mask_impl(args.mask_impl)
    print(f"[patch] {attn_cls.__name__}.forward 에 visual bias 훅 설치", flush=True)

    samples = load_samples("videomme", 50, 42)
    vindex = index_videos(args.video_dir)
    # short → medium → long 순으로 n 개 (짧은 것부터 빠르게)
    rows = [r for dur in ("short", "medium", "long") for r in samples.get(dur, [])]
    rows = [r for r in rows if vindex.get(r["videoID"]) is not None][: args.n]
    keeps = [float(x) for x in args.keeps.split(",")]
    gen = torch.Generator().manual_seed(args.seed)

    results, timing = [], None
    for i, row in enumerate(rows):
        frames = read_frames(vindex[row["videoID"]], args.num_frames)
        ids = build_prompt_ids(tokenizer, row["prompt"])
        vi = encode_video_inputs(model, image_processor, frames, ids)
        letters = letters_in_prompt(row["prompt"])
        lid = letter_token_ids(tokenizer, letters).to(model.device)
        if i == 0:
            print(f"[inputs] L={vi.L}, visual span=[{vi.vis_start},{vi.vis_end}) n_vis={vi.n_vis} "
                  f"({vi.n_frames} frames × {vi.n_vis // vi.n_frames}), newline={vi.has_newline}, letters={letters}",
                  flush=True)

        kctx = math_sdpa() if args.kernel == "math" else contextlib.nullcontext()
        with torch.no_grad():
            # 잡음 바닥: 같은 입력을 기본 커널과 math 커널로 — bf16 커널 차이만으로 logit 이 얼마나 흔들리는가
            full_default = last_logits(model, vi.embeds)
            with math_sdpa():
                full_math = last_logits(model, vi.embeds)
            full = full_math if args.kernel == "math" else full_default
        with torch.no_grad(), kctx:
            # 무마스크 bias(전부 0)가 full 과 같은지 — 패치 자체의 무해성 검사
            bias0 = make_bias(vi, torch.ones(vi.n_vis, device=model.device))
            full_bias0 = last_logits(model, vi.embeds, bias=bias0)
            rec = {"videoID": row["videoID"], "L": vi.L, "n_vis": vi.n_vis,
                   "full_argmax_letter": letters[int(letter_dist(full, lid).argmax())],
                   "gt": str(row.get("answer", "")).strip().strip("()").upper()[:1],
                   "noise_floor_full_vs_math": compare(full_default, full_math, lid),
                   "full_vs_fullbias0": compare(full, full_bias0, lid), "conds": []}

            for ratio in keeps:
                for mode in ("token", "frame"):
                    keep = random_keep(vi.n_vis, vi.n_frames, ratio, gen, mode)
                    m = keep.float().to(model.device)
                    set_mask_impl("keydim")
                    lb = last_logits(model, vi.embeds, bias=make_bias(vi, m))      # keydim 구현
                    set_mask_impl("bias")
                    lb_b = last_logits(model, vi.embeds, bias=make_bias(vi, m))    # bias 구현
                    set_mask_impl(args.mask_impl)
                    e_k, p_k = delete_tokens(vi, keep, renumber=False)
                    ld_keep = last_logits(model, e_k, position_ids=p_k)
                    e_r, p_r = delete_tokens(vi, keep, renumber=True)
                    ld_ren = last_logits(model, e_r, position_ids=p_r)
                    c = {"ratio": ratio, "mode": mode, "n_kept": int(keep.sum()),
                         "bias_vs_del_keeppos": compare(lb, ld_keep, lid),          # (keydim 구현)
                         "biasimpl_vs_del_keeppos": compare(lb_b, ld_keep, lid),    # (bias 구현)
                         "keydim_vs_biasimpl": compare(lb, lb_b, lid),
                         "bias_vs_del_renumber": compare(lb, ld_ren, lid),
                         "del_keeppos_vs_del_renumber": compare(ld_keep, ld_ren, lid),
                         "full_vs_bias": compare(full, lb, lid),
                         "bias_argmax_letter": letters[int(letter_dist(lb, lid).argmax())]}
                    rec["conds"].append(c)
        results.append(rec)
        w = max(rec["conds"], key=lambda c: c["bias_vs_del_keeppos"]["kl_letters"])["bias_vs_del_keeppos"]
        nf = rec["noise_floor_full_vs_math"]
        print(f"[{i + 1}/{len(rows)}] {row['videoID']}: bias-vs-delete(keep-pos) worst KL={w['kl_letters']:.2e} "
              f"max|Δ|={w['max_abs_dlogit_vocab']:.2e} argmax={'ok' if all(c['bias_vs_del_keeppos']['same_argmax'] for c in rec['conds']) else 'DIFF'} "
              f"| noise floor KL={nf['kl_letters']:.2e} max|Δ|={nf['max_abs_dlogit_vocab']:.2e} "
              f"| bias0 KL={rec['full_vs_fullbias0']['kl_letters']:.2e}", flush=True)

        # ---- (3) 연속 마스크 fwd+bwd 비용 (첫 샘플에서만) ----
        if args.timing and timing is None:
            kernels = ["default", "math"] if args.grad_kernel == "auto" else [args.grad_kernel]
            for gk in kernels:
                gctx = math_sdpa() if gk == "math" else contextlib.nullcontext()
                with gctx:
                    timing = run_timing(model, vi, full, lid, args.timing_steps,
                                        use_checkpoint=not args.no_checkpoint, pad_multiple=args.pad_multiple)
                timing["note"] += f" | mask_impl={args.mask_impl} grad_kernel={gk} pad_multiple={args.pad_multiple}"
                if timing["sec_per_step"] is not None:
                    break
                print(f"[timing] kernel={gk} 실패 → 다음 후보", flush=True)

    # ---- 요약 ----
    flat = [c for r in results for c in r["conds"]]
    summary = {
        "n_samples": len(results),
        "full_vs_fullbias0_dlogit_max": max(r["full_vs_fullbias0"]["max_abs_dlogit_vocab"] for r in results),
        "bias_vs_del_keeppos": summarize(flat, "bias", "del_keeppos"),
        "biasimpl_vs_del_keeppos": summarize(flat, "biasimpl", "del_keeppos"),
        "keydim_vs_biasimpl": summarize(flat, "keydim", "biasimpl"),
        "bias_vs_del_renumber": summarize(flat, "bias", "del_renumber"),
        "del_keeppos_vs_del_renumber": summarize(flat, "del_keeppos", "del_renumber"),
        "full_vs_bias": summarize(flat, "full", "bias"),
    }
    print("\n===== Stage 0 summary =====")
    print(f"패치 무해성  full vs bias(all 0):      max|Δlogit| = {summary['full_vs_fullbias0_dlogit_max']:.3e}")
    s = summary["bias_vs_del_keeppos"]
    print(f"(1) keydim 구현 vs 삭제(pos 유지): max|Δlogit|={s['dlogit_vocab_max']:.3e}  KL(mean/max)={s['kl_mean']:.2e}/{s['kl_max']:.2e}  argmax 일치={s['argmax_agree']:.3f}")
    s = summary["biasimpl_vs_del_keeppos"]
    print(f"(1') bias 구현 vs 삭제(pos 유지):  max|Δlogit|={s['dlogit_vocab_max']:.3e}  KL(mean/max)={s['kl_mean']:.2e}/{s['kl_max']:.2e}  argmax 일치={s['argmax_agree']:.3f}")
    s = summary["keydim_vs_biasimpl"]
    print(f"     keydim 구현 vs bias 구현:      max|Δlogit|={s['dlogit_vocab_max']:.3e}  KL(mean/max)={s['kl_mean']:.2e}/{s['kl_max']:.2e}  argmax 일치={s['argmax_agree']:.3f}")
    s = summary["bias_vs_del_renumber"]
    print(f"(2) bias vs 삭제(renumber):  max|Δlogit|={s['dlogit_vocab_max']:.3e}  KL(mean/max)={s['kl_mean']:.2e}/{s['kl_max']:.2e}  argmax 일치={s['argmax_agree']:.3f}")
    s = summary["full_vs_bias"]
    print(f"참고  full vs 무작위 마스크:  KL(mean/max)={s['kl_mean']:.2e}/{s['kl_max']:.2e}  argmax 일치={s['argmax_agree']:.3f}")
    nf = summarize([{"a_vs_b": r["noise_floor_full_vs_math"]} for r in results], "a", "b")
    print(f"잡음 바닥  full(default kernel) vs full(math kernel):  max|Δlogit|={nf['dlogit_vocab_max']:.3e}  KL(mean/max)={nf['kl_mean']:.2e}/{nf['kl_max']:.2e}  argmax 일치={nf['argmax_agree']:.3f}")
    print(f"판정 (kernel={args.kernel}): (1)의 KL·max|Δlogit| 이 잡음 바닥과 같은 자릿수이고 argmax 가 거의 다 일치하면 통과. "
          "kernel=math 이면 (1)은 이론상 정확히 0 에 가까워야 한다. (2)가 (1)보다 뚜렷이 크면 'position 유지' 관례를 기본으로 채택.")
    if timing and timing["sec_per_step"] is None:
        print(f"(3) fwd+bwd 측정 실패: {timing['note']}")
    if timing and timing["sec_per_step"] is not None:
        print(f"(3) fwd+bwd  {timing['sec_per_step']:.2f} s/step, peak mem {timing['peak_mem_gb']:.1f} GB, "
              f"grad|θ| = {timing['grad_norm']:.3e} (0 이면 grad 가 안 흐르는 것)  [{timing['note']}]")

    json.dump({"args": vars(args), "summary": summary, "timing": timing, "samples": results},
              open(out_path, "w"), indent=2)
    print(f"saved: {out_path}")


def run_timing(model, vi, full_logits, lid, steps, use_checkpoint=True, pad_multiple=0):
    """θ (n_vis,) → m=sigmoid(θ) → bias=log m → KL(p_full ‖ p_m) 의 fwd+bwd 비용."""
    note = "non-reentrant gradient checkpointing" if use_checkpoint else "no checkpointing"
    if use_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()  # HF 는 training 일 때만 checkpointing 을 적용 (Qwen2 dropout=0 이라 결과 동일)
    p_full = letter_dist(full_logits, lid).detach()
    theta = torch.zeros(vi.n_vis, device=model.device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=0.1)
    torch.cuda.reset_peak_memory_stats()
    times, grad_norm = [], 0.0
    try:
        for s in range(steps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            m = torch.sigmoid(theta)
            # clear=False: checkpointing 의 backward 재계산에서도 같은 bias 가 보여야 한다
            logits = last_logits(model, vi.embeds, bias=make_bias(vi, m), clear=False, pad_multiple=pad_multiple)
            p_m = letter_dist(logits, lid)
            loss = (p_full * (p_full.clamp_min(1e-12).log() - p_m.clamp_min(1e-12).log())).sum() + 0.1 * m.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clear_state()
            grad_norm = float(theta.grad.norm())
            opt.step()
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
            print(f"  [timing] step {s}: loss={float(loss):.4f} grad|θ|={grad_norm:.3e} {times[-1]:.2f}s", flush=True)
    except RuntimeError as e:
        note = f"FAILED: {type(e).__name__}: {str(e)[:200]} — --attn_impl eager 로 재시도"
        print(note, flush=True)
    finally:
        model.eval()
        clear_state()
        if use_checkpoint:
            model.gradient_checkpointing_disable()
    peak = torch.cuda.max_memory_allocated() / 1e9
    return {"sec_per_step": (sum(times[1:]) / max(1, len(times) - 1)) if len(times) > 1 else (times[0] if times else None),
            "peak_mem_gb": peak, "grad_norm": grad_norm, "steps": len(times), "note": note}


if __name__ == "__main__":
    main()
