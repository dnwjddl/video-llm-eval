#!/usr/bin/env python3
"""Caption-note 상한 실험 — Video-MME 150샘플판 (비디오 길이별).

MVBench판(run_mvbench_caption_note.py)과 동일한 5조건이되, 무대가 다르다:
MVBench(5~35초 클립)는 8프레임에 포화되어 full−reduced 격차가 없었다.
Video-MME long(30~60분)은 32프레임 자체가 결핍이라 notes가 일할 공간이 있다.

조건: full / reduced / notes / notes_qaware / deferral (정의는 MVBench판과 동일)
샘플: breakdown과 같은 seed 42의 150개 (short/medium/long 각 50)

실행 (llava 환경, GPU):
  python token_analysis/run_videomme_caption_note.py --keep 0.25 --video_dir ~/videomme_videos

결과: token_analysis/results_caption_note/videomme_keep<keep>.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "latency"))
sys.path.insert(0, os.path.join(BASE, "token_analysis"))
OUT_DIR = os.path.join(BASE, "token_analysis", "results_caption_note")

from run_mvbench_caption_note import LlavaRunner, extract_letter, frame_novelty  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=float, default=0.25)
    ap.add_argument("--video_dir", default=os.path.expanduser("~/videomme_videos"))
    ap.add_argument("--n_per_duration", type=int, default=250)
    ap.add_argument("--durations", default="short,medium,long",
                    help="GPU 분할용 — 예: GPU A는 --durations short,medium / GPU B는 --durations long")
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--n_notes", type=int, default=3)
    ap.add_argument("--note_mode", choices=["static", "uniform"], default="static",
                    help="static: novelty 하위(정적) 프레임 캡션 / uniform: 타임라인 균등 — 긴 비디오 커버리지 검증용 (--n_notes 8 권장)")
    ap.add_argument("--deferral_mode", choices=["strict", "loose"], default="strict",
                    help="strict: 확실할 때만 답 / loose: 관련 정보가 전혀 없을 때만 UNSURE (1차 응답률↑)")
    ap.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from profile_latency import index_videos, load_samples, read_frames

    samples = load_samples("videomme", args.n_per_duration, args.seed)
    want = [d.strip() for d in args.durations.split(",")]
    samples = {d: rows for d, rows in samples.items() if d in want}
    vindex = index_videos(args.video_dir)
    runner = LlavaRunner(args.pretrained)
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = "" if len(want) == 3 else "_" + "-".join(want)
    if args.note_mode != "static":
        tag += f"_notes-{args.note_mode}{args.n_notes}"
    if args.deferral_mode != "strict":
        tag += f"_def-{args.deferral_mode}"
    out_path = os.path.join(OUT_DIR, f"videomme_keep{args.keep}{tag}.json")

    conds = ["full", "reduced", "notes", "notes_qaware", "deferral"]
    k_frames = max(1, int(args.num_frames * args.keep))
    report = {"keep": args.keep, "n_notes": args.n_notes, "note_mode": args.note_mode,
              "deferral_mode": args.deferral_mode, "pretrained": args.pretrained,
              "per_duration": {}, "caption_examples": []}
    records = []

    STAGE1_STRICT = ('\nIf the notes clearly determine the answer, reply with only the option letter. '
                     'If you cannot determine it from the notes alone, reply exactly "UNSURE".')
    STAGE1_LOOSE = ('\nBased on the notes, reply with your best-guess option letter. '
                    'Reply exactly "UNSURE" only if the notes contain no information relevant to the question at all.')
    stage1_suffix = STAGE1_STRICT if args.deferral_mode == "strict" else STAGE1_LOOSE

    for dur, rows in samples.items():
        correct = {c: 0 for c in conds}
        total = escalated = 0
        t_caption = 0.0
        for i, row in enumerate(rows):
            path = vindex.get(row["videoID"])
            if path is None or row.get("answer") is None:
                continue
            try:
                frames = read_frames(path, args.num_frames)
                nov = frame_novelty(frames)
            except Exception as e:
                print(f"[err] {dur} #{i}: {type(e).__name__}: {e}", flush=True)
                continue

            q = row["prompt"]
            gt = str(row["answer"]).strip().strip("()").upper()[:1]

            keep_idx = np.sort(np.argsort(-nov)[:k_frames])
            if args.note_mode == "uniform":
                note_idx = sorted(set(np.linspace(0, len(frames) - 1, args.n_notes).astype(int).tolist()))
            else:
                static_order = np.argsort(nov)
                note_idx = sorted(static_order[: max(args.n_notes * 3, args.n_notes)][::3][: args.n_notes])

            preds = {}
            preds["full"] = runner.generate(frames, q)
            reduced = frames[keep_idx]
            preds["reduced"] = runner.generate(reduced, q)

            t0 = time.perf_counter()
            caps, caps_qa = [], []
            for ni in note_idx:
                one = frames[ni:ni + 1]
                caps.append(runner.generate(one, "Describe this frame in one detailed sentence.", max_new_tokens=48))
                caps_qa.append(runner.generate(
                    one, f"Describe this frame in one sentence, focusing on details relevant to: {row['prompt'].splitlines()[0]}",
                    max_new_tokens=48))
            t_caption += time.perf_counter() - t0

            notes_txt = "Scene notes from the full video:\n" + "\n".join(f"- {c}" for c in caps)
            preds["notes"] = runner.generate(reduced, notes_txt + "\n\n" + q)
            notes_txt_qa = "Scene notes from the full video:\n" + "\n".join(f"- {c}" for c in caps_qa)
            preds["notes_qaware"] = runner.generate(reduced, notes_txt_qa + "\n\n" + q)

            stage1 = runner.generate_text(notes_txt + "\n\n" + q + stage1_suffix)
            if "UNSURE" in stage1.upper() or extract_letter(stage1) is None:
                preds["deferral"] = preds["notes"]
                escalated += 1
            else:
                preds["deferral"] = stage1

            total += 1
            rec = {c: int(extract_letter(preds[c]) == gt) for c in conds}
            rec["_dur"] = dur
            records.append(rec)
            # 환각 검증용: duration별 처음 2개 샘플의 캡션 원문 저장
            if total <= 2:
                report["caption_examples"].append({
                    "duration": dur, "videoID": row["videoID"], "gt": gt,
                    "question_head": row["prompt"].splitlines()[0][:150],
                    "caps": caps, "caps_qaware": caps_qa,
                    "pred_notes": preds["notes"][:80], "pred_qaware": preds["notes_qaware"][:80]})
            for c in conds:
                correct[c] += rec[c]
            if (i + 1) % 10 == 0:
                accs = " ".join(f"{c}={correct[c] / total:.2f}" for c in conds)
                print(f"{dur} {i + 1}/{len(rows)}: {accs} (esc={escalated / total:.2f})", flush=True)

        if total:
            report["per_duration"][dur] = {"n": total,
                                           "caption_time_per_sample_s": round(t_caption / total, 2),
                                           "deferral_escalation_rate": round(escalated / total, 4),
                                           **{c: round(correct[c] / total, 4) for c in conds}}
            print(f"[{dur}] {report['per_duration'][dur]}", flush=True)
        json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)

    if records:
        from math import comb

        def mcnemar(a, b, recs):
            n01 = sum(1 for r in recs if not r[a] and r[b])
            n10 = sum(1 for r in recs if r[a] and not r[b])
            n = n01 + n10
            p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(n01, n10) + 1)) / (2 ** n))
            return n01, n10, p

        print("\n===== 쌍대 검정 (McNemar exact) =====")
        report["mcnemar"] = {}
        pairs = [("full", "reduced"), ("reduced", "notes"), ("notes", "notes_qaware"),
                 ("full", "notes_qaware"), ("notes", "deferral")]
        for scope, recs in [("all", records)] + [(d, [r for r in records if r["_dur"] == d]) for d in ("short", "medium", "long")]:
            if not recs:
                continue
            for a, b in pairs:
                n01, n10, p = mcnemar(a, b, recs)
                mark = "유의" if p < 0.05 else "ns"
                print(f"[{scope:6s}] {a:>13s} → {b:13s}: +{n01}/-{n10}, p={p:.4f} ({mark})")
                report["mcnemar"][f"{scope}:{a}->{b}"] = {"saved": n01, "broke": n10, "p": round(p, 5)}
        report["records"] = records   # 여러 실행 합산(pooled McNemar)용 문항별 정오
        report["complete"] = True
        json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
