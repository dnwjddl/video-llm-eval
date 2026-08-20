#!/usr/bin/env python3
"""Caption-note 상한 실험 (MVBench) — "textualize 축이 가치 있는가"의 upper bound.

한 번의 실행에서 샘플마다 3~4개 조건을 같은 모델로 평가:
  full          32프레임 전체 (상한 앵커, 비주얼 6272토큰)
  reduced       동적 프레임 K개만 (같은 비주얼 예산의 노트 없는 기준)
  notes         reduced + 정적 대표 프레임의 캡션을 텍스트 노트로 주입
  notes_qaware  캡션을 질문 조건부로 생성 (진짜 상한 — 배포 불가, Scribe re-fetch의 목표점)

읽는 법: notes가 reduced보다 높으면 textualize 축이 실재. full과의 격차를 얼마나
복구하는지가 효과 크기. notes_qaware는 그 축의 이론적 천장.

실행 (llava 환경, GPU):
  python token_analysis/run_mvbench_caption_note.py --keep 0.25 --n_per_subtask 50

결과: token_analysis/results_caption_note/keep<keep>.json (서브태스크×조건별 정확도)
"""

import argparse
import json
import os
import re
import string
import sys
import time

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "token_analysis", "results_caption_note")

# temporal 4 + spatial 2 (episodic_reasoning은 프레임 폴더 형식이라 제외)
DEFAULT_SUBTASKS = "action_sequence,moving_direction,moving_count,scene_transition,object_existence,fine_grained_pose"


def index_hf_videos():
    """HF 캐시에서 mvbench 비디오 파일 인덱스 (basename → 경로)."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    index = {}
    for root, _, files in os.walk(hf_home):
        if "mvbench" not in root.lower():
            continue
        for name in files:
            if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                index.setdefault(name, os.path.join(root, name))
    return index


def read_frames(path, num_frames):
    from decord import VideoReader, cpu

    vr = VideoReader(path, ctx=cpu(0), num_threads=1)
    idx = np.linspace(0, len(vr) - 1, min(num_frames, len(vr))).astype(int)
    return vr.get_batch(idx).asnumpy()


def frame_novelty(frames):
    """픽셀 기반 프레임별 변화량 (첫 프레임은 최대) — 외부 의존성 없이 numpy로."""
    gray = frames.astype(np.float32).mean(axis=-1)          # (T, H, W)
    sh, sw = max(1, gray.shape[1] // 64), max(1, gray.shape[2] // 64)
    small = gray[:, ::sh, ::sw]
    nov = np.ones(len(frames)) * 255.0
    nov[1:] = np.abs(small[1:] - small[:-1]).mean(axis=(1, 2))
    return nov


def extract_letter(pred):
    m = re.match(r"^\s*\(?([A-E])[).:\s]", pred + " ")
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-E])\b", pred)
    return m.group(1).upper() if m else None


class LlavaRunner:
    def __init__(self, pretrained):
        from llava.model.builder import load_pretrained_model

        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            pretrained, None, "llava_qwen", device_map="cuda",
            torch_dtype="bfloat16", attn_implementation="sdpa",
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, frames, text, max_new_tokens=24):
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        video = self.image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
        video = video.to(dtype=torch.bfloat16, device=self.model.device)
        out = self.model.generate(ids.unsqueeze(0).to(self.model.device), images=[video],
                                  modalities=["video"], do_sample=False, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtasks", default=DEFAULT_SUBTASKS)
    ap.add_argument("--keep", type=float, default=0.25, help="reduced 조건의 프레임 비율 (0.25 → 8/32프레임)")
    ap.add_argument("--n_per_subtask", type=int, default=50)
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--n_notes", type=int, default=3, help="캡션 노트 개수 (정적 대표 프레임 수)")
    ap.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_qaware", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset

    vindex = index_hf_videos()
    print(f"HF 캐시 내 mvbench 비디오: {len(vindex)}개", flush=True)
    runner = LlavaRunner(args.pretrained)
    letters = string.ascii_uppercase
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"keep{args.keep}.json")

    conds = ["full", "reduced", "notes"] + ([] if args.skip_qaware else ["notes_qaware"])
    report = {"keep": args.keep, "n_notes": args.n_notes, "pretrained": args.pretrained, "subtasks": {}}
    k_frames = max(1, int(args.num_frames * args.keep))
    records = []  # 문항별 조건별 정오 — 쌍대(McNemar) 검정용

    for sub in args.subtasks.split(","):
        ds = load_dataset("OpenGVLab/MVBench", sub, split="train")
        rng = np.random.RandomState(args.seed)
        picks = rng.permutation(len(ds))[:args.n_per_subtask]
        correct = {c: 0 for c in conds}
        total = 0
        t_caption = 0.0
        for pi, di in enumerate(picks):
            doc = ds[int(di)]
            vfile = os.path.basename(str(doc.get("video", "")))
            path = vindex.get(vfile)
            if not path:
                continue
            try:
                frames = read_frames(path, args.num_frames)
                nov = frame_novelty(frames)
            except Exception as e:
                print(f"[err] {sub} #{pi}: {type(e).__name__}: {e}", flush=True)
                continue

            cands = doc["candidates"]
            gt = letters[cands.index(doc["answer"])]
            opts = "\n".join(f"({letters[i]}) {c}" for i, c in enumerate(cands))
            q = f"Question: {doc['question']}\nOption:\n{opts}\nOnly give the best option letter."

            # 프레임 배분: retained = novelty 상위 k (시간순), 노트용 정적 대표 = novelty 하위에서 시간 분산되게
            keep_idx = np.sort(np.argsort(-nov)[:k_frames])
            static_order = np.argsort(nov)
            note_idx = sorted(static_order[: max(args.n_notes * 3, args.n_notes)][:: 3][: args.n_notes])

            preds = {}
            preds["full"] = runner.generate(frames, q)
            reduced = frames[keep_idx]
            preds["reduced"] = runner.generate(reduced, q)

            t0 = time.perf_counter()
            caps, caps_qa = [], []
            for ni in note_idx:
                one = frames[ni:ni + 1]
                caps.append(runner.generate(one, "Describe this frame in one detailed sentence.", max_new_tokens=48))
                if "notes_qaware" in conds:
                    caps_qa.append(runner.generate(
                        one, f"Describe this frame in one sentence, focusing on details relevant to: {doc['question']}",
                        max_new_tokens=48))
            t_caption += time.perf_counter() - t0

            notes_txt = "Scene notes from the full video:\n" + "\n".join(f"- {c}" for c in caps)
            preds["notes"] = runner.generate(reduced, notes_txt + "\n\n" + q)
            if "notes_qaware" in conds:
                notes_txt_qa = "Scene notes from the full video:\n" + "\n".join(f"- {c}" for c in caps_qa)
                preds["notes_qaware"] = runner.generate(reduced, notes_txt_qa + "\n\n" + q)

            total += 1
            rec = {c: int(extract_letter(preds[c]) == gt) for c in conds}
            records.append(rec)
            for c in conds:
                correct[c] += rec[c]
            if (pi + 1) % 10 == 0:
                accs = " ".join(f"{c}={correct[c] / total:.2f}" for c in conds)
                print(f"{sub} {pi + 1}/{len(picks)}: {accs}", flush=True)

        if total:
            report["subtasks"][sub] = {"n": total, "caption_time_per_sample_s": round(t_caption / total, 2),
                                       **{c: round(correct[c] / total, 4) for c in conds}}
            print(f"[{sub}] {report['subtasks'][sub]}", flush=True)
        json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)

    # 전체 요약 + 쌍대(McNemar exact) 검정
    if report["subtasks"]:
        print("\n===== 요약 (전 서브태스크 평균) =====")
        for c in conds:
            vals = [v[c] for v in report["subtasks"].values()]
            print(f"{c:14s} {sum(vals) / len(vals):.4f}")

        from math import comb

        def mcnemar(a, b):
            """조건 a→b 쌍대 비교: (b만 정답, a만 정답, 양측 exact p)."""
            n01 = sum(1 for r in records if not r[a] and r[b])   # b가 살린 문항
            n10 = sum(1 for r in records if r[a] and not r[b])   # b가 망친 문항
            n = n01 + n10
            if n == 0:
                return n01, n10, 1.0
            p = 2 * sum(comb(n, k) for k in range(0, min(n01, n10) + 1)) / (2 ** n)
            return n01, n10, min(1.0, p)

        print("\n===== 쌍대 검정 (McNemar exact, n=%d) =====" % len(records))
        pairs = [("reduced", "notes"), ("reduced", "notes_qaware"), ("notes", "notes_qaware"), ("full", "notes_qaware")]
        report["mcnemar"] = {}
        for a, b in pairs:
            if a not in conds or b not in conds:
                continue
            n01, n10, p = mcnemar(a, b)
            verdict = "유의" if p < 0.05 else "유의하지 않음"
            print(f"{a:>13s} → {b:13s}: +{n01}문항 살림 / -{n10}문항 망침, p={p:.4f} ({verdict})")
            report["mcnemar"][f"{a}->{b}"] = {"saved": n01, "broke": n10, "p": round(p, 5)}

        report["overall"] = {c: round(sum(v[c] for v in report["subtasks"].values()) / len(report["subtasks"]), 4) for c in conds}
        report["complete"] = True
        json.dump(report, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
