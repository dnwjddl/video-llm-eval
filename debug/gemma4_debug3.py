#!/usr/bin/env python3
"""Gemma 4 최종 진단 — 실제 MVBench 평가에 쓰인 질문/비디오 5개를 그대로 재실행.

실행 (videollm 환경): python debug/gemma4_debug3.py
출력은 debug/gemma4_debug3_output.txt 에도 저장됩니다.

E2B 평가 당시의 샘플 로그에서 실제 문항 5개를 꺼내:
  - 평가 당시 모델이 낸 답 vs 정답
  - 로그에 기록된 비디오 경로가 지금 실제로 존재하는지 (플러그인의 존재 체크 통과 여부)
  - 같은 질문+비디오를 지금 다시 추론하면 뭐라고 답하는지
를 비교합니다.
"""

import glob
import json
import os
import sys


class _Tee:
    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma4_debug3_output.txt")
sys.stdout = _Tee(OUT)
sys.stderr = sys.stdout

# 사용법: python debug/gemma4_debug3.py [서브태스크명]  (기본: action_sequence)
SUB = sys.argv[1] if len(sys.argv) > 1 else "action_sequence"
files = sorted(glob.glob(f"logs/gemma-4-E2B-it/**/*mvbench_{SUB}*.jsonl", recursive=True)) or \
    sorted(glob.glob("logs/gemma-4-E2B-it/**/*mvbench*.jsonl", recursive=True))
assert files, "E2B mvbench 샘플 로그가 없습니다 (logs/gemma-4-E2B-it/)"
f = files[-1]
print(f"samples log: {f}\n")

rows = [json.loads(l) for l in list(open(f))[:5]]

# 로그 구조 파악
r0 = rows[0]
print(f"row keys: {list(r0.keys())}")
doc = r0.get("doc", {})
print(f"doc keys: {list(doc.keys())}\n")

# 비디오 경로 후보 필드
def find_video_ref(doc):
    for k in ("video", "video_path", "videoID", "video_name", "data"):
        if k in doc and isinstance(doc[k], str):
            return k, doc[k]
    return None, None

# HF_HOME 아래에서 mvbench 비디오 실경로 인덱스
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
index = {}
for root, _, fs in os.walk(hf_home):
    if "mvbench" not in root.lower():
        continue
    for name in fs:
        if name.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            index[name] = os.path.join(root, name)
print(f"HF_HOME 내 mvbench 비디오 파일: {len(index)}개\n")

import torch
from transformers import AutoProcessor

try:
    from transformers import AutoModelForMultimodalLM as Cls
except ImportError:
    from transformers import AutoModelForImageTextToText as Cls

print("loading E2B...")
model = Cls.from_pretrained("google/gemma-4-E2B-it", torch_dtype=torch.bfloat16, device_map="cuda").eval()
processor = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")

def infer(video_path, question):
    message = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "max_pixels": 1605632, "min_pixels": 200704},
            {"type": "text", "text": question},
        ]},
    ]
    kwargs = {"num_frames": 32}
    import re as _re
    while True:
        try:
            inputs = processor.apply_chat_template(
                [message], add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True, **kwargs)
            break
        except ValueError as e:
            m = _re.search(r"total_num_frames=(\d+)", str(e))
            if m and int(m.group(1)) < kwargs["num_frames"]:
                kwargs["num_frames"] = int(m.group(1))
                continue
            raise
    inputs = inputs.to(model.device)
    for k, v in inputs.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs[k] = v.to(torch.bfloat16)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=32)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

print("=" * 70)
for i, row in enumerate(rows):
    doc = row.get("doc", {})
    args = row.get("arguments")
    # arguments[0][0]이 보통 평가 당시 프롬프트 텍스트
    prompt = None
    try:
        a = args
        while isinstance(a, (list, tuple)) and a:
            if isinstance(a[0], str):
                prompt = a[0]
                break
            a = a[0]
    except Exception:
        pass
    if prompt is None:
        prompt = str(doc.get("question", ""))[:500]
    key, ref = find_video_ref(doc)
    base = os.path.basename(ref) if ref else None
    actual = index.get(base) if base else None
    print(f"\n--- 샘플 {i} ---")
    print(f"video 필드({key}): {ref}")
    print(f"실제 파일: {'존재 → ' + actual if actual else '★ 못 찾음 (평가 때 비디오 없이 돌았을 가능성)'}")
    print(f"평가 당시 답: {row.get('filtered_resps') or row.get('resps')} | 정답: {row.get('target')}")
    print(f"프롬프트 앞 200자: {prompt[:200]!r}")
    if actual:
        try:
            ans = infer(actual, prompt)
            print(f"지금 다시 추론한 답: {ans!r}")
        except Exception as e:
            print(f"재추론 에러: {type(e).__name__}: {e}")

print("\n완료 — bash scripts/send_debug.sh 로 공유하거나 결과를 읽어주세요.")
