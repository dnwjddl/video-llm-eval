#!/usr/bin/env python3
"""Gemma 4 저점수 원인 확정 — 플러그인과 동일한 경로로 실제 추론 1회 수행.

실행 (videollm 환경): python debug/gemma4_debug2.py
출력은 debug/gemma4_debug2_output.txt 에도 저장됩니다.

확인 항목:
  [A] 플러그인과 똑같이 만든 입력의 키/모양 + 프롬프트에 비디오 토큰이 있는지
  [B] 그 경로로 "비디오를 한 문장으로 묘사해라" → 답이 실제 영상 내용과 맞는지
  [C] 같은 경로로 4지선다 질문 → 답 형식이 어떤지
"""

import glob
import os
import sys

class _Tee:
    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s); self.file.write(s)
    def flush(self):
        self.stdout.flush(); self.file.flush()

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemma4_debug2_output.txt")
sys.stdout = _Tee(OUT); sys.stderr = sys.stdout

import torch
from transformers import AutoProcessor

try:
    from transformers import AutoModelForMultimodalLM as Cls
except ImportError:
    from transformers import AutoModelForImageTextToText as Cls

CKPT = "google/gemma-4-E2B-it"

# 테스트 비디오
video = None
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
for pat in [os.path.expanduser("~/videomme_videos/*.mp4"), os.path.join(hf_home, "**", "*.mp4")]:
    hits = glob.glob(pat, recursive=True)
    if hits:
        video = hits[0]; break
if video is None and len(sys.argv) > 1:
    video = sys.argv[1]
assert video, "mp4를 찾지 못했습니다. 인자로 경로를 주세요."
print(f"test video: {video}")
print(f"(영상 파일명을 보고 [B]의 묘사가 실제 내용과 맞는지 판단하세요)\n")

print("loading model (E2B)...")
model = Cls.from_pretrained(CKPT, torch_dtype=torch.bfloat16, device_map="cuda").eval()
processor = AutoProcessor.from_pretrained(CKPT)
tokenizer = processor.tokenizer

def plugin_style_infer(question, tag):
    """플러그인 generate_until과 동일한 메시지/템플릿/생성 경로."""
    message = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "video", "video": video, "max_pixels": 1605632, "min_pixels": 200704},
            {"type": "text", "text": question},
        ]},
    ]
    inputs = processor.apply_chat_template(
        [message], add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", padding=True, num_frames=32,
    )
    shapes = {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in inputs.items()}
    print(f"[{tag}] inputs: {shapes}")
    ids = inputs["input_ids"][0]
    decoded_tail = tokenizer.decode(ids[-60:])
    print(f"[{tag}] prompt 끝부분: ...{decoded_tail!r}")
    # 비디오/이미지 자리표시 토큰이 프롬프트에 실제로 몇 개 들어갔는지
    special_counts = {}
    for tok_str in ("<image_soft_token>", "<video>", "<image>", "<start_of_image>", "<start_of_video>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if tid is not None and tid >= 0:
                c = int((ids == tid).sum())
                if c:
                    special_counts[tok_str] = c
        except Exception:
            pass
    print(f"[{tag}] 비주얼 자리표시 토큰 개수: {special_counts or '발견 못함(토큰 이름이 다를 수 있음)'}")
    print(f"[{tag}] input_ids 길이: {len(ids)}")

    inputs = inputs.to(model.device)
    for k, v in inputs.items():
        if torch.is_tensor(v) and torch.is_floating_point(v):
            inputs[k] = v.to(torch.bfloat16)
    out = model.generate(**inputs, do_sample=False, max_new_tokens=64, use_cache=True)
    ans = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print(f"[{tag}] 모델 답변: {ans!r}\n")
    return ans

print("=" * 70)
plugin_style_infer("Describe this video in one sentence.", "B: 묘사")
plugin_style_infer(
    "What type of content is this video?\nA. A cooking tutorial\nB. A sports game\nC. A news broadcast\nD. An animation\nAnswer with the option's letter from the given choices directly.",
    "C: 4지선다",
)
print("완료 — bash scripts/send_debug.sh 로 공유하거나, [B] 답변이 실제 영상 내용과 맞는지 알려주세요.")
