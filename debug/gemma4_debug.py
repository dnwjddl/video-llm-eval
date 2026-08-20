#!/usr/bin/env python3
"""Gemma 4 MVBench 저점수(무작위 수준) 원인 진단.

실행 (videollm 환경, video-llm-eval 폴더에서):
    python debug/gemma4_debug.py

하는 일:
  [1] E2B의 mvbench 샘플 로그에서 모델이 실제로 뭐라고 답했는지 출력
  [2] Gemma 4 프로세서가 비디오 파일을 실제로 픽셀 텐서로 로딩하는지 확인
"""

import glob
import json
import os
import sys

print("=" * 70)
print("[1] 모델 응답 샘플 확인")
print("=" * 70)
files = sorted(glob.glob("logs/gemma-4-E2B-it/**/*mvbench*.jsonl", recursive=True))
if not files:
    print("샘플 로그를 찾지 못했습니다: logs/gemma-4-E2B-it/**/*mvbench*.jsonl")
else:
    f = files[-1]
    print(f"file: {f}\n")
    for line in list(open(f))[:8]:
        row = json.loads(line)
        resp = row.get("filtered_resps") or row.get("resps")
        print(f"  RESP: {str(resp)[:100]!s:100s} | 정답: {row.get('target')}")

print()
print("=" * 70)
print("[2] 프로세서 비디오 로딩 확인")
print("=" * 70)

# 테스트용 비디오 하나 찾기 (mvbench 캐시 → videomme 추출본 순으로)
video = None
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
for pattern in [
    os.path.join(hf_home, "**", "*.mp4"),
    os.path.expanduser("~/videomme_videos/*.mp4"),
]:
    hits = glob.glob(pattern, recursive=True)
    if hits:
        video = hits[0]
        break
if video is None:
    print("테스트용 mp4를 찾지 못했습니다. 아무 mp4 경로를 인자로 주세요:")
    print("  python debug/gemma4_debug.py /path/to/video.mp4")
    if len(sys.argv) > 1:
        video = sys.argv[1]
    else:
        sys.exit(1)
print(f"test video: {video}")

from transformers import AutoProcessor

p = AutoProcessor.from_pretrained("google/gemma-4-E2B-it")
msgs = [{"role": "user", "content": [
    {"type": "video", "video": video},
    {"type": "text", "text": "Describe this video."},
]}]
try:
    out = p.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", num_frames=8,
    )
    shapes = {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in out.items()}
    print(f"\napply_chat_template 출력 키: {shapes}")
    visual_keys = [k for k in shapes if "pixel" in k or "video" in k or "image" in k]
    if visual_keys:
        print(f"\n→ 비주얼 텐서 존재 ({visual_keys}): 비디오가 로딩되고 있음. 원인은 다른 곳.")
    else:
        print("\n→ ★ 비주얼 텐서 없음: 채팅 템플릿이 video 항목을 무시하고 있음 — 저점수의 원인!")
except Exception as e:
    print(f"\napply_chat_template 에러: {type(e).__name__}: {e}")
    print("→ 이 에러 메시지가 원인 단서입니다.")

# 참고: 프로세서가 별도 videos= 인자를 받는지도 확인
try:
    import inspect
    sig = inspect.signature(p.__call__)
    print(f"\nprocessor.__call__ 파라미터: {list(sig.parameters)}")
except Exception:
    pass
