#!/usr/bin/env python3
"""토큰 압축 ablation — LLaVA-OneVision의 비디오 토큰 pooling 지점을 가로채
compress.py의 방법을 적용한 뒤, lmms-eval 파이프라인 그대로 MVBench를 평가.

실행 (llava 환경):
  python token_analysis/run_token_ablation.py --method pca_select --keep 0.25 \
      --tasks mvbench_action_sequence,mvbench_object_existence

  --method: none|random|pool_avg|pool_max|temporal_pool|framediff|pca_select|pca_recon|tome|kmeans
  --keep:   유지 비율 (예: 1.0, 0.5, 0.25, 0.125, 0.05)
  --tasks:  mvbench 서브태스크들 (기본: 대표 8개 — README 참고)

결과: token_analysis/results/<method>_keep<keep>/ 아래 lmms-eval 결과 JSON.
원리: llava의 get_2dPool(프레임당 729토큰 → 풀링) 호출을 통째로 대체하므로,
      모든 방법이 같은 입력(729토큰/프레임)에서 출발해 공정 비교가 된다.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 대표 서브태스크: temporal 4 + spatial/static 4 (전체 스윕 비용 절감용)
DEFAULT_TASKS = ",".join([
    "mvbench_action_sequence", "mvbench_moving_direction",
    "mvbench_object_shuffle", "mvbench_scene_transition",          # temporal
    "mvbench_object_existence", "mvbench_object_interaction",
    "mvbench_fine_grained_pose", "mvbench_unexpected_action",      # spatial/static
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--keep", type=float, required=True)
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--pretrained", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    ap.add_argument("--max_frames", type=int, default=32)
    args = ap.parse_args()

    from compress import METHODS, compress

    assert args.method in METHODS, f"지원 방법: {list(METHODS)}"

    # --- llava의 비디오 풀링 지점을 가로채기 ---
    import llava.model.llava_arch as llava_arch

    orig = llava_arch.LlavaMetaForCausalLM.get_2dPool
    method, keep = args.method, args.keep

    def patched(self, image_feature, stride=2):
        if method == "none":
            return orig(self, image_feature, stride)
        return compress(image_feature, method, keep)

    llava_arch.LlavaMetaForCausalLM.get_2dPool = patched
    print(f"[token_ablation] get_2dPool patched: method={method}, keep={keep}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"{method}_keep{keep}")
    os.makedirs(out_dir, exist_ok=True)

    # --- lmms-eval CLI를 in-process로 호출 (데이터/채점 파이프라인 재사용) ---
    sys.argv = [
        "lmms_eval",
        "--model", "llava_onevision",
        "--model_args", f"pretrained={args.pretrained},conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num={args.max_frames},attn_implementation=sdpa",
        "--tasks", args.tasks,
        "--batch_size", "1",
        "--log_samples",
        "--output_path", out_dir,
    ]
    from lmms_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
