#!/usr/bin/env python3
"""프로파일러가 사용할 샘플 비디오만 HF 캐시의 zip에서 선택적으로 압축 해제.

profile_latency.py와 같은 --dataset/--n_per_duration/--seed를 주면 같은 샘플이
선택되므로, 전체(~100GB)를 풀지 않고 필요한 150개(기본)만 풉니다.

사용 예:
  python latency/extract_videos_subset.py --dataset videomme \
      --out_dir ~/videomme_videos --n_per_duration 50 --seed 42
"""

import argparse
import os
import zipfile

from profile_latency import load_samples

REPOS = {"videomme": "lmms-eval/Video-MME", "videomme_v2": "MME-Benchmarks/Video-MME-v2"}
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="videomme", choices=list(REPOS))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_per_duration", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = load_samples(args.dataset, args.n_per_duration, args.seed)
    wanted = {row["videoID"] for rows in samples.values() for row in rows}
    print(f"\n필요한 비디오: {len(wanted)}개")

    from huggingface_hub import snapshot_download

    snap = snapshot_download(REPOS[args.dataset], repo_type="dataset")
    print(f"snapshot: {snap}")

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    zips = []
    for root, _, files in os.walk(snap):
        zips += [os.path.join(root, f) for f in files if f.endswith(".zip")]
    print(f"zip 파일 {len(zips)}개 스캔 중...")

    found = set()
    for zp in zips:
        try:
            with zipfile.ZipFile(zp) as zf:
                for member in zf.namelist():
                    base = os.path.basename(member)
                    stem, ext = os.path.splitext(base)
                    if ext.lower() in VIDEO_EXTS and stem in wanted and stem not in found:
                        dst = os.path.join(out_dir, base)
                        if not os.path.exists(dst):
                            with zf.open(member) as src, open(dst, "wb") as f:
                                while True:
                                    chunk = src.read(1 << 20)
                                    if not chunk:
                                        break
                                    f.write(chunk)
                        found.add(stem)
        except zipfile.BadZipFile:
            print(f"[warn] 손상된 zip 건너뜀: {zp}")
        if len(found) == len(wanted):
            break

    missing = wanted - found
    print(f"\n압축 해제 완료: {len(found)}/{len(wanted)} → {out_dir}")
    if missing:
        print(f"[warn] 못 찾은 비디오 {len(missing)}개 (프로파일러가 자동으로 건너뜁니다): {sorted(missing)[:10]}...")


if __name__ == "__main__":
    main()
