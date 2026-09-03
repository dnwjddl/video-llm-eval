"""Build unified item tables (one parquet per benchmark).

Examples
  python -m encoder_study.blind.build_items --inspect tomato
  python -m encoder_study.blind.build_items --benchmarks mvbench,tvbench,tomato,vsibench,perceptiontest,motionbench --out items/
  python -m encoder_study.blind.build_items --benchmarks star --star-json /data/STAR/STAR_val.json --video-root /data/Charades_v1_480 --out items/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from .loaders import INSPECT, LOADERS
from .schema import items_to_frame


def inspect(name: str, n: int = 1):
    from datasets import load_dataset

    path, cfg, split = INSPECT[name]
    ds = load_dataset(path, cfg, split=split) if cfg else load_dataset(path, split=split)
    print(f"== {name}: {path} cfg={cfg} split={split} rows={len(ds)}")
    print("columns:", ds.column_names)
    for i in range(min(n, len(ds))):
        row = {k: (v if not isinstance(v, (bytes, bytearray)) else f"<{len(v)} bytes>") for k, v in ds[i].items()}
        print(json.dumps(row, ensure_ascii=False, indent=1, default=str)[:3000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="", help="comma list; 'all' = every HF-hosted loader")
    ap.add_argument("--out", default="items")
    ap.add_argument("--inspect", default="", help="print raw columns/rows of one HF benchmark and exit")
    ap.add_argument("--star-json", default="")
    ap.add_argument("--clevrer-json", default="")
    ap.add_argument("--video-root", default="", help="video dir for star / clevrer / mme_videoocr")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.inspect)
        return

    names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    if names == ["all"]:
        names = ["mvbench", "tvbench", "tomato", "vsibench", "perceptiontest", "motionbench", "mme_videoocr"]
    os.makedirs(args.out, exist_ok=True)
    summary = []
    for name in names:
        try:
            items = LOADERS[name](star_json=args.star_json, clevrer_json=args.clevrer_json, video_root=args.video_root)
        except Exception:
            print(f"[FAIL] {name}\n{traceback.format_exc()}", file=sys.stderr)
            continue
        df = items_to_frame(items)
        dup = df["item_id"].duplicated().sum()
        if dup:
            print(f"[WARN] {name}: {dup} duplicate item_ids -> suffixing with row index", file=sys.stderr)
            df["item_id"] = [f"{i}#{k}" if d else i for k, (i, d) in enumerate(zip(df["item_id"], df["item_id"].duplicated(keep=False)))]
        path = os.path.join(args.out, f"{name}.parquet")
        df.to_parquet(path, index=False)
        n_video = int((df["video_path"] != "").sum()) if len(df) else 0
        summary.append((name, len(df), n_video, df["category"].nunique() if len(df) else 0))
        print(f"[OK] {name}: {len(df)} items, {n_video} with resolved video, {summary[-1][3]} categories -> {path}")
    print("\nbenchmark\titems\tvideos_resolved\tcategories")
    for s in summary:
        print("\t".join(map(str, s)))


if __name__ == "__main__":
    main()
