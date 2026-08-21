#!/usr/bin/env python3
"""노트 내용 유형 × 질문 유형 매트릭스 — "무엇을 텍스트로 들 것인가"의 근거표.

results_caption_note/의 각 실행(파일)에서 records를 읽어,
질문 유형(Video-MME task_type)별로 notes 정확도를 교차 집계한다.
같은 seed·같은 문항이므로 파일 간(=노트 유형 간) 비교가 쌍대로 성립한다.

실행: python token_analysis/note_type_matrix.py
"""

import glob
import json
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))


def tag_of(f):
    t = os.path.basename(f).replace(".json", "")
    for p in ("videomme_keep0.25", "videomme_keep", "keep"):
        if t.startswith(p):
            t = t[len(p):]
    t = t.strip("_") or "detail(base)"
    return t.replace("_fast", "").replace("long_", "") or "detail(base)"


def main():
    runs = {}
    for f in sorted(glob.glob(os.path.join(BASE, "results_caption_note", "*.json"))):
        try:
            recs = json.load(open(f)).get("records", [])
        except Exception:
            continue
        recs = [r for r in recs if r.get("_qtype") and "notes" in r]
        if recs:
            runs[tag_of(f)] = recs
    if not runs:
        print("records(+_qtype)가 있는 결과가 없습니다 — qtype 저장 이후 실행분만 집계됩니다.")
        return

    qtypes = sorted({r["_qtype"] for recs in runs.values() for r in recs},
                    key=lambda q: -sum(1 for recs in runs.values() for r in recs if r["_qtype"] == q))
    tags = list(runs)

    # notes 정확도 매트릭스
    print("=== notes 정확도: 노트 유형(행) × 질문 유형(열) ===")
    head = f"{'노트 유형':22s}" + "".join(f"{q[:16]:>18s}" for q in qtypes) + f"{'전체':>10s}"
    print(head)
    for tag in tags:
        recs = runs[tag]
        cells = []
        for q in qtypes:
            sub = [r["notes"] for r in recs if r["_qtype"] == q]
            cells.append(f"{sum(sub)/len(sub):.3f}(n{len(sub)})".rjust(18) if sub else f"{'—':>18s}")
        allv = [r["notes"] for r in recs]
        print(f"{tag:22s}" + "".join(cells) + f"{sum(allv)/len(allv):>10.3f}")

    # reduced 기준선이 있는 파일에서 델타도
    base = next((recs for tag, recs in runs.items() if all("reduced" in r for r in recs)), None)
    if base:
        print("\n=== notes − reduced (같은 문항 기준, %p) ===")
        print(head)
        by_q_base = defaultdict(list)
        for r in base:
            by_q_base[r["_qtype"]].append(r["reduced"])
        for tag in tags:
            recs = runs[tag]
            cells = []
            for q in qtypes:
                sub = [r["notes"] for r in recs if r["_qtype"] == q]
                b = by_q_base.get(q)
                if sub and b:
                    cells.append(f"{(sum(sub)/len(sub) - sum(b)/len(b))*100:+.1f}".rjust(18))
                else:
                    cells.append(f"{'—':>18s}")
            allv = [r["notes"] for r in recs]
            allb = [v for vs in by_q_base.values() for v in vs]
            print(f"{tag:22s}" + "".join(cells) + f"{(sum(allv)/len(allv) - sum(allb)/len(allb))*100:>+10.1f}")


if __name__ == "__main__":
    main()
