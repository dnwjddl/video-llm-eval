#!/usr/bin/env python3
"""caption-note 결과 JSON들을 전부 모아 한 표로 — baseline과 변형 판을 나란히.

실행: python token_analysis/summarize_caption_note.py
      (results_caption_note/ 의 모든 videomme_*.json / keep*.json을 자동 수집)

행 = (파일 태그, 조건), 열 = short/medium/long. esc는 deferral 행에 병기.
같은 조건이 여러 파일에 있으면 각각 다른 행으로 표시 (태그로 구분).
"""

import glob
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DURS = ["short", "medium", "long"]
COND_ORDER = ["full", "reduced", "notes", "notes_qaware", "deferral"]


def tag_of(fname):
    t = os.path.basename(fname).replace(".json", "")
    for p in ("videomme_keep", "keep"):
        if t.startswith(p):
            t = t[len(p):]
    return t


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(BASE, "results_caption_note", "*.json"))):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        per = r.get("per_duration") or r.get("subtasks")
        if not per:
            continue
        tag = tag_of(f)
        # per_duration 형식(videomme)만 duration 열로; mvbench(subtasks)는 overall만
        is_dur = any(d in per for d in DURS)
        conds = [c for c in COND_ORDER if any(c in v for v in per.values())]
        for c in conds:
            row = {"tag": tag, "cond": c}
            if is_dur:
                for d in DURS:
                    if d in per and c in per[d]:
                        row[d] = per[d][c]
                        if c == "deferral":
                            row[f"{d}_esc"] = per[d].get("deferral_escalation_rate")
            else:
                vals = [v[c] for v in per.values() if c in v]
                row["overall"] = sum(vals) / len(vals) if vals else None
            rows.append(row)

    if not rows:
        print("results_caption_note/ 에 결과가 없습니다.")
        return

    print(f"{'파일 태그':28s} {'조건':13s} " + "".join(f"{d:>16s}" for d in DURS) + f"{'overall':>10s}")
    print("-" * 100)
    for row in rows:
        cells = []
        for d in DURS:
            v = row.get(d)
            if v is None:
                cells.append(f"{'—':>16s}")
            elif row["cond"] == "deferral" and row.get(f"{d}_esc") is not None:
                cells.append(f"{v:.3f}(e{row[f'{d}_esc']:.2f})".rjust(16))
            else:
                cells.append(f"{v:.4f}".rjust(16))
        ov = row.get("overall")
        print(f"{row['tag']:28s} {row['cond']:13s} " + "".join(cells) + (f"{ov:>10.4f}" if ov is not None else f"{'':>10s}"))


if __name__ == "__main__":
    main()
