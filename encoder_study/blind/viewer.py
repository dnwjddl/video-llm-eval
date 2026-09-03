"""Static HTML viewer: inspect excluded (blind-solvable) and kept items with their videos.

  python -m encoder_study.blind.viewer --items items/*.parquet --flags report/flags.parquet \
      --preds preds/*.parquet --out-dir viewer --n-excluded 30 --n-kept 10

Videos are symlinked (or copied with --copy) into <out-dir>/videos so the page
works when the directory is served or opened locally:  cd viewer && python -m http.server 8000
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import random
import shutil

import pandas as pd

from .schema import LETTERS, frame_to_items

TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>Blind filter viewer</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f5f7;color:#1c2330}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #d9dde3;padding:10px 16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
select,button{font:inherit;padding:4px 8px}
.card{background:#fff;margin:14px 16px;border:1px solid #d9dde3;border-radius:6px;padding:14px;display:grid;grid-template-columns:360px 1fr;gap:16px}
.card.hidden{display:none}
video,img{max-width:360px;width:100%;background:#000;border-radius:4px}
.frames{display:grid;grid-template-columns:repeat(4,1fr);gap:4px}
.frames img{width:100%}
.badge{display:inline-block;padding:2px 8px;border-radius:3px;font-size:12px;font-weight:600;margin-right:6px}
.ex{background:#fde3e3;color:#a12626}.kp{background:#dff3e8;color:#1f6b40}
.q{font-weight:600;margin:6px 0}
ol{margin:4px 0 8px 18px;padding:0}
li.ans{font-weight:700;color:#1f6b40}
table{border-collapse:collapse;font-size:12.5px;margin-top:6px}
td,th{border:1px solid #e1e5ea;padding:3px 7px;text-align:left;vertical-align:top}
td.ok{background:#e8f6ee}td.bad{background:#fbe9e9}
.meta{color:#6b7480;font-size:12px}
</style></head><body>
<header>
<b>Blind filter viewer</b>
<label>benchmark <select id="fb"><option value="">all</option>__BENCH__</select></label>
<label>category <select id="fc"><option value="">all</option></select></label>
<label>status <select id="fs"><option value="">all</option><option value="excluded">excluded (blind-solvable)</option><option value="kept">kept</option></select></label>
<span id="count" class="meta"></span>
</header>
<div id="cards"></div>
<script>
const DATA = __DATA__;
const cards = document.getElementById('cards');
const fb = document.getElementById('fb'), fc = document.getElementById('fc'), fs = document.getElementById('fs');
function esc(s){return (s??'').toString().replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function media(d){
  if(!d.video) return '<div class="meta">video not resolved</div>';
  if(d.frames && d.frames.length) return '<div class="frames">'+d.frames.map(f=>'<img src="'+f+'">').join('')+'</div>';
  return '<video controls preload="metadata" src="'+d.video+'"></video>';
}
function render(){
  cards.innerHTML='';
  const cats = new Set();
  let n=0;
  for(const d of DATA){
    if(fb.value && d.benchmark!==fb.value) continue;
    cats.add(d.category);
    if(fc.value && d.category!==fc.value) continue;
    if(fs.value && d.status!==fs.value) continue;
    n++;
    const opts = d.options.map((o,i)=>'<li class="'+(i===d.answer_idx?'ans':'')+'">'+esc(o)+'</li>').join('');
    let rows='';
    for(const m of d.models){
      rows += '<tr><th>'+esc(m.name)+'</th>'+m.rot.map(r=>'<td class="'+(r.correct?'ok':'bad')+'">shift '+r.shift+': <code>'+esc(r.raw)+'</code> &rarr; '+(r.pred>=0?LETTERS[r.pred]:'?')+' (gt '+LETTERS[r.answer]+')</td>').join('')+'</tr>';
    }
    cards.insertAdjacentHTML('beforeend',
      '<div class="card"><div>'+media(d)+'<div class="meta">'+esc(d.video_id)+' '+esc(d.extra)+'</div></div><div>'+
      '<span class="badge '+(d.status==='excluded'?'ex':'kp')+'">'+d.status+'</span><span class="meta">'+esc(d.benchmark)+' / '+esc(d.category)+(d.subcategory?' / '+esc(d.subcategory):'')+' &middot; '+esc(d.item_id)+'</span>'+
      '<div class="q">'+esc(d.question).replace(/\\n/g,'<br>')+'</div><ol type="A">'+opts+'</ol><table>'+rows+'</table></div></div>');
  }
  if(!fc.value || !cats.has(fc.value)){ fc.innerHTML='<option value="">all</option>'+[...cats].sort().map(c=>'<option>'+esc(c)+'</option>').join(''); }
  document.getElementById('count').textContent = n+' items shown';
}
const LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ';
fb.onchange=()=>{fc.value='';render();}; fc.onchange=render; fs.onchange=render; render();
</script></body></html>"""


def link_media(item, out_dir, copy=False):
    """Return (video_rel, frames_rel_list). Handles mp4 files and frame folders."""
    src = item.video_path
    if not src or not os.path.exists(src):
        return "", []
    vdir = os.path.join(out_dir, "videos")
    os.makedirs(vdir, exist_ok=True)
    safe = item.item_id.replace(":", "_").replace("/", "_")
    if os.path.isdir(src):
        frames = sorted(f for f in os.listdir(src) if f.lower().endswith((".jpg", ".png", ".jpeg")))
        if not frames:
            return "", []
        pick = [frames[int(i * (len(frames) - 1) / 7)] for i in range(8)] if len(frames) > 8 else frames
        rel = []
        for f in pick:
            dst = os.path.join(vdir, f"{safe}__{f}")
            _link(os.path.join(src, f), dst, copy)
            rel.append(os.path.relpath(dst, out_dir))
        return "frames", rel
    ext = os.path.splitext(src)[1] or ".mp4"
    dst = os.path.join(vdir, safe + ext)
    _link(src, dst, copy)
    return os.path.relpath(dst, out_dir), []


def _link(src, dst, copy):
    if os.path.lexists(dst):
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--flags", required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out-dir", default="viewer")
    ap.add_argument("--n-excluded", type=int, default=30, help="per benchmark")
    ap.add_argument("--n-kept", type=int, default=10, help="per benchmark")
    ap.add_argument("--benchmarks", default="", help="comma list to restrict")
    ap.add_argument("--copy", action="store_true", help="copy videos instead of symlinking")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = [f for p in args.items for f in sorted(glob.glob(p))]
    items = {it.item_id: it for it in frame_to_items(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))}
    flags = pd.read_parquet(args.flags).set_index("item_id")
    preds = {os.path.splitext(os.path.basename(p))[0]: pd.read_parquet(p) for p in args.preds}
    want = set(b for b in args.benchmarks.split(",") if b)

    rng = random.Random(args.seed)
    chosen = []
    for b, g in flags.groupby("benchmark"):
        if want and b not in want:
            continue
        ex = [i for i in g.index[g["excluded"] == 1] if i in items]
        kp = [i for i in g.index[g["excluded"] == 0] if i in items]
        rng.shuffle(ex); rng.shuffle(kp)
        chosen += [(i, "excluded") for i in ex[: args.n_excluded]] + [(i, "kept") for i in kp[: args.n_kept]]

    os.makedirs(args.out_dir, exist_ok=True)
    data = []
    for item_id, status in chosen:
        it = items[item_id]
        video, frames = link_media(it, args.out_dir, args.copy)
        models = []
        for name, df in preds.items():
            rows = df[df["item_id"] == item_id].sort_values("shift")
            if rows.empty:
                continue
            models.append({"name": name, "rot": [{"shift": int(r.shift), "raw": str(r.raw_output).strip(), "pred": int(r.pred_idx),
                                                  "answer": int(r.answer_idx), "correct": int(r.correct)} for r in rows.itertuples()]})
        data.append({"item_id": it.item_id, "benchmark": it.benchmark, "category": it.category, "subcategory": it.subcategory,
                     "question": it.question, "options": it.options, "answer_idx": it.answer_idx, "video": video, "frames": frames,
                     "video_id": it.video_id, "extra": json.dumps(it.extra, ensure_ascii=False) if it.extra else "", "status": status, "models": models})
    benches = sorted({d["benchmark"] for d in data})
    page = (TEMPLATE.replace("__BENCH__", "".join(f"<option>{html.escape(b)}</option>" for b in benches))
            .replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    with open(os.path.join(args.out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {os.path.join(args.out_dir, 'index.html')} with {len(data)} items; serve with: cd {args.out_dir} && python -m http.server 8000")


if __name__ == "__main__":
    main()
