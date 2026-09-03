#!/usr/bin/env python3
"""Stage 1 결과 집계·그림. GPU 불필요 (JSON + npz 만 읽음).

입력: stage1_mask_opt.py 결과 디렉터리 (results/stage1_<model>_<mode>_<verifier>/)
출력: <dir>/figs/*.png, <dir>/summary.md, <dir>/summary.json

그림
  1. rd_curves.png      비디오별 (keep, KL) 곡선 + oracle/기준선 중앙값 곡선
  2. budget_eps.png     허용 손실 ε 별 최소 충분 예산 b*(ε) 분포 — oracle vs 기준선 (예산 배수)
  3. budget_by_task.png 질문 task_type 별 b*(ε) 분포 (agnostic 마스크 아래 질문별 KL 곡선 기준)
  4. preserve.png       keep 에 따른 답 일치율(same_as_full)·정확도 / 캡션 토큰 일치율
  5. mask_profile.png   마스크의 시간 프로파일(프레임별 keep 비율)과 공간 지도(14×14) — λ 별
  6. nested.png         λ 간 subset 포함 비율 (작은 집합이 큰 집합에 얼마나 들어 있나)
  --compare 로 여러 디렉터리 중앙값 곡선 겹치기 (letters vs caption, agnostic vs aware, 0.5B vs 7B)

실행:
  python oracle/plot_stage1.py oracle/results/stage1_llava-onevision-qwen2-7b-ov_agnostic_letters
  python oracle/plot_stage1.py A_DIR --compare B_DIR C_DIR --labels letters caption aware
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

EPS_LIST = [0.003, 0.01, 0.03, 0.1]


# ----------------------------------------------------------------------------
# 로드
# ----------------------------------------------------------------------------
def load_dir(d):
    recs = []
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        if os.path.basename(p).startswith("_") or os.path.basename(p) == "summary.json":
            continue
        try:
            r = json.load(open(p))
        except Exception as e:  # 진행 중 파일
            print(f"[skip] {p}: {e}")
            continue
        if r.get("runs"):
            recs.append(r)
    return recs


def curves_from_record(r):
    """run 마다 (keep_frac[], oracle_kl[], baselines{name: kl[]}, per_q_kl[[q][pt]], q_meta[]) 를 낸다."""
    out = []
    qmeta_all = r["questions"]
    for ri, run in enumerate(r["runs"]):
        pts = sorted(run["points"], key=lambda p: p["keep_frac"])
        keep = np.array([p["keep_frac"] for p in pts])
        kl = np.array([p["oracle"]["kl_mean"] for p in pts])
        bl = {b: np.array([p["baselines"][b]["kl_mean"] for p in pts]) for b in pts[0]["baselines"]}
        # 질문별 KL 곡선
        if r["mode"] == "aware":
            qmeta = [qmeta_all[ri]]
        else:
            qmeta = qmeta_all
        per_q = np.array([[p["per_q"][qi]["kl"] for p in pts] for qi in range(len(pts[0]["per_q"]))])
        same = np.array([p["oracle"].get("same_as_full", np.nan) for p in pts])
        acc = np.array([p["oracle"].get("acc", np.nan) for p in pts])
        cap = np.array([p["oracle"].get("caption_token_agree", np.nan) for p in pts])
        out.append({"keep": keep, "kl": kl, "bl": bl, "per_q": per_q, "qmeta": qmeta,
                    "same": same, "acc": acc, "cap": cap, "lambdas": [p["lambda"] for p in pts],
                    "n_keep": [p["n_keep"] for p in pts]})
    return out


# ----------------------------------------------------------------------------
# 최소 충분 예산 b*(ε): 곡선을 log-log 로 보간해 KL ≤ ε 인 가장 작은 keep
# ----------------------------------------------------------------------------
def budget_at(keep, kl, eps):
    """keep 오름차순. 반환 (b*, censored) — censored: 'low'(가장 작은 점도 ε 이하), 'high'(가장 큰 점도 ε 초과), None"""
    keep = np.asarray(keep, float)
    kl = np.asarray(kl, float)
    order = np.argsort(keep)
    keep, kl = keep[order], np.maximum(kl[order], 1e-8)
    ok = kl <= eps
    if ok[0]:
        return float(keep[0]), "low"
    if not ok.any():
        return float("nan"), "high"
    i = int(np.argmax(ok))            # 처음으로 ε 이하가 되는 점 (i-1 은 초과)
    x0, x1 = np.log(keep[i - 1]), np.log(keep[i])
    y0, y1 = np.log(kl[i - 1]), np.log(kl[i])
    t = (np.log(eps) - y0) / (y1 - y0) if y1 != y0 else 1.0
    return float(np.exp(x0 + t * (x1 - x0))), None


# ----------------------------------------------------------------------------
# 그림
# ----------------------------------------------------------------------------
def fig_rd(curves, out, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in curves:
        ax.plot(c["keep"], np.maximum(c["kl"], 1e-5), color="tab:orange", alpha=0.15, lw=1)
    grid = np.logspace(np.log10(min(c["keep"].min() for c in curves)), np.log10(max(c["keep"].max() for c in curves)), 40)

    def median_curve(get):
        ys = []
        for c in curves:
            k, v = c["keep"], np.maximum(get(c), 1e-5)
            o = np.argsort(k)
            ys.append(np.interp(np.log(grid), np.log(k[o]), np.log(v[o]), left=np.nan, right=np.nan))
        return np.exp(np.nanmedian(np.array(ys), axis=0))

    ax.plot(grid, median_curve(lambda c: c["kl"]), color="tab:orange", lw=2.5, label="oracle (median)")
    colors = {"random": "tab:gray", "frame_uniform": "tab:blue", "grid": "tab:green"}
    for b in curves[0]["bl"]:
        ax.plot(grid, median_curve(lambda c, b=b: c["bl"][b]), color=colors.get(b, None), lw=2, ls="--", label=f"{b} (median)")
    for e in EPS_LIST:
        ax.axhline(e, color="k", lw=0.6, ls=":", alpha=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("kept visual tokens (fraction of 6272)"); ax.set_ylabel("KL(p_full ‖ p_S)  (real deletion)")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_budget_eps(curves, out, title):
    import matplotlib.pyplot as plt
    names = ["oracle"] + list(curves[0]["bl"])
    fig, axes = plt.subplots(1, len(EPS_LIST), figsize=(3.2 * len(EPS_LIST), 4), sharey=True)
    stats = {}
    for ax, eps in zip(axes, EPS_LIST):
        data, cens = [], []
        for nm in names:
            vals, nlow, nhigh = [], 0, 0
            for c in curves:
                kl = c["kl"] if nm == "oracle" else c["bl"][nm]
                b, cz = budget_at(c["keep"], kl, eps)
                if cz == "high":
                    nhigh += 1
                    vals.append(1.0)          # 전부 넣어야 함 (상한)
                else:
                    if cz == "low":
                        nlow += 1
                    vals.append(b)
            data.append(vals); cens.append((nlow, nhigh))
            stats[(eps, nm)] = {"median": float(np.nanmedian(vals)), "mean": float(np.nanmean(vals)),
                                "censored_low": nlow, "censored_high": nhigh, "n": len(vals)}
        ax.boxplot(data, labels=names, showfliers=False)
        for i, (nl, nh) in enumerate(cens):
            ax.text(i + 1, 1.02, f"↓{nl} ↑{nh}", ha="center", fontsize=7, color="gray")
        ax.set_yscale("log"); ax.set_title(f"ε = {eps}"); ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y", which="both")
    axes[0].set_ylabel("minimal sufficient budget b*(ε)  (fraction)")
    fig.suptitle(title + "   (↓ = 최소점도 ε 이하, ↑ = 최대점도 ε 초과 → 1.0 으로)", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return stats


def fig_budget_by_task(curves, out, title, eps=0.03):
    import matplotlib.pyplot as plt
    by = defaultdict(list)
    for c in curves:
        for qi, qm in enumerate(c["qmeta"]):
            if qi >= c["per_q"].shape[0]:
                continue
            b, cz = budget_at(c["keep"], c["per_q"][qi], eps)
            label = "caption" if qm.get("qid") == "caption" else qm.get("task_type", "?")
            by[label].append(1.0 if cz == "high" else b)
    if not by:
        return {}
    keys = sorted(by, key=lambda k: -np.nanmedian(by[k]))
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(keys)), 4.5))
    ax.boxplot([by[k] for k in keys], labels=[f"{k}\n(n={len(by[k])})" for k in keys], showfliers=False)
    ax.set_yscale("log"); ax.set_ylabel(f"b*(ε={eps}) per question"); ax.set_title(title)
    ax.tick_params(axis="x", rotation=60, labelsize=7); ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return {k: {"median": float(np.nanmedian(v)), "n": len(v)} for k, v in by.items()}


def fig_preserve(curves, out, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    grid = np.logspace(np.log10(min(c["keep"].min() for c in curves)), np.log10(max(c["keep"].max() for c in curves)), 30)

    def med(get):
        ys = []
        for c in curves:
            v = get(c)
            if np.all(np.isnan(v)):
                continue
            o = np.argsort(c["keep"])
            ys.append(np.interp(np.log(grid), np.log(c["keep"][o]), v[o], left=np.nan, right=np.nan))
        return np.nanmean(np.array(ys), axis=0) if ys else None

    for key, lab in (("same", "answer = full-token answer"), ("acc", "accuracy (GT)"), ("cap", "caption token agreement")):
        y = med(lambda c, key=key: c[key])
        if y is not None:
            ax.plot(grid, y, marker="o", ms=3, label=lab)
    ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_xlabel("kept fraction"); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_mask_profile(d, recs, out, title):
    """masks_<vid>.npz 의 keep 마스크로 λ 별 시간 프로파일(프레임별 keep 비율)과 공간 지도(14×14)."""
    import matplotlib.pyplot as plt
    prof = defaultdict(list)   # lam -> [ (n_frames, per) 배열 ]
    for r in recs:
        p = os.path.join(d, f"masks_{r['videoID']}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        nf, per = r["n_frames"], r["n_vis"] // r["n_frames"]
        for k in z.files:
            if not k.endswith("_keep") or not k.startswith("g0_"):
                continue
            lam = k[len("g0_lam"):-len("_keep")]
            prof[lam].append(z[k].astype(float).reshape(nf, per))
    if not prof:
        return {}
    lams = sorted(prof, key=float)
    side = int(round((next(iter(prof.values()))[0].shape[1]) ** 0.5))
    fig, axes = plt.subplots(2, len(lams), figsize=(2.6 * len(lams), 5.2))
    axes = np.atleast_2d(axes)
    summary = {}
    for j, lam in enumerate(lams):
        m = np.mean(np.stack(prof[lam]), axis=0)           # (nf, per) 비디오 평균 keep 확률
        temporal, spatial = m.mean(1), m.mean(0)
        axes[0, j].bar(np.arange(len(temporal)), temporal, color="tab:orange")
        axes[0, j].set_ylim(0, 1); axes[0, j].set_title(f"λ={lam}  keep={m.mean():.3f}", fontsize=9)
        axes[0, j].set_xlabel("frame"); axes[0, j].tick_params(labelsize=7)
        im = axes[1, j].imshow(spatial.reshape(side, side), vmin=0, vmax=max(0.05, spatial.max()), cmap="magma")
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        # 분해: 분산 중 프레임 항 / 위치 항 이 설명하는 비율 (a[f] + b[p] 근사)
        tot = m.var()
        summary[lam] = {"keep": float(m.mean()), "temporal_var_frac": float(temporal.var() / tot) if tot else None,
                        "spatial_var_frac": float(spatial.var() / tot) if tot else None,
                        "first_frame_keep": float(temporal[0]), "last_frame_keep": float(temporal[-1])}
    axes[0, 0].set_ylabel("keep prob per frame"); axes[1, 0].set_ylabel("keep prob per position")
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return summary


def fig_nested(d, recs, out, title):
    """연속한 λ 쌍에서 |S_small ∩ S_large| / |S_small| — warm start 라 높게 나오는 것이 정상 (cold start 로 재검정 필요)."""
    import matplotlib.pyplot as plt
    vals = defaultdict(list)
    for r in recs:
        p = os.path.join(d, f"masks_{r['videoID']}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        ks = sorted([k for k in z.files if k.startswith("g0_") and k.endswith("_keep")], key=lambda k: float(k[6:-5]))
        for a, b in zip(ks, ks[1:]):        # a: 작은 λ (큰 집합), b: 큰 λ (작은 집합)
            big, small = z[a].astype(bool), z[b].astype(bool)
            if small.sum():
                vals[f"{a[6:-5]}→{b[6:-5]}"].append(float((big & small).sum() / small.sum()))
    if not vals:
        return {}
    keys = list(vals)
    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(keys)), 3.8))
    ax.boxplot([vals[k] for k in keys], labels=keys, showfliers=False)
    ax.set_ylim(0, 1.02); ax.set_ylabel("|S_small ∩ S_large| / |S_small|"); ax.set_title(title, fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return {k: float(np.median(v)) for k, v in vals.items()}


def fig_compare(dirs, labels, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for d, lab in zip(dirs, labels):
        recs = load_dir(d)
        if not recs:
            continue
        curves = [c for r in recs for c in curves_from_record(r)]
        grid = np.logspace(np.log10(min(c["keep"].min() for c in curves)), np.log10(max(c["keep"].max() for c in curves)), 40)
        ys = []
        for c in curves:
            o = np.argsort(c["keep"])
            ys.append(np.interp(np.log(grid), np.log(c["keep"][o]), np.log(np.maximum(c["kl"][o], 1e-5)), left=np.nan, right=np.nan))
        ax.plot(grid, np.exp(np.nanmedian(np.array(ys), axis=0)), lw=2.2, label=f"{lab} (n={len(curves)})")
    for e in EPS_LIST:
        ax.axhline(e, color="k", lw=0.6, ls=":", alpha=0.5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("kept fraction"); ax.set_ylabel("oracle KL (median)")
    ax.legend(); ax.grid(alpha=0.3, which="both"); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--eps_task", type=float, default=0.03)
    args = ap.parse_args()

    d = args.dir.rstrip("/")
    recs = load_dir(d)
    if not recs:
        print("결과 JSON 이 없습니다:", d); return
    curves = [c for r in recs for c in curves_from_record(r)]
    figs = os.path.join(d, "figs"); os.makedirs(figs, exist_ok=True)
    title = os.path.basename(d)
    print(f"[load] {len(recs)} videos, {len(curves)} curves, verifier={recs[0].get('verifier')}, mode={recs[0]['mode']}")

    fig_rd(curves, os.path.join(figs, "rd_curves.png"), title)
    stats_eps = fig_budget_eps(curves, os.path.join(figs, "budget_eps.png"), title)
    by_task = fig_budget_by_task(curves, os.path.join(figs, "budget_by_task.png"), title, eps=args.eps_task)
    fig_preserve(curves, os.path.join(figs, "preserve.png"), title)
    prof = fig_mask_profile(d, recs, os.path.join(figs, "mask_profile.png"), title)
    nested = fig_nested(d, recs, os.path.join(figs, "nested.png"), title + "  (warm start → 낙관적)")
    if args.compare:
        labels = args.labels or [os.path.basename(x.rstrip("/")) for x in [d] + args.compare]
        fig_compare([d] + args.compare, labels, os.path.join(figs, "compare.png"))

    # ---- λ 점별 표 ----
    lam_rows = defaultdict(list)
    for c in curves:
        for i, lam in enumerate(c["lambdas"]):
            lam_rows[lam].append({"keep": c["keep"][i], "kl": c["kl"][i], "same": c["same"][i], "acc": c["acc"][i],
                                  "cap": c["cap"][i], **{f"bl_{b}": c["bl"][b][i] for b in c["bl"]}})
    md = [f"# {title}", "", f"videos={len(recs)}  curves={len(curves)}  verifier={recs[0].get('verifier')}  mode={recs[0]['mode']}", "",
          "## λ 점별 중앙값", "",
          "| λ | keep | oracle KL | " + " | ".join(f"{b} KL" for b in curves[0]["bl"]) + " | same_as_full | acc | cap_agree |",
          "|---|---|---|" + "---|" * len(curves[0]["bl"]) + "---|---|---|"]
    for lam in sorted(lam_rows):
        rows = lam_rows[lam]
        f = lambda k: np.nanmedian([r[k] for r in rows])
        md.append(f"| {lam:g} | {f('keep'):.3f} | {f('kl'):.4f} | " + " | ".join(f"{f('bl_' + b):.4f}" for b in curves[0]["bl"])
                  + f" | {np.nanmean([r['same'] for r in rows]):.2f} | {np.nanmean([r['acc'] for r in rows]):.2f} | {np.nanmean([r['cap'] for r in rows]):.2f} |")
    md += ["", "## 최소 충분 예산 b*(ε) 중앙값 (fraction) — oracle 대비 기준선 배수", "",
           "| ε | " + " | ".join(["oracle"] + list(curves[0]["bl"])) + " | " + " | ".join(f"{b}/oracle" for b in curves[0]["bl"]) + " |",
           "|---|" + "---|" * (1 + 2 * len(curves[0]["bl"]))]
    for eps in EPS_LIST:
        o = stats_eps[(eps, "oracle")]["median"]
        cells = [f"{stats_eps[(eps, nm)]['median']:.3f}" for nm in ["oracle"] + list(curves[0]["bl"])]
        ratio = [f"{stats_eps[(eps, b)]['median'] / o:.1f}×" if o else "-" for b in curves[0]["bl"]]
        md.append(f"| {eps} | " + " | ".join(cells) + " | " + " | ".join(ratio) + " |")
    if by_task:
        md += ["", f"## task_type 별 b*(ε={args.eps_task}) 중앙값 (질문별, agnostic 마스크 아래)", "", "| task_type | b* | n |", "|---|---|---|"]
        for k, v in sorted(by_task.items(), key=lambda kv: -kv[1]["median"]):
            md.append(f"| {k} | {v['median']:.3f} | {v['n']} |")
    if prof:
        md += ["", "## 마스크 프로파일 (비디오 평균)", "", "| λ | keep | 분산 중 프레임 항 | 위치 항 | 첫 프레임 | 마지막 프레임 |", "|---|---|---|---|---|---|"]
        for lam, v in prof.items():
            md.append(f"| {lam} | {v['keep']:.3f} | {v['temporal_var_frac'] if v['temporal_var_frac'] is None else round(v['temporal_var_frac'], 3)} | "
                      f"{v['spatial_var_frac'] if v['spatial_var_frac'] is None else round(v['spatial_var_frac'], 3)} | {v['first_frame_keep']:.3f} | {v['last_frame_keep']:.3f} |")
    if nested:
        md += ["", "## λ 간 포함 비율 중앙값 (warm start → 낙관적)", "", "| λ 쌍 | 포함 비율 |", "|---|---|"]
        md += [f"| {k} | {v:.3f} |" for k, v in nested.items()]
    open(os.path.join(d, "summary.md"), "w").write("\n".join(md) + "\n")
    json.dump({"n_videos": len(recs), "n_curves": len(curves),
               "budget_eps": {f"{e}|{nm}": v for (e, nm), v in stats_eps.items()},
               "by_task": by_task, "mask_profile": prof, "nested": nested},
              open(os.path.join(d, "summary.json"), "w"), indent=2)
    print("\n".join(md))
    print(f"\nfigs → {figs}/   summary → {d}/summary.md")


if __name__ == "__main__":
    main()
