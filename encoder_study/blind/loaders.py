"""Benchmark loaders -> unified Item list.

Dataset ids / splits / video cache conventions mirror lmms-eval's task yamls so
that videos already extracted by lmms-eval (under $HF_HOME/<cache_dir>) are
reused. Only the text (question/options/answer) is required for the blind
filter; video paths are resolved best-effort for the viewer and left "" when
not found.

Every loader signature: load_xxx(**kw) -> List[Item]. Register in LOADERS.
Use `python -m encoder_study.blind.build_items --inspect <name>` to print raw
columns of a benchmark before trusting a loader on a new machine.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Callable, Dict, List, Optional

from .schema import LETTERS, Item, strip_letter_prefix

HF_HOME = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))


def _hf(path: str, name: Optional[str] = None, split: str = "test", revision: Optional[str] = None):
    from datasets import load_dataset

    kw = {}
    if revision:
        kw["revision"] = revision
    try:
        return load_dataset(path, name, split=split, **kw) if name else load_dataset(path, split=split, **kw)
    except Exception as e:  # fall back without revision
        if revision:
            return load_dataset(path, name, split=split) if name else load_dataset(path, split=split)
        raise e


def _first_existing(paths: List[str]) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""


def _answer_index(options: List[str], answer) -> int:
    """answer may be an int index, a letter, or the option text."""
    if isinstance(answer, bool):
        raise ValueError("bool answer")
    if isinstance(answer, int):
        if 0 <= answer < len(options):
            return answer
        if 1 <= answer <= len(options):
            return answer - 1
    s = str(answer).strip()
    if len(s) == 1 and s.upper() in LETTERS[: len(options)]:
        return LETTERS.index(s.upper())
    m = re.match(r"^[\(\[]?([A-Za-z])[\)\].:]", s)
    if m and m.group(1).upper() in LETTERS[: len(options)]:
        return LETTERS.index(m.group(1).upper())
    norm = lambda x: " ".join(str(x).strip().lower().split())
    for i, o in enumerate(options):
        if norm(o) == norm(s) or norm(strip_letter_prefix(o)) == norm(strip_letter_prefix(s)):
            return i
    raise ValueError(f"cannot map answer {answer!r} to options {options!r}")


# ---------------------------------------------------------------- MVBench
MVBENCH_DATA_LIST = {
    "object_interaction": "star/Charades_segment",
    "action_sequence": "star/Charades_segment",
    "action_prediction": "star/Charades_segment",
    "action_localization": "sta/sta_video_segment",
    "moving_count": "clevrer/video_validation",
    "fine_grained_pose": "nturgbd_convert",
    "character_order": "perception/videos",
    "object_shuffle": "perception/videos",
    "egocentric_navigation": "vlnqa",
    "moving_direction": "clevrer/video_validation",
    "episodic_reasoning": "tvqa/video_fps3_hq_segment",
    "fine_grained_action": "Moments_in_Time_Raw/videos",
    "scene_transition": "scene_qa/video",
    "state_change": "perception/videos",
    "moving_attribute": "clevrer/video_validation",
    "action_antonym": "ssv2_video_mp4",
    "unexpected_action": "FunQA_test/test",
    "counterfactual_inference": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "action_count": "perception/videos",
}


def load_mvbench(subtasks: Optional[List[str]] = None, **_) -> List[Item]:
    cache = os.path.join(HF_HOME, "mvbench_video")
    items = []
    for sub in subtasks or MVBENCH_DATA_LIST:
        ds = _hf("OpenGVLab/MVBench", sub, split="train", revision="video")
        folder = MVBENCH_DATA_LIST[sub]
        for i, doc in enumerate(ds):
            opts = [str(c) for c in doc["candidates"]]
            ans = _answer_index(opts, doc["answer"])
            vp = _first_existing([os.path.join(cache, folder, doc["video"]), os.path.join(cache, "data0613", folder, doc["video"])])
            items.append(Item(f"mvbench:{sub}:{i}", "mvbench", sub, doc["question"], opts, ans, vp, str(doc["video"])))
    return items


# ---------------------------------------------------------------- TVBench
TVBENCH_SUBTASKS = [
    "action_antonym", "action_count", "action_localization", "action_sequence", "egocentric_sequence",
    "moving_direction", "object_count", "object_shuffle", "scene_transition", "unexpected_action",
]


def _tv_candidates(doc) -> List[str]:
    c = doc.get("candidates", doc.get("options"))
    if isinstance(c, list):
        return [str(x) for x in c]
    out = []
    for k in range(26):
        v = doc.get(f"option{k}")
        if v not in (None, ""):
            out.append(str(v))
    return out


def load_tvbench(subtasks: Optional[List[str]] = None, **_) -> List[Item]:
    cache = os.path.join(HF_HOME, "tvbench")
    items = []
    for sub in subtasks or TVBENCH_SUBTASKS:
        ds = _hf("FunAILab/TVBench", sub, split="train")
        for i, doc in enumerate(ds):
            opts = _tv_candidates(doc)
            answer = next((doc.get(k) for k in ["answer", "correct_answer", "label", "correct_choice"] if doc.get(k) is not None), "")
            ans = _answer_index(opts, answer)
            vname = doc.get("video") or doc.get("video_path") or doc.get("video_file") or ""
            if isinstance(vname, dict):
                vname = vname.get("path") or vname.get("video") or vname.get("filename") or ""
            vname = str(vname)
            cands = [os.path.join(cache, p, vname) for p in ["", "video", "videos", "data", sub, f"video/{sub}", f"videos/{sub}"]]
            q = str(doc.get("question") or doc.get("prompt") or doc.get("query") or "")
            items.append(Item(f"tvbench:{sub}:{i}", "tvbench", sub, q, opts, ans, _first_existing(cands), vname))
    return items


# ---------------------------------------------------------------- TOMATO
def _guess_category_column(ds, exclude=("question", "options", "answer", "video_path", "id", "key")) -> Optional[str]:
    """Pick the string column with 2..30 distinct values (benchmark task/category labels)."""
    best = None
    for col in ds.column_names:
        if col in exclude:
            continue
        vals = ds[col][:2000] if len(ds) > 2000 else ds[col]
        if not vals or not isinstance(vals[0], str):
            continue
        n = len(set(vals))
        if 2 <= n <= 30 and (best is None or n > best[1]):
            best = (col, n)
    return best[0] if best else None


def load_tomato(**_) -> List[Item]:
    cache = os.path.join(HF_HOME, "TOMATO")
    ds = _hf("lmms-eval/TOMATO", split="test")
    cat_col = next((k for k in ["task", "task_type", "reasoning_type", "category", "type"] if k in ds.column_names), None) or _guess_category_column(ds)
    print(f"[tomato] columns={ds.column_names}  category column -> {cat_col}")
    items = []
    for i, doc in enumerate(ds):
        opts = [str(o) for o in doc["options"]]
        ans = _answer_index(opts, doc["answer"])
        cat = str(doc[cat_col]) if cat_col else "unknown"
        sub = str(doc.get("demonstration_type") or doc.get("video_type") or "")
        vp = _first_existing([os.path.join(cache, str(doc["video_path"]))])
        items.append(Item(f"tomato:{i}", "tomato", cat, doc["question"], opts, ans, vp, str(doc["video_path"]), sub))
    return items


# ---------------------------------------------------------------- VSI-Bench (MC types only)
VSI_MCA = ["object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard",
           "object_rel_distance", "route_planning", "obj_appearance_order"]


def load_vsibench(**_) -> List[Item]:
    cache = os.path.join(HF_HOME, "vsibench")
    ds = _hf("nyu-visionx/VSI-Bench", "full", split="test")
    items = []
    for i, doc in enumerate(ds):
        if doc["question_type"] not in VSI_MCA:
            continue  # numeric-answer types cannot be rotated; handled by MRA later
        opts = [strip_letter_prefix(o) for o in doc["options"]]
        ans = _answer_index(opts, doc["ground_truth"])
        vp = _first_existing([os.path.join(cache, f"{doc['dataset']}/{doc['scene_name']}.mp4")])
        items.append(Item(f"vsibench:{i}", "vsibench", doc["question_type"], doc["question"], opts, ans, vp,
                          f"{doc['dataset']}/{doc['scene_name']}", str(doc.get("dataset", ""))))
    return items


# ---------------------------------------------------------------- Perception Test (val, MC)
def load_perceptiontest(**_) -> List[Item]:
    cache = os.path.join(HF_HOME, "perceptiontest_val", "videos")
    ds = _hf("lmms-eval/PerceptionTest_Val", "mc_question_val", split="validation")
    items = []
    for i, doc in enumerate(ds):
        opts = [str(o) for o in doc["options"]]
        ans = int(doc["answer_id"])
        vp = _first_existing([os.path.join(cache, f"{doc['video_name']}.mp4"), os.path.join(cache, f"{doc['video_name']}.MP4")])
        items.append(Item(f"perceptiontest:{doc.get('question_id', i)}", "perceptiontest", str(doc.get("area", "")), doc["question"],
                          opts, ans, vp, str(doc["video_name"]), str(doc.get("reasoning", "")), {"tag": doc.get("tag")}))
    return items


# ---------------------------------------------------------------- MotionBench
_MB_META = "https://raw.githubusercontent.com/zai-org/MotionBench/main/data/video_info.meta.jsonl"
_OPT_LINE = re.compile(r"^\s*[\(\[]?([A-F])[\)\].:]\s*(.+?)\s*$")


def _split_question_options(text: str):
    lines = [l for l in str(text).replace("\r", "").split("\n")]
    q, opts = [], []
    for l in lines:
        m = _OPT_LINE.match(l)
        if m:
            opts.append(m.group(2))
        elif not opts:
            q.append(l)
    return "\n".join(q).strip(), opts


def load_motionbench(**_) -> List[Item]:
    cache = os.path.join(HF_HOME, "motionbench")
    os.makedirs(cache, exist_ok=True)
    meta = os.path.join(cache, "video_info.meta.jsonl")
    if not os.path.exists(meta):
        urllib.request.urlretrieve(_MB_META, meta)
    roots = [os.getenv("MOTIONBENCH_VIDEO_DIR", ""), os.path.join(cache, "videos"), cache]
    items = []
    with open(meta, encoding="utf-8") as f:
        for li, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            qtype, vtype = str(row.get("question_type", "Unknown")), str(row.get("video_type", ""))
            vname, key = str(row.get("video_path", "")), str(row.get("key", ""))
            vp = _first_existing([os.path.join(r, vname) for r in roots if r])
            qas = row.get("qa") or ([{"question": row.get("question"), "answer": row.get("answer"), "options": row.get("options")}] if row.get("question") else [])
            for qi, qa in enumerate(qas):
                ans_letter = str(qa.get("answer", "")).strip().upper()
                if not ans_letter or ans_letter == "NA":
                    continue
                opts = qa.get("options")
                if isinstance(opts, list) and opts and isinstance(opts[0], dict):
                    opts = [str(o.get("text", "")) for o in opts]
                    q = str(qa.get("question", "")).strip()
                else:
                    q, opts = _split_question_options(qa.get("question", ""))
                if len(opts) < 2:
                    continue
                items.append(Item(f"motionbench:{key or li}:{qi}", "motionbench", qtype, q, opts, _answer_index(opts, ans_letter), vp, vname, vtype))
    return items


# ---------------------------------------------------------------- MME-VideoOCR (MC rows only)
def load_mme_videoocr(video_root: str = "", **_) -> List[Item]:
    ds = _hf("DogNeverSleep/MME-VideoOCR_Dataset", split="train")
    items = []
    for i, doc in enumerate(ds):
        opts = [strip_letter_prefix(o) for o in (doc.get("option") or [])]
        if len(opts) < 2:
            continue
        try:
            ans = _answer_index(opts, doc["answer"])
        except ValueError:
            continue
        vid = str(doc.get("video_index", i))
        vp = _first_existing([os.path.join(video_root, f"{vid}.mp4")]) if video_root else ""
        items.append(Item(f"mme_videoocr:{doc.get('index', i)}", "mme_videoocr", str(doc.get("task_type", "")), doc["question"], opts, ans, vp, vid,
                          str(doc.get("task", "")), {"eval_method": doc.get("eval_method")}))
    return items


# ---------------------------------------------------------------- STAR (val json)
def load_star(star_json: str = "", video_root: str = "", **_) -> List[Item]:
    if not star_json:
        raise ValueError("--star-json path to STAR_val.json is required")
    with open(star_json, encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for doc in data:
        choices = sorted(doc["choices"], key=lambda c: c.get("choice_id", 0))
        opts = [str(c["choice"]) for c in choices]
        ans = _answer_index(opts, doc["answer"])
        qid = str(doc["question_id"])
        cat = qid.split("_")[0]
        vp = _first_existing([os.path.join(video_root, f"{doc['video_id']}.mp4")]) if video_root else ""
        items.append(Item(f"star:{qid}", "star", cat, doc["question"], opts, ans, vp, str(doc["video_id"]), "_".join(qid.split("_")[:2]),
                          {"start": doc.get("start"), "end": doc.get("end")}))
    return items


# ---------------------------------------------------------------- CLEVRER (val json; MC choices -> binary items)
def load_clevrer(clevrer_json: str = "", video_root: str = "", **_) -> List[Item]:
    if not clevrer_json:
        raise ValueError("--clevrer-json path to CLEVRER validation.json is required")
    with open(clevrer_json, encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for vid in data:
        vname = vid.get("video_filename", "")
        vp = _first_existing([os.path.join(video_root, vname)]) if video_root and vname else ""
        for q in vid.get("questions", []):
            qtype = q.get("question_type", "")
            if "choices" not in q:
                continue  # descriptive (open-ended) handled elsewhere
            for c in q["choices"]:
                text = f"{q['question']}\nStatement: {c['choice']}\nIs the statement correct?"
                ans = 0 if str(c.get("answer", "")).lower().startswith("correct") else 1
                items.append(Item(f"clevrer:{vid.get('scene_index')}:{q.get('question_id')}:{c.get('choice_id')}", "clevrer", qtype, text,
                                  ["yes", "no"], ans, vp, vname))
    return items


LOADERS: Dict[str, Callable[..., List[Item]]] = {
    "mvbench": load_mvbench,
    "tvbench": load_tvbench,
    "tomato": load_tomato,
    "vsibench": load_vsibench,
    "perceptiontest": load_perceptiontest,
    "motionbench": load_motionbench,
    "mme_videoocr": load_mme_videoocr,
    "star": load_star,
    "clevrer": load_clevrer,
}

# Raw dataset handles for --inspect (HF-hosted ones only)
INSPECT = {
    "mvbench": ("OpenGVLab/MVBench", "action_count", "train"),
    "tvbench": ("FunAILab/TVBench", "action_count", "train"),
    "tomato": ("lmms-eval/TOMATO", None, "test"),
    "vsibench": ("nyu-visionx/VSI-Bench", "full", "test"),
    "perceptiontest": ("lmms-eval/PerceptionTest_Val", "mc_question_val", "validation"),
    "mme_videoocr": ("DogNeverSleep/MME-VideoOCR_Dataset", None, "train"),
}
