#!/usr/bin/env python3
"""Streaming loaders for the SCALE-phase ARC relation corpus (survey: 1D-ARC / Mini-ARC / neoneye
arc-dataset-collection / BARC ARC-Heavy). Same task format as ground_conceptarc.py:
  yields (task_id, train, test) where train/test are lists of (input:np.int, output:np.int) pairs.

RESOURCE NOTE: every loader here is LAZY/STREAMING. Nothing pulls a full 100k-task corpus into RAM.
- neoneye loaders read one small per-task JSON at a time (download-on-demand via hf_hub_download or raw GitHub).
- barc_heavy_stream reads the 2.8GB JSONL LINE-BY-LINE (constant memory) and can stop after `limit`.
Plan the FULL pull on Kaggle, not the box.

Two corpus *shapes*:
  (A) GRID-only ARC JSON  -> {"train":[{"input","output"}...], "test":[...]}  (1D-ARC, Mini-ARC, all neoneye)
  (B) PROGRAM+examples    -> BARC ARC-Heavy JSONL: {"source": <py>, "examples": [[in,out],...], "seeds":[...]}
      (B) is the gold for a relation-GENERATING proposer: `source` is the executable relation.
"""
import json, os, urllib.request
import numpy as np

# ---------- (A) generic ARC-grid JSON ----------

def _g(grid):
    return np.asarray(grid, dtype=np.int8)

def task_from_arc_json(obj, tid):
    """Standard ARC task dict -> (tid, train, test). test outputs present in these corpora (self-contained)."""
    train = [(_g(p["input"]), _g(p["output"])) for p in obj["train"]]
    test = [(_g(p["input"]), _g(p.get("output", p["input"]))) for p in obj["test"]]
    return tid, train, test


def stream_arc_json_dir(local_dir, recursive=True):
    """Stream every *.json ARC task under a local directory (1D-ARC clone, Mini-ARC clone, neoneye clone)."""
    for root, _d, files in os.walk(local_dir):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(root, fn)
            try:
                obj = json.load(open(p))
                if "train" in obj and "test" in obj:
                    yield task_from_arc_json(obj, os.path.relpath(p, local_dir))
            except Exception:
                continue
        if not recursive:
            break


# ---------- neoneye/arc-dataset-collection: stream a sub-dataset from HF Hub (lazy, per-file) ----------
# Repo (HF dataset mirror of the GitHub collection). Sub-dataset = a folder under dataset/<NAME>/data/.
NEONEYE_REPO = "neoneye/arc-dataset-collection"  # HF dataset id (mirror); also github.com/neoneye/arc-dataset-collection


def stream_neoneye_subset(name, repo_local_root, limit=None):
    """Stream tasks from a CLONED neoneye collection at repo_local_root/dataset/<name>/.
    On Kaggle: `huggingface-cli download neoneye/arc-dataset-collection --repo-type dataset` once, then point here.
    On the box: clone only the sub-folder you want (git sparse-checkout) — do NOT pull all 20."""
    base = os.path.join(repo_local_root, "dataset", name)
    n = 0
    for tid, train, test in stream_arc_json_dir(base):
        yield f"{name}/{tid}", train, test
        n += 1
        if limit and n >= limit:
            return


def stream_neoneye_github(name, sub="data", limit=20):
    """SAMPLE-ONLY: pull a few tasks straight from raw.githubusercontent (no clone). For inspection on the box.
    Recurses one level into sharded layouts (ARC-Heavy/data/a, arc_1d/<type>), honoring the global `limit`."""
    import json as _j
    api = f"https://api.github.com/repos/neoneye/arc-dataset-collection/contents/dataset/{name}/{sub}"
    listing = _j.loads(urllib.request.urlopen(api, timeout=30).read())
    n = 0
    for ent in listing:
        if limit and n >= limit:
            return
        if ent["type"] == "dir":
            for t in stream_neoneye_github(name, sub=f"{sub}/{ent['name']}", limit=(limit - n) if limit else None):
                yield t
                n += 1
                if limit and n >= limit:
                    return
        elif ent["name"].endswith(".json"):
            raw = urllib.request.urlopen(ent["download_url"], timeout=30).read()
            obj = _j.loads(raw)
            if "train" in obj and "test" in obj:
                yield task_from_arc_json(obj, f"{name}/{ent['name']}")
                n += 1


# ---------- (B) BARC ARC-Heavy: PROGRAM + examples, streamed line-by-line ----------
BARC_HEAVY_REPO = "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems"
BARC_HEAVY_FILE = "data_100k.jsonl"  # 2.8GB; also data_suggestfunction_100k.jsonl (3.4GB)


def barc_record_to_task(rec, tid):
    """BARC heavy record -> (tid, train, test, program). `examples`=[[in,out],...]; last pair held out as test."""
    ex = rec["examples"]
    pairs = [(_g(i), _g(o)) for i, o in ex]
    train, test = pairs[:-1], pairs[-1:]
    return tid, train, test, rec["source"]


def barc_heavy_stream(jsonl_path, limit=None):
    """Stream ARC-Heavy tasks+programs from a LOCAL JSONL, line-by-line (constant RAM). Yields
    (tid, train, test, program_source). `program_source` is the relation curriculum gold."""
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                yield barc_record_to_task(rec, f"barc_heavy/{i}")
            except Exception:
                continue
            if limit and i + 1 >= limit:
                return


def barc_heavy_stream_remote(limit=20, suggestfunction=False):
    """SAMPLE-ONLY: stream the first `limit` BARC records over HTTP Range (no 2.8GB download). Box inspection."""
    from huggingface_hub import hf_hub_url
    fname = "data_suggestfunction_100k.jsonl" if suggestfunction else BARC_HEAVY_FILE
    url = hf_hub_url(BARC_HEAVY_REPO, fname, repo_type="dataset")
    # grab enough bytes for `limit` records (~10-20KB each); cap to keep it tiny
    nbytes = max(80000, limit * 25000)
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes}"})
    raw = urllib.request.urlopen(req, timeout=120).read()
    out = []
    for i, line in enumerate(raw.split(b"\n")):
        if i >= limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            out.append(barc_record_to_task(rec, f"barc_heavy/{i}"))
        except Exception:
            break  # last line in the range is likely truncated
    return out


if __name__ == "__main__":
    print("== SAMPLE: neoneye 1D-ARC via GitHub ==")
    for tid, train, test in stream_neoneye_github("1D-ARC", sub="data", limit=3):
        print(" ", tid, "train", len(train), "shapes", [t[0].shape for t in train[:1]])
    print("== SAMPLE: BARC ARC-Heavy via HTTP Range (program gold) ==")
    for tid, train, test, prog in barc_heavy_stream_remote(limit=2):
        print(" ", tid, "train", len(train), "test", len(test), "prog_lines", prog.count(chr(10)))
        desc = [l for l in prog.split(chr(10)) if l.startswith("# ")][:4]
        print("   concepts/desc:", " | ".join(desc)[:160])
