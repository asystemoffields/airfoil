#!/usr/bin/env python3
"""Pull a small VERIFIED on-box SAMPLE of the program-bearing sources and write a manifest.

RAM-safe: BARC is streamed line-by-line (we stop after a few hundred records, never the 3GB
file); RE-ARC is generated on CPU from the <1MB code files. Outputs compact JSONL under
data/samples/ + data/manifest.json. Each sample line:
  {"task_id", "concepts"/"program_snippet", "train":[[in,out]...], "test":[...], "program":...}
grids are plain int lists (re-load with np.asarray). Run:
  /data/llm/.venv/bin/python data/pull_sample.py
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import arc_sources as S  # noqa: E402

OUT = os.path.join(HERE, "samples")
os.makedirs(OUT, exist_ok=True)


def grids(pairs):
    return [[a.tolist(), b.tolist()] for a, b in pairs]


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return os.path.getsize(path)


def main():
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "On-box VERIFIED sample for SCALE-phase relation curriculum. "
                        "Full corpus -> Kaggle (see DATA.md). Grids are int lists.",
                "samples": {}}

    # --- BARC ARC-Heavy (canonical 4omini): 300 streamed tasks w/ program + concepts ---
    print("Streaming BARC heavy_4omini sample (300 tasks)...")
    t0 = time.time(); rows = []
    for tid, train, test, meta in S.load_barc("heavy_4omini", limit=300, n_train=4):
        rows.append({"task_id": tid, "concepts": meta["concepts"],
                     "description": meta["description"], "seeds": meta["seeds"],
                     "program": meta["program"], "train": grids(train), "test": grids(test)})
    p = os.path.join(OUT, "barc_heavy_4omini_sample.jsonl")
    size = dump(p, rows)
    print(f"  wrote {len(rows)} tasks, {size/1e6:.1f}MB, {time.time()-t0:.0f}s")
    manifest["samples"]["barc_heavy_4omini"] = {
        "file": "samples/barc_heavy_4omini_sample.jsonl", "n_tasks": len(rows),
        "bytes": size, "has_programs": True, "split": "first 4 demos=train, rest=test (shuffled)",
        "source_repo": S.BARC_REPOS["heavy_4omini"][0], "source_file": S.BARC_REPOS["heavy_4omini"][1],
        "pull": "remote HTTP line-stream (no bulk download)"}

    # --- BARC suggestfunction (library-aware programs): 100 tasks ---
    print("Streaming BARC 200k_suggestfunction sample (100 tasks)...")
    t0 = time.time(); rows = []
    try:
        for tid, train, test, meta in S.load_barc("heavy_200k_suggestfn", limit=100, n_train=4):
            rows.append({"task_id": tid, "concepts": meta["concepts"],
                         "description": meta["description"], "seeds": meta["seeds"],
                         "program": meta["program"], "train": grids(train), "test": grids(test)})
        p = os.path.join(OUT, "barc_suggestfunction_sample.jsonl")
        size = dump(p, rows)
        print(f"  wrote {len(rows)} tasks, {size/1e6:.1f}MB, {time.time()-t0:.0f}s")
        manifest["samples"]["barc_suggestfunction"] = {
            "file": "samples/barc_suggestfunction_sample.jsonl", "n_tasks": len(rows),
            "bytes": size, "has_programs": True,
            "note": "programs written against a SUGGESTED helper LIBRARY -> compositional vocabulary",
            "source_repo": S.BARC_REPOS["heavy_200k_suggestfn"][0],
            "source_file": S.BARC_REPOS["heavy_200k_suggestfn"][1]}
    except Exception as e:
        print(f"  SKIP suggestfunction ({e!r})")
        manifest["samples"]["barc_suggestfunction"] = {"error": repr(e)}

    # --- RE-ARC generated: all 400 relations x 2 instances, w/ verifier program ---
    print("Generating RE-ARC sample (400 relations x 2 instances)...")
    t0 = time.time(); rows = []
    ids = S.rearc_task_ids()
    for tid, train, test, prog in S.load_rearc_generated(
            task_ids=ids, n_train=4, n_test=1, diff_lb=0.0, diff_ub=0.6, instances_per_task=2):
        rows.append({"task_id": tid, "program": prog,
                     "train": grids(train), "test": grids(test)})
    p = os.path.join(OUT, "rearc_generated_sample.jsonl")
    size = dump(p, rows)
    print(f"  wrote {len(rows)} instances ({len(ids)} relations), {size/1e6:.1f}MB, {time.time()-t0:.0f}s")
    manifest["samples"]["rearc_generated"] = {
        "file": "samples/rearc_generated_sample.jsonl", "n_instances": len(rows),
        "n_relations": len(ids), "bytes": size, "has_programs": True,
        "note": "verifier_<id> SSA DSL program per relation; UNLIMITED more via load_rearc_generated",
        "source_repo": S.REARC_REPO, "diff_band": [0.0, 0.6]}

    # --- record local (no-pull) sources too ---
    manifest["local_sources"] = {
        "conceptarc": {"loader": "arc_sources.load_conceptarc", "n_tasks": 160,
                       "path": "/data/conceptarc", "has_programs": False, "role": "held-out eval"},
        "arc1": {"loader": "arc_sources.load_arc1", "n_train": 400, "n_eval": 400,
                 "path": S.ARC1_DIR, "has_programs": False, "role": "gate target"},
        "arc2": {"loader": "arc_sources.load_arc2", "n_train": 1000, "n_eval": 120,
                 "path": S.ARC2_DIR, "has_programs": False, "role": "gate target (harder)"}}

    mpath = os.path.join(HERE, "manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=2)
    print(f"manifest -> {mpath}")


if __name__ == "__main__":
    main()
