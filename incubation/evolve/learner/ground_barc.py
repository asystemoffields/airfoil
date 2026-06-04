#!/usr/bin/env python3
"""Branch-B diversity/curriculum source — BARC ARC-Heavy (barc0/* on HF). LLM-GENERATED ARC tasks, each with a
PYTHON SOLUTION PROGRAM. The program is gold for seeding a RELATION curriculum (generative-proposer training):
its `# concepts:` / `# description:` header is a natural-language relation label, and the body is the executable
relation. The `examples` field already contains the program's precomputed (input,output) pairs, so we do NOT need
to run the programs (which would require BARC's seeds/common.py DSL) just to get grids.

RESOURCE NOTE (7GB box): the canonical sets are a SINGLE ~3.15GB JSONL file. DO NOT load_dataset() / download it
whole. This loader STREAMS the JSONL line-by-line over HTTP and holds one record at a time. Pull a small sample on
the box; do the full pull on Kaggle. Run: /data/llm/.venv/bin/python ground_barc.py
"""
import json, urllib.request, random
import numpy as np

# The clean structured JSON sets (source=program, examples=grids, seeds=origin .py names).
# ARC-Heavy canonical = the gpt4omini-description set (~100k); the gpt4-description set is the other ~100k;
# 200k_HEAVY_* is the largest (~68k rows post-filter, more pairs/row). All share the SAME schema.
REPOS = {
    "heavy_4omini": "barc0/100k-gpt4omini-description-gpt4omini-code_generated_problems",
    "heavy_gpt4":   "barc0/100k-gpt4-description-gpt4omini-code_generated_problems",
    "heavy_200k":   "barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
}
# Each repo is one big JSONL named differently; resolve URL is repo/resolve/main/<file>.
FILES = {
    "heavy_4omini": "100k_gpt4o-mini_generated_problems.jsonl",
    "heavy_gpt4":   "100k_gpt4_generated_problems.jsonl",          # verify exact name via the tree API on Kaggle
    "heavy_200k":   "200k_HEAVY_gpt4o_generated_problems.jsonl",   # ^ same
}


def stream_barc(which="heavy_4omini", limit=None, timeout=120):
    """Yield raw records {source, examples, seeds} by streaming the remote JSONL. Memory = one line at a time."""
    url = f"https://huggingface.co/datasets/{REPOS[which]}/resolve/main/{FILES[which]}"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        buf = b""
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                n += 1
                if limit and n >= limit:
                    return


def parse_concepts(source):
    """Pull the relation label from the program header: returns (concepts_str, description_str)."""
    lines = source.splitlines()
    concepts = description = ""
    for i, ln in enumerate(lines):
        if ln.strip().startswith("# concepts") and i + 1 < len(lines):
            concepts = lines[i + 1].lstrip("# ").strip()
        if ln.strip().startswith("# description") and i + 1 < len(lines):
            j = i + 1
            ds = []
            while j < len(lines) and lines[j].lstrip().startswith("#"):
                ds.append(lines[j].lstrip("# ").strip()); j += 1
            description = " ".join(ds).strip()
    return concepts, description


def record_to_task(rec, n_train=4, seed=0, max_dim=30):
    """Turn ONE BARC record into our (train, test) format: lists of (input_grid, output_grid) int arrays.
    BARC packs many demo pairs (6..50) all solved by the SAME program. We split: first n_train -> train demos,
    the rest -> test. Returns (train, test, meta) or None if malformed / oversized."""
    pairs = rec.get("examples") or []
    out = []
    for p in pairs:
        if len(p) != 2:
            return None
        a = np.asarray(p[0], dtype=int); b = np.asarray(p[1], dtype=int)
        if a.ndim != 2 or b.ndim != 2:
            return None
        if max(a.shape + b.shape) > max_dim:
            return None
        if a.min() < 0 or a.max() > 9 or b.min() < 0 or b.max() > 9:
            return None
        out.append((a, b))
    if len(out) < n_train + 1:
        return None
    rng = random.Random(seed)
    rng.shuffle(out)
    train, test = out[:n_train], out[n_train:]
    concepts, description = parse_concepts(rec.get("source", ""))
    meta = {"concepts": concepts, "description": description,
            "program": rec.get("source", ""), "seeds": rec.get("seeds", [])}
    return train, test, meta


def load_barc_tasks(which="heavy_4omini", limit=200, n_train=4):
    """Streaming task generator in the learner's (tid, train, test) shape (+meta with the relation program)."""
    for i, rec in enumerate(stream_barc(which, limit=limit)):
        t = record_to_task(rec, n_train=n_train, seed=i)
        if t is None:
            continue
        train, test, meta = t
        yield f"{which}:{i}", train, test, meta


def main():
    print("Streaming a 50-record BARC ARC-Heavy sample (no bulk download)...")
    n = 0; concept_counts = {}
    for tid, train, test, meta in load_barc_tasks("heavy_4omini", limit=50):
        n += 1
        for c in [c.strip() for c in meta["concepts"].split(",") if c.strip()]:
            concept_counts[c] = concept_counts.get(c, 0) + 1
    print(f"parsed {n} tasks; top concepts:")
    for c, k in sorted(concept_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k:3d}  {c}")


if __name__ == "__main__":
    main()
