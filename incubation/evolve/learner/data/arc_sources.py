#!/usr/bin/env python3
"""Unified SCALE-phase relation-corpus loader for Branch-B.

Every source yields the learner's task contract, identical to ground_conceptarc.py:

    (task_id: str, train: list[(in, out)], test: list[(in, out)])

where each grid is a numpy int array, <=30x30, values 0-9. PROGRAM-bearing sources
(BARC, RE-ARC verifiers) additionally carry the relation program via load_*_meta /
the 4-tuple variants — that program/concept text is the *relation curriculum gold*
for training a generative proposer (see DATA.md).

DESIGN (which sources, why):
  - BARC ARC-Heavy (barc0/*)  -> ~274k LLM tasks, EACH with a Python solver + `# concepts:`
        + `# description:` NL label. The single richest (relation-program, demos) corpus.
        Imperative numpy, NOT in Branch-B's grammar -> use as a diversity / relation-mining
        pool, not drop-in labels. STREAMED line-by-line (the file is ~3GB; never download whole).
  - RE-ARC (michaelhodel/re-arc via HF mirror) -> 400 ARC-1 relations, each a PROCEDURAL
        generator + an executable DSL verifier program. UNLIMITED CPU-cheap instances per
        relation with a difficulty knob -> the strongest *curriculum* lever on a 7GB box.
  - ConceptARC (local) -> 160 human concept-labeled tasks. Held-out eval distribution.
  - ARC-1 / ARC-2 (local) -> the real targets; tiny, no programs. Held-out gate.

RESOURCE DISCIPLINE (7GB box): every large source here is LAZY/STREAMING and bounded by a
`limit`. Nothing pulls a 100k-task corpus into RAM. Plan the FULL pull on Kaggle (see DATA.md).

Verified loaders:
  /data/llm/.venv/bin/python data/arc_sources.py        # runs every source on a small sample
"""
import json, os, sys, types, urllib.request, random
import numpy as np

# ----------------------------------------------------------------------------
# Paths / repo ids (filenames VERIFIED 2026-06-04 against the HF tree API).
# ----------------------------------------------------------------------------
CONCEPTARC_CH = "/data/conceptarc/arc-agi_concept-challenges.json"
CONCEPTARC_SO = "/data/conceptarc/arc-agi_concept-solutions.json"
ARC1_DIR = "/data/arc/data"            # training/ + evaluation/
ARC2_DIR = "/data/arc-agi-2/data"      # training/ + evaluation/

# BARC ARC-Heavy: repo -> exact resolve filename (CONFIRMED via HfApi.list_repo_files).
BARC_REPOS = {
    "heavy_4omini": ("barc0/100k-gpt4omini-description-gpt4omini-code_generated_problems",
                     "100k_gpt4o-mini_generated_problems.jsonl"),
    "heavy_gpt4":   ("barc0/100k-gpt4-description-gpt4omini-code_generated_problems",
                     "100k-gpt4-description-gpt4omini-code_generated_problems.jsonl"),
    "heavy_200k":   ("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
                     "data_100k.jsonl"),
    "heavy_200k_suggestfn": ("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems",
                             "data_suggestfunction_100k.jsonl"),
}

REARC_REPO = "Alignment-Lab-AI/ARC-stuff"
REARC_TASK_TMPL = "re-arc/re_arc (1)/re_arc/tasks/{tid}.json"  # literal space + parens — quote exactly
REARC_CODE_FILES = ["dsl.py", "utils.py", "generators.py", "verifiers.py", "main.py"]


def _g(grid):
    return np.asarray(grid, dtype=int)


# ============================================================================
# (1) ConceptARC  — local, human concept-labeled. Reuses ground_conceptarc.py pattern.
# ============================================================================
def load_conceptarc():
    """Yield (tid, train, test). concept label = alpha prefix of tid (see DATA.md)."""
    ch = json.load(open(CONCEPTARC_CH)); so = json.load(open(CONCEPTARC_SO))
    for tid in ch:
        train = [(_g(p["input"]), _g(p["output"])) for p in ch[tid]["train"]]
        test = [(_g(t["input"]), _g(so[tid][i])) for i, t in enumerate(ch[tid]["test"])]
        yield tid, train, test


# ============================================================================
# (2) ARC-1 / ARC-2  — local standard ARC JSON (train/test, outputs present).
# ============================================================================
def load_arc_dir(base_dir, split="training", limit=None):
    """Stream local ARC tasks (one .json each). split in {"training","evaluation"}."""
    d = os.path.join(base_dir, split)
    files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
    for n, fn in enumerate(files):
        if limit and n >= limit:
            return
        obj = json.load(open(os.path.join(d, fn)))
        tid = fn[:-5]
        train = [(_g(p["input"]), _g(p["output"])) for p in obj["train"]]
        test = [(_g(p["input"]), _g(p["output"])) for p in obj["test"]]
        yield f"{split}/{tid}", train, test


def load_arc1(split="training", limit=None):
    return load_arc_dir(ARC1_DIR, split, limit)


def load_arc2(split="training", limit=None):
    return load_arc_dir(ARC2_DIR, split, limit)


# ============================================================================
# (3) BARC ARC-Heavy  — PROGRAM + precomputed examples. The relation-program gold.
#     STREAMED line-by-line over HTTP (remote) or from a local JSONL (Kaggle).
# ============================================================================
def parse_barc_concepts(source):
    """Extract (concepts, description) NL relation labels from a BARC program header."""
    lines = source.splitlines()
    concepts = ""; desc_parts = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("# concepts") and i + 1 < len(lines):
            concepts = lines[i + 1].lstrip("# ").strip()
        if s.startswith("# description"):
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("#"):
                desc_parts.append(lines[j].lstrip("# ").strip()); j += 1
    return concepts, " ".join(desc_parts).strip()


def barc_record_to_task(rec, tid, n_train=4, seed=0, max_dim=30, validate=True):
    """One BARC record {source, examples, seeds} -> (tid, train, test, meta) or None.

    BARC packs 6..50 demo pairs ALL solved by the same program (no built-in split):
    we shuffle (per-record seed) then carve first n_train as train demos, rest as test.
    `meta.program` is the executable relation; meta.concepts/description the NL label.
    Filters degenerate/oversized tasks when validate=True (LLM data is only auto-filtered).
    """
    pairs = rec.get("examples") or []
    out = []
    for p in pairs:
        if len(p) != 2:
            return None
        a = _g(p[0]); b = _g(p[1])
        if a.ndim != 2 or b.ndim != 2 or a.size == 0 or b.size == 0:
            return None
        if max(a.shape + b.shape) > max_dim:
            return None
        if validate and (a.min() < 0 or a.max() > 9 or b.min() < 0 or b.max() > 9):
            return None
        out.append((a, b))
    if len(out) < n_train + 1:
        return None
    if validate:
        # drop tasks where every output == its input (pure identity = no relation)
        if all(np.array_equal(a, b) for a, b in out):
            return None
    rng = random.Random(seed); rng.shuffle(out)
    train, test = out[:n_train], out[n_train:]
    concepts, description = parse_barc_concepts(rec.get("source", ""))
    meta = {"concepts": concepts, "description": description,
            "program": rec.get("source", ""), "seeds": rec.get("seeds", [])}
    return tid, train, test, meta


def _stream_barc_lines(which, limit, timeout):
    """Yield raw decoded JSON records by streaming the remote JSONL (1 line at a time)."""
    repo, fname = BARC_REPOS[which]
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fname}"
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
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                if limit and n >= limit:
                    return


def load_barc(which="heavy_4omini", limit=200, n_train=4, validate=True):
    """Stream BARC tasks remotely (no bulk download). Yields (tid, train, test, meta)."""
    seen = 0
    for i, rec in enumerate(_stream_barc_lines(which, limit=None, timeout=180)):
        t = barc_record_to_task(rec, f"{which}:{i}", n_train=n_train, seed=i, validate=validate)
        if t is None:
            continue
        yield t
        seen += 1
        if limit and seen >= limit:
            return


def load_barc_local(jsonl_path, limit=None, n_train=4, validate=True):
    """KAGGLE path: stream a LOCAL BARC JSONL line-by-line (constant RAM). 4-tuple w/ meta."""
    with open(jsonl_path) as f:
        seen = 0
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = barc_record_to_task(rec, f"barc/{i}", n_train=n_train, seed=i, validate=validate)
            if t is None:
                continue
            yield t
            seen += 1
            if limit and seen >= limit:
                return


# ============================================================================
# (4) RE-ARC  — procedural generators + DSL verifier programs. UNLIMITED CPU instances.
#     The strongest curriculum lever on the box (code files total <1MB, numpy only).
# ============================================================================
_REARC_CACHE = {"gens": None, "vers": None, "dir": None}


def _install_rearc_shims():
    """utils.py imports matplotlib only for a plotting helper we never call; shim it so
    generation runs without the heavy stack. tqdm is shimmed if absent."""
    if "matplotlib" not in sys.modules:
        mpl = types.ModuleType("matplotlib")
        plt = types.ModuleType("matplotlib.pyplot")
        colors = types.ModuleType("matplotlib.colors")
        colors.ListedColormap = colors.Normalize = object
        mpl.pyplot = plt; mpl.colors = colors
        sys.modules["matplotlib"] = mpl
        sys.modules["matplotlib.pyplot"] = plt
        sys.modules["matplotlib.colors"] = colors
    try:
        import tqdm  # noqa: F401
    except ImportError:
        t = types.ModuleType("tqdm"); t.tqdm = lambda x, *a, **k: x
        sys.modules["tqdm"] = t


def ensure_rearc_code(local_dir="/data/rearc_code"):
    """Download the small RE-ARC .py files (NOT re_arc.zip / not the 400 task JSONs) and
    put the re-arc dir on sys.path. Returns the dir containing the modules. Idempotent."""
    from huggingface_hub import hf_hub_download
    for f in REARC_CODE_FILES:
        hf_hub_download(REARC_REPO, f"re-arc/{f}", repo_type="dataset", local_dir=local_dir)
    code_dir = os.path.join(local_dir, "re-arc")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    _install_rearc_shims()
    return code_dir


def _rearc_fns(local_dir="/data/rearc_code"):
    if _REARC_CACHE["gens"] is None:
        ensure_rearc_code(local_dir)
        from main import get_generators, get_verifiers  # type: ignore
        _REARC_CACHE["gens"] = get_generators()
        _REARC_CACHE["vers"] = get_verifiers()
    return _REARC_CACHE["gens"], _REARC_CACHE["vers"]


def rearc_task_ids(local_dir="/data/rearc_code"):
    gens, _ = _rearc_fns(local_dir)
    return sorted(gens.keys())


def load_rearc_generated(task_ids=None, n_train=4, n_test=1, diff_lb=0.0, diff_ub=1.0,
                         seed=0, local_dir="/data/rearc_code", instances_per_task=1,
                         max_attempts=8, skip_failing=True):
    """UNLIMITED on-CPU generation. For each task id, sample `instances_per_task` fresh
    (train,test) instances at the given difficulty band. Yields (tid, train, test, program)
    where `program` is the SSA-style DSL verifier source (the relation curriculum gold).

    A handful of upstream generators raise on some diff draws (a known RE-ARC quirk, e.g.
    generate_11852cab). We retry each pair up to `max_attempts` times; if a relation still
    can't produce a full instance we skip it (skip_failing) rather than crash the stream."""
    import inspect
    gens, vers = _rearc_fns(local_dir)
    ids = task_ids or sorted(gens.keys())
    rng = random.Random(seed)
    for tid in ids:
        if tid not in gens:
            continue
        try:
            program = inspect.getsource(vers[tid]) if tid in vers else ""
        except (OSError, TypeError):
            program = ""
        for k in range(instances_per_task):
            pairs = []
            ok = True
            for _ in range(n_train + n_test):
                got = None
                for _att in range(max_attempts):
                    try:
                        ex = gens[tid](diff_lb, diff_ub)
                        got = (_g(ex["input"]), _g(ex["output"]))
                        break
                    except Exception:
                        continue
                if got is None:
                    ok = False
                    break
                pairs.append(got)
            if not ok:
                if skip_failing:
                    break  # this relation is flaky at this band; skip its remaining instances
                raise RuntimeError(f"RE-ARC generator {tid} failed after {max_attempts} attempts")
            rng.shuffle(pairs)
            train, test = pairs[:n_train], pairs[n_train:n_train + n_test]
            yield f"rearc/{tid}#{k}", train, test, program


def load_rearc_pregen_local(tasks_dir, n_train=4, n_test=1, seed=0, limit=None):
    """KAGGLE/offline path: carve (train,test) from pre-generated re_arc/tasks/<id>.json
    files (flat list of 1000 {input,output} per file, all one relation). No programs here;
    use load_rearc_generated for verifier programs."""
    files = sorted(f for f in os.listdir(tasks_dir) if f.endswith(".json"))
    rng = random.Random(seed)
    for n, fn in enumerate(files):
        if limit and n >= limit:
            return
        ex = json.load(open(os.path.join(tasks_dir, fn)))
        pairs = [(_g(e["input"]), _g(e["output"])) for e in ex]
        s = rng.sample(pairs, min(n_train + n_test, len(pairs)))
        yield f"rearc/{fn[:-5]}", s[:n_train], s[n_train:n_train + n_test]


# ============================================================================
# Saved-sample loader — re-read the on-disk JSONL samples (data/samples/*.jsonl)
# written by pull_sample.py, without re-streaming the source. Yields the same
# (tid, train, test, program) 4-tuple. Use for fast iteration on the box.
# ============================================================================
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def load_sample_file(path, limit=None):
    """Stream a saved sample JSONL -> (tid, train, test, program). grids re-cast to int."""
    if not os.path.isabs(path):
        path = os.path.join(SAMPLES_DIR, path)
    with open(path) as f:
        for n, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if limit and n >= limit:
                return
            r = json.loads(line)
            train = [(_g(a), _g(b)) for a, b in r["train"]]
            test = [(_g(a), _g(b)) for a, b in r["test"]]
            yield r["task_id"], train, test, r.get("program", "")


# ============================================================================
# Registry — uniform 3-tuple view (drops BARC/RE-ARC program meta) for code that
# only wants (tid, train, test). Program-aware code calls the load_* fns directly.
# ============================================================================
def iter_source(name, limit=200, **kw):
    """Uniform (tid, train, test) stream by source name; drops the 4th program element."""
    if name == "conceptarc":
        yield from load_conceptarc()
    elif name == "arc1":
        yield from load_arc1(limit=limit, **kw)
    elif name == "arc2":
        yield from load_arc2(limit=limit, **kw)
    elif name.startswith("barc:"):
        for tid, tr, te, _meta in load_barc(which=name.split(":", 1)[1], limit=limit, **kw):
            yield tid, tr, te
    elif name == "rearc":
        for tid, tr, te, _prog in load_rearc_generated(**kw):
            yield tid, tr, te
    else:
        raise ValueError(f"unknown source {name!r}")


if __name__ == "__main__":
    def shapes(train):
        return [tuple(t[0].shape) for t in train[:1]]

    print("== (1) ConceptARC (local, 160 human tasks) ==")
    cc = list(load_conceptarc())
    print(f"   tasks={len(cc)}  e.g. {cc[0][0]} train={len(cc[0][1])} test={len(cc[0][2])} in0={shapes(cc[0][1])}")

    print("== (2) ARC-1 / ARC-2 (local) ==")
    a1 = list(load_arc1("training"))
    a2 = list(load_arc2("training"))
    print(f"   ARC-1 training tasks={len(a1)}  e.g. {a1[0][0]} in0={shapes(a1[0][1])}")
    print(f"   ARC-2 training tasks={len(a2)}  e.g. {a2[0][0]} in0={shapes(a2[0][1])}")

    print("== (3) BARC ARC-Heavy (STREAMED remote, program gold) ==")
    bn = 0; concept_hist = {}
    for tid, tr, te, meta in load_barc("heavy_4omini", limit=30):
        bn += 1
        for c in (x.strip() for x in meta["concepts"].split(",") if x.strip()):
            concept_hist[c] = concept_hist.get(c, 0) + 1
        if bn == 1:
            print(f"   first: {tid} train={len(tr)} test={len(te)} in0={shapes(tr)} "
                  f"concepts='{meta['concepts'][:60]}' prog_lines={meta['program'].count(chr(10))}")
    print(f"   streamed+parsed {bn} tasks; top concepts: " +
          ", ".join(f"{c}({k})" for c, k in sorted(concept_hist.items(), key=lambda x: -x[1])[:6]))

    print("== (4) RE-ARC (generated on CPU, verifier programs) ==")
    rn = 0
    ids = rearc_task_ids()
    for tid, tr, te, prog in load_rearc_generated(task_ids=ids[:5], n_train=3, n_test=1,
                                                  diff_lb=0.0, diff_ub=0.5, instances_per_task=2):
        rn += 1
        if rn == 1:
            print(f"   first: {tid} train={len(tr)} test={len(te)} in0={shapes(tr)} "
                  f"prog_lines={prog.count(chr(10))}")
    print(f"   total RE-ARC relations available={len(ids)}; generated {rn} instances on CPU")
    print("ALL SOURCES OK.")
