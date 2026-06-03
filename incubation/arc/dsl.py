#!/usr/bin/env python3
"""ARC grounding — a small grid DSL + exact verifier + task loader.
Primitives are pure grid->grid (numpy int2d). Color-parametric ops are instantiated with colors PRESENT in
the task, keeping the per-task arg space small. A PROGRAM = list of (op_name, args). Verify = applies to all
train inputs and matches train outputs EXACTLY (shape+values). Run/imported with /data/llm/.venv/bin/python."""
import json, glob
import numpy as np

TRAIN_DIR = "/data/arc/data/training"
EVAL_DIR = "/data/arc/data/evaluation"


def load_task(path):
    d = json.load(open(path))
    g = lambda pr: (np.array(pr["input"], int), np.array(pr["output"], int))
    return [g(p) for p in d["train"]], [g(p) for p in d["test"]]


def load_all(d=TRAIN_DIR, n=None):
    paths = sorted(glob.glob(d + "/*.json"))
    if n: paths = paths[:n]
    return [(p.split("/")[-1][:8], *load_task(p)) for p in paths]


# ---------- primitives (each grid->grid) ----------
def identity(g): return g
def reflect_h(g): return g[:, ::-1]
def reflect_v(g): return g[::-1, :]
def rot90(g): return np.rot90(g, 1)
def rot180(g): return np.rot90(g, 2)
def rot270(g): return np.rot90(g, 3)
def transpose(g): return g.T
def tile_h2(g): return np.concatenate([g, g], 1)
def tile_v2(g): return np.concatenate([g, g], 0)
def tile_2x2(g): return np.concatenate([np.concatenate([g, g], 1)] * 2, 0)
def scale2(g): return np.kron(g, np.ones((2, 2), int))


def crop_content(g):
    nz = np.argwhere(g != 0)
    if nz.size == 0: return g
    (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
    return g[r0:r1, c0:c1]


def gravity_down(g):
    out = np.zeros_like(g)
    for c in range(g.shape[1]):
        col = g[:, c]; vals = col[col != 0]
        if len(vals): out[g.shape[0] - len(vals):, c] = vals
    return out


def _bg(g):
    v, ct = np.unique(g, return_counts=True); return int(v[ct.argmax()])


def fill_holes(g, color):                                   # fill 0-cells not connected to the border with `color`
    from collections import deque
    h, w = g.shape; bg = (g == 0); reach = np.zeros_like(bg)
    q = deque()
    for i in range(h):
        for j in (0, w - 1):
            if bg[i, j] and not reach[i, j]: reach[i, j] = True; q.append((i, j))
    for j in range(w):
        for i in (0, h - 1):
            if bg[i, j] and not reach[i, j]: reach[i, j] = True; q.append((i, j))
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w and bg[a, b] and not reach[a, b]: reach[a, b] = True; q.append((a, b))
    out = g.copy(); out[bg & ~reach] = color; return out


def recolor(g, a, b):
    out = g.copy(); out[g == a] = b; return out


def largest_object(g):                                      # keep only the largest 4-conn nonzero component
    from collections import deque
    h, w = g.shape; seen = np.zeros_like(g, bool); best = None; bestsz = 0
    for i in range(h):
        for j in range(w):
            if g[i, j] != 0 and not seen[i, j]:
                comp = []; q = deque([(i, j)]); seen[i, j] = True
                while q:
                    a, b = q.popleft(); comp.append((a, b))
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + di, b + dj
                        if 0 <= x < h and 0 <= y < w and g[x, y] != 0 and not seen[x, y]: seen[x, y] = True; q.append((x, y))
                if len(comp) > bestsz: bestsz = len(comp); best = comp
    out = np.zeros_like(g)
    if best:
        for a, b in best: out[a, b] = g[a, b]
    return out


# op registry: name -> (fn, n_color_args)  (color args instantiated from task palette)
OPS = {
    "identity": (identity, 0), "reflect_h": (reflect_h, 0), "reflect_v": (reflect_v, 0),
    "rot90": (rot90, 0), "rot180": (rot180, 0), "rot270": (rot270, 0), "transpose": (transpose, 0),
    "tile_h2": (tile_h2, 0), "tile_v2": (tile_v2, 0), "tile_2x2": (tile_2x2, 0), "scale2": (scale2, 0),
    "crop_content": (crop_content, 0), "gravity_down": (gravity_down, 0), "largest_object": (largest_object, 0),
    "fill_holes": (fill_holes, 1), "recolor": (recolor, 2),
}
OP_NAMES = list(OPS)


def apply_prog(g, prog):
    try:
        for name, args in prog:
            g = OPS[name][0](g, *args)
        return g
    except Exception:
        return None


def solves(prog, train):
    for gi, go in train:
        out = apply_prog(gi, prog)
        if out is None or out.shape != go.shape or not np.array_equal(out, go):
            return False
    return True


def palette(train):                                         # colors present across the task (for arg instantiation)
    cs = set()
    for gi, go in train:
        cs |= set(np.unique(gi).tolist()) | set(np.unique(go).tolist())
    return sorted(cs)
