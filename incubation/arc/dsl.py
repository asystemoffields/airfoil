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


# ---------- richer primitives (object-centric + geometry) for a DEPTH regime ----------
def _shift(g, dr, dc):
    r = np.roll(np.roll(g, dr, 0), dc, 1)
    if dr > 0: r[:dr, :] = 0
    elif dr < 0: r[dr:, :] = 0
    if dc > 0: r[:, :dc] = 0
    elif dc < 0: r[:, dc:] = 0
    return r
def shift_up(g): return _shift(g, -1, 0)
def shift_down(g): return _shift(g, 1, 0)
def shift_left(g): return _shift(g, 0, -1)
def shift_right(g): return _shift(g, 0, 1)


def gravity_up(g): return gravity_down(g[::-1, :])[::-1, :]
def gravity_left(g): return gravity_down(g.T).T
def gravity_right(g): return gravity_up(g.T).T


def sym_lr(g):                                              # overlay with horizontal mirror (nonzero wins)
    out = g.copy(); m = g[:, ::-1]; out[out == 0] = m[out == 0]; return out
def sym_ud(g):
    out = g.copy(); m = g[::-1, :]; out[out == 0] = m[out == 0]; return out


def _components(g):
    from collections import deque
    h, w = g.shape; seen = np.zeros_like(g, bool); comps = []
    for i in range(h):
        for j in range(w):
            if g[i, j] != 0 and not seen[i, j]:
                comp = []; q = deque([(i, j)]); seen[i, j] = True
                while q:
                    a, b = q.popleft(); comp.append((a, b))
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + di, b + dj
                        if 0 <= x < h and 0 <= y < w and g[x, y] != 0 and not seen[x, y]: seen[x, y] = True; q.append((x, y))
                comps.append(comp)
    return comps


def keep_smallest(g):
    comps = _components(g)
    if not comps: return g
    sm = min(comps, key=len); out = np.zeros_like(g)
    for a, b in sm: out[a, b] = g[a, b]
    return out


def downscale2(g): return g[::2, ::2]
def trim_border(g): return g[1:-1, 1:-1] if g.shape[0] > 2 and g.shape[1] > 2 else g


def bbox_fill(g, color):
    nz = np.argwhere(g != 0)
    if nz.size == 0: return g
    (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1
    out = g.copy(); out[r0:r1, c0:c1] = color; return out


def outline(g, color):
    nz = np.argwhere(g != 0)
    if nz.size == 0: return g
    (r0, c0), (r1, c1) = nz.min(0), nz.max(0) + 1; out = g.copy()
    out[r0:r1, c0] = color; out[r0:r1, c1 - 1] = color; out[r0, c0:c1] = color; out[r1 - 1, c0:c1] = color
    return out


def swap_colors(g, a, b):
    out = g.copy(); out[g == a] = b; out[g == b] = a; return out
def keep_color(g, c):
    out = np.zeros_like(g); out[g == c] = c; return out
def remove_color(g, c):
    out = g.copy(); out[g == c] = 0; return out


# op registry: name -> (fn, n_color_args)  (color args instantiated from task palette)
OPS = {
    "identity": (identity, 0), "reflect_h": (reflect_h, 0), "reflect_v": (reflect_v, 0),
    "rot90": (rot90, 0), "rot180": (rot180, 0), "rot270": (rot270, 0), "transpose": (transpose, 0),
    "tile_h2": (tile_h2, 0), "tile_v2": (tile_v2, 0), "tile_2x2": (tile_2x2, 0), "scale2": (scale2, 0),
    "crop_content": (crop_content, 0), "gravity_down": (gravity_down, 0), "largest_object": (largest_object, 0),
    "shift_up": (shift_up, 0), "shift_down": (shift_down, 0), "shift_left": (shift_left, 0), "shift_right": (shift_right, 0),
    "gravity_up": (gravity_up, 0), "gravity_left": (gravity_left, 0), "gravity_right": (gravity_right, 0),
    "sym_lr": (sym_lr, 0), "sym_ud": (sym_ud, 0), "keep_smallest": (keep_smallest, 0),
    "downscale2": (downscale2, 0), "trim_border": (trim_border, 0),
    "fill_holes": (fill_holes, 1), "bbox_fill": (bbox_fill, 1), "outline": (outline, 1),
    "keep_color": (keep_color, 1), "remove_color": (remove_color, 1),
    "recolor": (recolor, 2), "swap_colors": (swap_colors, 2),
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
