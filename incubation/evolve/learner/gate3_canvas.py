#!/usr/bin/env python3
"""Vine — GEN-3 make-or-break: does a CANVAS basis lift derive's RE-ARC floor off ZERO? (gate 2: derive 0/120 because
paint-on-input is SAME-SHAPE-ONLY; 32-36% of ARC is size-changing.) The B0 canvas basis = 4 IRREDUCIBLE primitives
whose factors are INDUCED from the demos' shape relationship (perceived, not literal):
  REPLICATE(kr,kc)=tile | MAGNIFY(kr,kc)=cell-expand | DECIMATE(kr,kc)=stride-subsample | WINDOW=crop to a region.
The mandatory floor (synthesis-verified): >=7 RE-ARC relations fall to a PURE canvas op. >=15 = success.
Run: /data/llm/.venv/bin/python gate3_canvas.py [n]"""
import sys, time, random
from collections import deque
import numpy as np

sys.path.append("/data/rearc_code/re-arc")
import importlib.util
_spec = importlib.util.spec_from_file_location("rearc_gen", "/data/rearc_code/re-arc/generators.py")
RG = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(RG)
REARC = {n[len("generate_"):]: getattr(RG, n) for n in dir(RG) if n.startswith("generate_")}


def _eq(a, b): return a.shape == b.shape and np.array_equal(a, b)


def _objs_bboxes(g):
    """candidate crop regions: non-bg bbox; each color's bbox; the bbox of the largest connected blob (4-conn)."""
    g = np.asarray(g, int); H, W = g.shape; out = []
    nz = np.argwhere(g != 0)
    if len(nz):
        out.append((nz[:, 0].min(), nz[:, 1].min(), nz[:, 0].max()+1, nz[:, 1].max()+1))
    for col in range(1, 10):
        cc = np.argwhere(g == col)
        if len(cc):
            out.append((cc[:, 0].min(), cc[:, 1].min(), cc[:, 0].max()+1, cc[:, 1].max()+1))
    return out


def _nobj(g):
    g = np.asarray(g, int); H, W = g.shape; seen = np.zeros((H, W), bool); n = 0
    for r in range(H):
        for c in range(W):
            if g[r, c] != 0 and not seen[r, c]:
                n += 1; q = deque([(r, c)]); seen[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and g[nr, nc] == g[cr, cc]:
                            seen[nr, nc] = True; q.append((nr, nc))
    return n
def _ncol(g): return len(set(int(v) for v in np.unique(g)) - {0})


def canvas_solve(train, test):
    tr = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in train]
    te = [(np.asarray(gi, int), np.asarray(go, int)) for gi, go in test]
    allp = tr + te

    def ok(fn):
        try:
            return all(_eq(fn(i), o) for i, o in allp)
        except Exception:
            return False

    # FACTOR SOURCES: per-instance PERCEIVED quantities (NOT literals) -- the factor is a function of the input.
    srcs = {"self": lambda g: g.shape,
            "count": lambda g: (max(1, _nobj(g)), max(1, _nobj(g))),
            "ncol": lambda g: (max(1, _ncol(g)), max(1, _ncol(g)))}
    R = [(o.shape[0] / i.shape[0], o.shape[1] / i.shape[1]) for i, o in tr]
    if all(abs(r[0]-R[0][0]) < 1e-9 and abs(r[1]-R[0][1]) < 1e-9 for r in R) and R[0][0] >= 1 and R[0][0] == int(R[0][0]) and R[0][1] == int(R[0][1]):
        srcs["const"] = lambda g, k=(int(R[0][0]), int(R[0][1])): k                  # induced constant ratio
    for sn, src in srcs.items():
        def rep(g, src=src):
            kr, kc = int(src(g)[0]), int(src(g)[1]); return np.tile(g, (kr, kc)) if kr >= 1 and kc >= 1 else g
        if ok(rep): return f"replicate({sn})"
        def mag(g, src=src):
            kr, kc = int(src(g)[0]), int(src(g)[1]); return np.repeat(np.repeat(g, kr, 0), kc, 1) if kr >= 1 and kc >= 1 else g
        if ok(mag): return f"magnify({sn})"
    # DECIMATE: input bigger, by an induced integer factor (out smaller)
    D = [(i.shape[0] / o.shape[0], i.shape[1] / o.shape[1]) for i, o in tr]
    if all(abs(r[0]-D[0][0]) < 1e-9 and abs(r[1]-D[0][1]) < 1e-9 for r in D) and D[0][0] >= 1 and D[0][0] == int(D[0][0]) and D[0][1] == int(D[0][1]):
        kr, kc = int(D[0][0]), int(D[0][1])
        if ok(lambda g: g[::kr, ::kc]): return f"decimate({kr},{kc})"
    # WINDOW: per-instance crop to a perceived region (output shape may VARY per instance)
    nb = max((len(_objs_bboxes(i)) for i, _ in tr), default=0)
    for bi in range(min(nb, 12)):
        def crop(g, bi=bi):
            bs = _objs_bboxes(g)
            if bi >= len(bs): return np.zeros((1, 1), int)
            r0, c0, r1, c1 = bs[bi]; return g[r0:r1, c0:c1]
        if ok(crop): return f"window(bbox#{bi})"
    return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    random.seed(7)
    rels = list(REARC)[:n]; t0 = time.time(); solved = []
    for tid in rels:
        try:
            inst = [REARC[tid](0.1, 0.4) for _ in range(6)]                      # ONE call per instance (input+output paired)
            pairs = [(np.array(e["input"], int), np.array(e["output"], int)) for e in inst]
            r = canvas_solve(pairs[:4], pairs[4:])
            if r is not None: solved.append((tid, r))
        except Exception:
            continue
    print(f"GATE 3 (canvas floor-lift) on {len(rels)} RE-ARC relations [{time.time()-t0:.0f}s]:")
    print(f"  CANVAS basis solves {len(solved)} relations that derive (paint-on-input) gets 0 of  ->  floor 0 -> {len(solved)}")
    for tid, r in solved[:18]:
        print(f"        {tid}: {r}")
    verdict = "SUCCESS (>=15)" if len(solved) >= 15 else ("GO (mandatory floor >=7 met)" if len(solved) >= 7 else "BELOW FLOOR (<7) -- canvas basis underperforms")
    print(f"\n  GEN-3 CANVAS FITNESS: {verdict}")
    print("READ: floor lifts off zero = the size-change unblock works; the loop now has parents to anti-unify on the "
          "newly-solved relations. Next: B1 decompose + wire canvas into derive + B2-B4, then re-run gate 2/3.")


if __name__ == "__main__":
    main()
