#!/usr/bin/env python3
"""Render train pairs of given task ids as compact colored-digit grids for visual categorization."""
import sys, os, json
import numpy as np

TRAIN_DIR = "/data/arc/data/training"

def grid_str(g):
    return "\n".join("".join(str(int(v)) for v in row) for row in g)

def show(tid, max_pairs=4):
    d = json.load(open(os.path.join(TRAIN_DIR, tid + ".json")))
    print("="*70)
    print("TASK", tid, " ntrain=", len(d["train"]), " ntest=", len(d["test"]))
    for k, p in enumerate(d["train"][:max_pairs]):
        i = np.array(p["input"]); o = np.array(p["output"])
        print(f"--- pair {k}  in{i.shape} -> out{o.shape} ---")
        istr = grid_str(i).split("\n"); ostr = grid_str(o).split("\n")
        H = max(len(istr), len(ostr))
        wi = max((len(r) for r in istr), default=0)
        for r in range(H):
            li = istr[r] if r < len(istr) else ""
            lo = ostr[r] if r < len(ostr) else ""
            print(f"  {li.ljust(wi)}   |  {lo}")

if __name__ == "__main__":
    for tid in sys.argv[1:]:
        show(tid)
