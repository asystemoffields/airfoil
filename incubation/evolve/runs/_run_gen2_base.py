#!/usr/bin/env python3
import sys, os, json, time
HERE = "/data/Windows-files/Documents/airfoil/incubation/evolve"
sys.path.insert(0, HERE)
import harness

BUDGET = int(sys.argv[1]) if len(sys.argv) > 1 else 6000

solver = harness.load_solver(os.path.join(HERE, "cand", "gen2_base.py"))
print("build_sec:", getattr(solver, "_BUILD_SEC", None), flush=True)

out = {}
for split in ("arc1-train", "arc1-eval"):
    tasks = harness.load_split(split)
    t0 = time.time()
    m, rows = harness.evaluate(solver, tasks, BUDGET, label="gen2_base/" + split, verbose=True)
    solved_ids = sorted([r["task_id"] for r in rows if r["solved"]])
    missed_ids = sorted([r["task_id"] for r in rows if not r["solved"]])
    out[split] = {"metrics": m, "solved_ids": solved_ids, "missed_ids": missed_ids}
    print(split, "->", m, "(%.0fs)" % (time.time() - t0), flush=True)

res = {
    "budget": BUDGET,
    "arc1_train": {"solved": out["arc1-train"]["solved_ids"],
                   "missed": out["arc1-train"]["missed_ids"],
                   "metrics": out["arc1-train"]["metrics"]},
    "arc1_eval": {"solved": out["arc1-eval"]["solved_ids"],
                  "missed": out["arc1-eval"]["missed_ids"],
                  "metrics": out["arc1-eval"]["metrics"]},
}
dest = os.path.join(HERE, "runs", "gen2_base_solved.json")
json.dump(res, open(dest, "w"), indent=1)
print("WROTE", dest, flush=True)
print("TRAIN solved:", len(res["arc1_train"]["solved"]),
      "EVAL solved:", len(res["arc1_eval"]["solved"]), flush=True)
