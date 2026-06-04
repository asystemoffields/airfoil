import sys, json, time
sys.path.insert(0,'/data/Windows-files/Documents/airfoil/incubation/evolve')
import invention_gate as IG, harness
sol=IG.load_solver('/data/Windows-files/Documents/airfoil/incubation/evolve/cand/gen6_03_object-movement.py')
out={}
for sp in ('arc1-train','arc1-eval'):
    m=IG.evaluate_invention(sol, harness.load_split(sp), 4000, log=False)
    beyond=[p['task_id'] for p in []]  # filled below
    # re-run to get per_task via log; evaluate_invention returns metrics only, so call internals:
print("use the FINAL json from main run")
