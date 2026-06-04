#!/usr/bin/env python3
"""Vine — PERSISTENT LIBRARY: the earned vocabulary accumulates across runs/sessions instead of resetting.

Every earned item (feature key, relational predicate, substrate sense, composition, motor gesture) serializes to a
spec and back, so a growing library of concepts survives to disk and is reloaded next run. This is the substrate for
real experience-acquisition (the openness-to-experience thesis): the loop loads the library, tries its concepts
first, grows it from new invented solves, and saves. (Expert iteration -- the recognizer/policy LEARNING from this
accumulated experience -- is the next, deeper layer.) Run: /data/llm/.venv/bin/python persist_library.py"""
import json, os, sys
import rel_dsl as D
import substrate_eye as SE
import effect_faculty as EF


def serialize(p):
    if isinstance(p, D.FeatKey):   return {"t": "feat", "name": p.name}
    if isinstance(p, D.Composed):  return {"t": "comp", "ch": p.ch, "val": p.value, "mode": p.mode, "inner": serialize(p.inner)}
    if isinstance(p, D.Quantify):  return {"t": "quant", "ch": p.ch, "val": p.value, "mode": p.mode}
    if isinstance(p, SE.SubQuantify): return {"t": "sub", "pa": p.ch.pa, "op": p.ch.op, "pb": p.ch.pb, "mode": p.mode}
    if isinstance(p, EF.Gesture):  return {"t": "gesture", "target": serialize(p.target_pred), "mode": p.mode, "erase": p.do_erase}
    raise ValueError(f"cannot serialize {type(p)}")


def deserialize(s):
    t = s["t"]
    if t == "feat":    return D.FeatKey(s["name"])
    if t == "quant":   return D.Quantify(s["ch"], s["val"], s["mode"])
    if t == "sub":     return SE.SubQuantify(SE.SubChannel(s["pa"], s["op"], s["pb"]), s["mode"])
    if t == "comp":    return D.Composed(s["ch"], s["val"], deserialize(s["inner"]), s["mode"])
    if t == "gesture": return EF.Gesture(deserialize(s["target"]), s["mode"], s["erase"])
    raise ValueError(f"cannot deserialize {s}")


class Library:
    """a growing, persistent set of earned concepts (deduped). Survives across runs/sessions."""
    def __init__(self, path):
        self.path = path; self.specs = []; self.seen = set()
        if os.path.exists(path):
            self.specs = json.load(open(path))
            self.seen = {json.dumps(s, sort_keys=True) for s in self.specs}

    def add(self, concept):
        s = serialize(concept); k = json.dumps(s, sort_keys=True)
        if k in self.seen:
            return False
        self.seen.add(k); self.specs.append(s); return True

    def concepts(self): return [deserialize(s) for s in self.specs]
    def save(self): json.dump(self.specs, open(self.path, "w"))
    def __len__(self): return len(self.specs)


def _earned(prog):
    return getattr(prog, "key", prog)   # recolor -> its key; gesture -> itself


def _demo():
    from open_loop import open_solve
    from grow_library import make_containment_task
    from effect_faculty import make_align_task, make_copy_task
    PATH = "/tmp/vine_library.json"
    if os.path.exists(PATH): os.remove(PATH)

    fams = [("containment", make_containment_task), ("share-height", SE.make_share_height_task),
            ("align", make_align_task), ("copy", make_copy_task)]
    lib = Library(PATH)
    print("RUN 1 (cold) — accumulate earned concepts across a mixed stream:")
    for i in range(16):
        name, gen = fams[i % len(fams)]
        prog, kind, _c = open_solve(gen(4), gen(2))
        if prog is not None and lib.add(_earned(prog)):
            print(f"  +earned from {name:<12}: {_earned(prog)}")
    lib.save()
    print(f"  library size: {len(lib)} -> persisted to {PATH}")

    lib2 = Library(PATH)   # reload in a fresh process-equivalent
    ok = all(serialize(deserialize(s)) == s for s in lib2.specs)
    print(f"\nRUN 2 (warm) — reloaded library: {len(lib2)} concepts, round-trips clean: {ok}")
    print(f"  concepts: {[str(c) for c in lib2.concepts()]}")
    print("READ: the earned vocabulary (senses + gestures + compositions) ACCUMULATES across runs and survives to "
          "disk -> experience persists instead of resetting. Substrate for expert iteration (the recognizer LEARNING "
          "from this accumulated experience), the openness-to-experience endgame.")


if __name__ == "__main__":
    _demo()
