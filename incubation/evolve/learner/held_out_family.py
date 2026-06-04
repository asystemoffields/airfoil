#!/usr/bin/env python3
"""Vine RL loop — leave-one-FAMILY-out splitter (brick 2). The headline falsifier metric is held-out-FAMILY
certified-invented solve-rate: train (distill) on N-1 relation families, test on a family NEVER in the distill
set. This is the honest generalization gate -- it forbids overfit-to-the-solved-set and forces the policy to have
learned WHEN to reach for WHICH kind of concept, not memorized a family. Each family needs a DIFFERENTLY-earned
concept (relational sense, substrate sense, or motor gesture), so cross-family transfer is genuine recombination."""
from grow_library import make_containment_task, make_contained_in_largest_task
from policy_eval import make_adjacency_task
from substrate_eye import make_share_height_task
from effect_faculty import make_align_task, make_copy_task

FAMILIES = {
    "containment":          make_containment_task,        # fixed relational channel
    "adjacency":            make_adjacency_task,           # fixed relational channel (other channel)
    "share-height":         make_share_height_task,        # SUBSTRATE sense (self-evolved eye)
    "align":                make_align_task,               # MOVE gesture (motor hand)
    "copy":                 make_copy_task,                # COPY gesture (motor hand)
    "contained-in-largest": make_contained_in_largest_task,  # COMPOSED relation (containment o is-largest)
}


def leave_one_out(held_out):
    """-> (distill_families dict, held_out_family dict). The policy distills on `distill`, is scored on `held_out`."""
    assert held_out in FAMILIES, held_out
    distill = {k: v for k, v in FAMILIES.items() if k != held_out}
    return distill, {held_out: FAMILIES[held_out]}


def all_splits():
    return [leave_one_out(h) for h in FAMILIES]


if __name__ == "__main__":
    print(f"families ({len(FAMILIES)}): {list(FAMILIES)}")
    for h in FAMILIES:
        d, ho = leave_one_out(h)
        print(f"  held-out '{h}': distill on {list(d)}")
    print("READ: held-out-FAMILY certified-invented solve-rate (distill on N-1, score on the never-seen family) is "
          "the falsifier metric -- forbids overfit, demands the policy learned WHICH kind of concept WHEN.")
