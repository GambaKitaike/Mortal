#!/usr/bin/env python3
"""Stage2 launch gate (stage2_design.md §3): cumulative n_call_possible_aka_held
/ n_call_possible from action_mass events up to a given trainer_step cutoff.
Usage: check_stage2_launch_gate.py <ppo_diag.jsonl> <cutoff_step>
"""
import json
import sys

path = sys.argv[1]
cutoff = int(sys.argv[2])

n_call_possible = 0
n_call_possible_aka_held = 0
n_events = 0
max_step_seen = -1

with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("event") != "action_mass":
            continue
        step = d.get("trainer_step")
        if step is None or step > cutoff:
            continue
        n_call_possible += d.get("n_call_possible", 0)
        n_call_possible_aka_held += d.get("n_call_possible_aka_held", 0)
        n_events += 1
        if step > max_step_seen:
            max_step_seen = step

ratio = (n_call_possible_aka_held / n_call_possible) if n_call_possible else float("nan")
print(f"cutoff_step={cutoff} max_trainer_step_included={max_step_seen} n_action_mass_events={n_events}")
print(f"n_call_possible={n_call_possible} n_call_possible_aka_held={n_call_possible_aka_held}")
print(f"ratio={ratio:.4f} ({ratio*100:.2f}%)")
print(f"gate (>=54%): {'PASS' if ratio >= 0.54 else 'FAIL'}")
