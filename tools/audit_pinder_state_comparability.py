#!/usr/bin/env python3
"""Compare current PINDER/pKAI targets with the earlier pKPDB/PypKa validation."""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from pkanet.model import PKPDB_PK_MOD
from jaxpropka.parameters import GROUPS
from pkatrain.gqt_paired_pinder import PairedMMap


runtime = Path(os.environ["PKABENCH_RUNTIME"])
base = runtime / "training/ogqt-pinder-factorial-v1"
manifest = json.loads((base / "manifest.json").read_text())
store = PairedMMap(Path(os.environ.get("PKATRAIN_PAIRED_MMAP", base / "mmap-v1")), manifest["records"])

state = []
state_by_branch = [[], []]
paired = []
paired_interface = []
by_group = defaultdict(list)
for record in manifest["records"]:
    if record["split"] != "val":
        continue
    raw = store.raw(record["id"])
    groups = np.asarray(raw["query_group"], dtype=np.int64)
    expected = np.asarray(raw["targets"], dtype=np.float64) - np.asarray(PKPDB_PK_MOD)[groups][None, :]
    values = np.abs(expected)
    state.extend(values.ravel().tolist())
    for branch in range(2):
        state_by_branch[branch].extend(values[branch].tolist())
    delta = np.abs(expected[0] - expected[1])
    paired.extend(delta.tolist())
    paired_interface.extend(delta[np.asarray(raw["interface"], dtype=bool)].tolist())
    for group, branch_values in zip(groups, values.T):
        by_group[int(group)].extend(branch_values.tolist())
store.close()

previous_path = runtime / "pretraining/gqt-site-weighting-v1/baseline/seed-17/validation_predictions.csv"
previous_target = []
previous_error = []
previous_by_group_target = defaultdict(list)
previous_by_group_error = defaultdict(list)
with previous_path.open(newline="") as stream:
    for row in csv.DictReader(stream):
        target = float(row["teacher_shift"])
        error = float(row["predicted_shift"]) - target
        previous_target.append(abs(target))
        previous_error.append(abs(error))
        previous_by_group_target[row["group"]].append(abs(target))
        previous_by_group_error[row["group"]].append(abs(error))

result = {
    "current_pinder_pkai": {
        "sites": len(paired),
        "state_targets": len(state),
        "pkmod_baseline_state_micro_mae": float(np.mean(state)),
        "pkmod_baseline_ab_micro_mae": float(np.mean(state_by_branch[0])),
        "pkmod_baseline_free_micro_mae": float(np.mean(state_by_branch[1])),
        "pkmod_baseline_state_group_macro_mae": float(np.mean([np.mean(v) for v in by_group.values()])),
        "zero_binding_shift_baseline_micro_mae": float(np.mean(paired)),
        "zero_binding_shift_interface_micro_mae": float(np.mean(paired_interface)),
        "groups": len(by_group),
        "site_counts_by_group": {GROUPS[group]: len(values) // 2 for group, values in by_group.items()},
    },
    "previous_pkpdb_pypka": {
        "sites": len(previous_target),
        "pkmod_baseline_micro_mae": float(np.mean(previous_target)),
        "pkmod_baseline_group_macro_mae": float(np.mean([np.mean(v) for v in previous_by_group_target.values()])),
        "orientation_seed17_micro_mae": float(np.mean(previous_error)),
        "orientation_seed17_group_macro_mae": float(np.mean([np.mean(v) for v in previous_by_group_error.values()])),
        "groups": len(previous_by_group_target),
        "site_counts_by_group": {group: len(values) for group, values in previous_by_group_target.items()},
    },
}
shared_groups = {GROUPS[group] for group in by_group}
shared_target = [value for group, values in previous_by_group_target.items() if group in shared_groups for value in values]
shared_error = [value for group, values in previous_by_group_error.items() if group in shared_groups for value in values]
result["previous_restricted_to_current_groups"] = {
    "groups": sorted(shared_groups),
    "sites": len(shared_target),
    "pkmod_baseline_micro_mae": float(np.mean(shared_target)),
    "orientation_seed17_micro_mae": float(np.mean(shared_error)),
}
destination = base / "comparability-audit.json"
pending = destination.with_suffix(".json.tmp")
pending.write_text(json.dumps(result, indent=2) + "\n")
pending.replace(destination)
print(json.dumps(result, indent=2), flush=True)
