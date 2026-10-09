"""Randomized-Jacobian attribution of oGQT auxiliary heads to graph features."""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pkanet.ogqt import initialize_auxiliary, predict_multi
from pkabench.runtime import atomic_json, require_compute
from pkatrain.gqt_auxiliary_pilot import AuxiliaryEngine, _prefetched
from pkatrain.gqt_paired_pinder import Loader, _bucket_n
from pkatrain.trainer import load_checkpoint

SEED = 17
N_COMPLEXES = 50
PROBES = 3
GROUPS = {
    "residue_identity": ("nodes", 0, 20, "node_mask"),
    "terminal_flags": ("nodes", 20, 22, "node_mask"),
    "disulfide_flag": ("nodes", 22, 23, "node_mask"),
    "frame_valid": ("nodes", 23, 24, "node_mask"),
    "residue_distance": ("edge", 0, 16, "edge_mask"),
    "residue_direction": ("edge", 16, 19, "edge_mask"),
    "residue_same_chain": ("edge", 19, 20, "edge_mask"),
    "site_distance": ("site_edge", 0, 16, "site_edge_mask"),
    "site_direction": ("site_edge", 16, 22, "site_edge_mask"),
    "site_orientation": ("site_edge", 22, 31, "site_edge_mask"),
    "site_same_chain": ("site_edge", 31, 32, "site_edge_mask"),
    "site_same_residue": ("site_edge", 32, 33, "site_edge_mask"),
    "site_sequence_separation": ("site_edge", 33, 34, "site_edge_mask"),
}


def read(path):
    with Path(path).open() as stream: return json.load(stream)


def _gradient_function(head, branch):
    def objective(edge, site_edge, nodes, graphs, signs, active):
        changed = {**graphs, "edge": edge, "site_edge": site_edge, "nodes": nodes}
        batch, branches = nodes.shape[:2]
        flat = jax.tree.map(lambda value: value.reshape((batch*branches,) + value.shape[2:]), changed)
        prediction = jax.vmap(predict_multi, in_axes=(None, 0))(PARAMS, flat)[head].reshape(batch, branches, -1)
        return jnp.sum(prediction[:, branch] * signs * active) / jnp.sqrt(jnp.maximum(active.sum(), 1))
    return jax.jit(jax.grad(objective, argnums=(0, 1, 2)))


PARAMS = None


def run(runtime):
    global PARAMS
    require_compute(threads=8, gpu_benchmark=True, allow_comp1400=True)
    runtime = Path(runtime); base = runtime / "training/ogqt-auxiliary-pilot-v1"; manifest = read(base / "manifest.json")
    validation = [r for r in manifest["records"] if r["split"] == "val"]
    rank = lambda row: hashlib.sha256(f"aux-gradient-v1|{row['id']}".encode()).hexdigest()
    records = sorted(validation, key=rank)[:N_COMPLEXES]; grouped = defaultdict(list)
    for record in records: grouped[_bucket_n(record["n"])].append(record["id"])
    plans = []
    for bucket in sorted(grouped, key=int):
        size = manifest["batch_sizes"][bucket]; ids = grouped[bucket]
        plans.extend(ids[i:i+size] for i in range(0, len(ids), size))
    params = initialize_auxiliary(jax.random.PRNGKey(SEED), **manifest["architecture"])
    engine = AuxiliaryEngine(params, 0, {"burial": 1, "interface": 1}); state = engine.optimizer.init(params)
    verification = read(base / "standard/seed-17/verification.json")
    PARAMS, _, _ = load_checkpoint(base / "standard/seed-17/checkpoints" /
        f"epoch-{verification['selected_epoch']:03d}", (params, state))
    functions = {"burial": _gradient_function("burial", 1), "interface": _gradient_function("interface", 0)}
    totals = {head: {name: {"gradient_square": 0.0, "gradient_input_square": 0.0, "count": 0}
        for name in GROUPS} for head in functions}
    loader = Loader(base, manifest)
    for number, (ids, batch) in enumerate(_prefetched(loader, plans), 1):
        graphs, _, active, _, _, _ = batch; device = jax.tree.map(jnp.asarray, graphs); active_device = jnp.asarray(active)
        for probe in range(PROBES):
            seed = int.from_bytes(hashlib.sha256(f"{SEED}|{number}|{probe}".encode()).digest()[:8], "little")
            signs = np.random.default_rng(seed).choice(np.asarray([-1., 1.], np.float32), size=active.shape) * active
            for head, function in functions.items():
                gradients = function(device["edge"], device["site_edge"], device["nodes"], device,
                    jnp.asarray(signs), active_device)
                values = {"edge": np.asarray(gradients[0]), "site_edge": np.asarray(gradients[1]),
                          "nodes": np.asarray(gradients[2])}
                branch = 1 if head == "burial" else 0
                for name, (field, start, stop, mask_name) in GROUPS.items():
                    value = np.asarray(graphs[field])[:, branch, ..., start:stop]
                    gradient = values[field][:, branch, ..., start:stop]
                    mask = np.asarray(graphs[mask_name])[:, branch]
                    expanded = np.broadcast_to(mask[..., None], gradient.shape)
                    selected_gradient = gradient[expanded]; selected_value = value[expanded]
                    item = totals[head][name]
                    item["gradient_square"] += float(np.sum(selected_gradient**2))
                    item["gradient_input_square"] += float(np.sum((selected_gradient*selected_value)**2))
                    item["count"] += int(len(selected_gradient))
        if number % 10 == 0: print(json.dumps({"batches": number, "total": len(plans)}), flush=True)
    loader.close(); results = {}
    for head, groups in totals.items():
        results[head] = {}
        total_energy = sum(x["gradient_input_square"] for x in groups.values())
        for name, values in groups.items():
            results[head][name] = {"scalar_observations": values["count"],
                "rms_derivative": float(np.sqrt(values["gradient_square"]/max(values["count"], 1))),
                "rms_gradient_times_input": float(np.sqrt(values["gradient_input_square"]/max(values["count"], 1))),
                "gradient_times_input_energy_share": values["gradient_input_square"]/max(total_energy, 1e-30)}
    output = base / "auxiliary-gradient-attribution-v1"; output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "summary.json", {"version": "auxiliary-gradient-attribution-v1",
        "complexes": len(records), "probes_per_batch": PROBES, "checkpoint_epoch": verification["selected_epoch"],
        "method": "Rademacher randomized-Jacobian probes; head-specific local input gradients",
        "interpretation": "gradient-times-input shares are local sensitivity, not causal importance; use with permutation audit",
        "test_data_included": False, "results": results})


if __name__ == "__main__": run(Path(os.environ["PKABENCH_RUNTIME"]))
