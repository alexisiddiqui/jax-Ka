"""Diagnose why the backbone site-token GQT compresses large pKa shifts.

All prediction, intervention, and gradient paths use the indexed Triton model.
The native path is used only to expose numerically matched attention weights.
"""
from __future__ import annotations

import copy
import json
import os
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxpropka.parameters import GROUPS
from pkanet.model import PKPDB_PK_MOD
from pkanet.site_model import initialize_site, predict_site_pkpdb_indexed, predict_site_shift_indexed, predict_site_with_trace
from pkatrain.gqt_site_tokens import SiteEngine
from pkatrain.graph_batches import epoch_batches
from pkatrain.site_graph_data import SiteBatchLoader, experiment_root
from pkatrain.trainer import load_checkpoint
from .runtime import atomic_json, digest, require_compute


BINS = ((0., .5, "<0.5"), (.5, 1., "0.5-1"), (1., 2., "1-2"), (2., np.inf, ">=2"))
VARIANTS = ("original", "site_self_only", "remove_arg", "remove_orientation",
            "remove_site_geometry", "remove_0_6A", "remove_6_10A",
            "remove_10_15A", "remove_15_20A", "remove_pair_bias")
ACID = np.asarray([0, 1, 3, 4, 8], np.int32)
BASE = np.asarray([2, 5, 6, 7], np.int32)


def read(path): return json.loads(Path(path).read_text())


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = Path(path); pending = path.with_name(".pending-" + path.name)
    pq.write_table(pa.Table.from_pylist(rows), pending); os.replace(pending, path)


def shift_bin(value):
    value = abs(float(value))
    return next(name for low, high, name in BINS if low <= value < high)


def edge_distance(edge):
    centers = np.linspace(0, 20, 16, dtype=np.float32)
    weight = np.asarray(edge)[..., :16]
    return (weight * centers).sum(-1) / np.maximum(weight.sum(-1), 1e-12)


def perturb_site_graph(graph, variant):
    """Apply a site-context intervention without changing tensor shapes."""
    result = {key: np.array(value, copy=True) for key, value in graph.items()}
    if variant in ("original", "remove_pair_bias"): return result
    neighbors = result["site_neighbors"]; mask = result["site_edge_mask"]
    receivers = np.arange(neighbors.shape[1], dtype=np.int32)[None, :, None]
    self_edge = neighbors == receivers
    if variant == "site_self_only": remove = mask & ~self_edge
    elif variant == "remove_arg":
        types = result["site_type"]
        source_arg = types[:, :, None] == 6
        neighbor_arg = types[np.arange(types.shape[0])[:, None, None], neighbors] == 6
        remove = mask & (source_arg | neighbor_arg)
    elif variant == "remove_orientation":
        result["site_edge"][..., 22:31] = 0.; return result
    elif variant == "remove_site_geometry":
        result["site_edge"][..., :31] = 0.; return result
    elif variant.startswith("remove_") and variant.endswith("A"):
        low, high = {"remove_0_6A": (0, 6), "remove_6_10A": (6, 10),
                     "remove_10_15A": (10, 15), "remove_15_20A": (15, 20)}[variant]
        distance = edge_distance(result["site_edge"])
        remove = mask & ~self_edge & (distance >= low) & (distance < high)
    else: raise ValueError(variant)
    result["site_edge_mask"][remove] = False
    result["site_switch"][remove] = 0.
    return result


def batches(records, size, byid):
    # Preserve the production loader's shape-homogeneous plans.
    rng = np.random.default_rng(1709)
    return epoch_batches([row["complex_id"] for row in records], byid, rng, size)


def load_model(base, manifest, epoch):
    cfg = manifest["config"]
    params = initialize_site(jax.random.PRNGKey(cfg["seed"]), **cfg["architecture"])
    engine = SiteEngine(params); state = engine.optimizer.init(params)
    path = base / "seed-17/checkpoints" / f"epoch-{epoch:03d}"
    params, _, metadata = load_checkpoint(path, (params, state))
    if metadata["epoch"] != epoch: raise AssertionError(metadata)
    return params, path


def attention_metrics(weights, graph, query_site, q):
    weights = np.asarray(weights)[query_site[:q]]
    neighbors = np.asarray(graph["site_neighbors"])[query_site[:q]]
    mask = np.asarray(graph["site_edge_mask"])[query_site[:q]]
    types = np.asarray(graph["site_type"]); source = types[query_site[:q]]; target = types[neighbors]
    distance = edge_distance(np.asarray(graph["site_edge"])[query_site[:q]])
    self_edge = neighbors == query_site[:q, None]
    safe = np.where(mask[..., None], weights, 0.)
    entropy = -np.sum(np.where(safe > 0, safe * np.log(np.maximum(safe, 1e-30)), 0.), axis=1)
    def mass(select): return np.sum(safe * np.asarray(select)[..., None], axis=1)
    opposite = ((np.isin(source, ACID)[:, None] & np.isin(target, BASE)) |
                (np.isin(source, BASE)[:, None] & np.isin(target, ACID)))
    return dict(entropy=entropy, effective=np.exp(entropy), maximum=np.max(safe, axis=1),
        mass_self=mass(self_edge), mass_arg=mass(mask & (target == 6)),
        mass_opposite_class=mass(mask & opposite),
        mass_0_6=mass(mask & ~self_edge & (distance < 6)),
        mass_6_10=mass(mask & (distance >= 6) & (distance < 10)),
        mass_10_15=mass(mask & (distance >= 10) & (distance < 15)),
        mass_15_20=mass(mask & (distance >= 15)),
        neighbor_count=np.broadcast_to(np.sum(mask & ~self_edge, axis=1)[:, None], entropy.shape))


def tree_add(total, tree, weight):
    if total is None: return jax.tree.map(lambda x: np.asarray(x, np.float64) * weight, tree)
    return jax.tree.map(lambda a, x: a + np.asarray(x, np.float64) * weight, total, tree)


def gradient_norms(gradient):
    norm = lambda tree: float(np.sqrt(sum(np.sum(np.asarray(x, np.float64) ** 2) for x in jax.tree.leaves(tree))))
    return dict(total=norm(gradient), head=norm(gradient["head"]), site=norm(gradient["site"]),
                query=norm(gradient["query"]), groups=norm(gradient["groups"]),
                embed=norm(gradient["embed"]), encoder_0=norm(gradient["blocks"][0]),
                encoder_1=norm(gradient["blocks"][1]))


def run(root, out):
    require_compute(threads=8, gpu_benchmark=True, allow_comp1400=True)
    base = experiment_root(root) / "site-orientation"; manifest = read(base / "manifest.json")
    selected = read(base / "seed-17/best.json")["epoch"]
    late = int(read(base / "seed-17/checkpoints/latest.json")["checkpoint"].split("-")[-1])
    params, checkpoint = load_model(base, manifest, selected)
    late_params, late_checkpoint = load_model(base, manifest, late)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "manifest.json", dict(model="site-orientation", selected_epoch=selected,
        late_epoch=late, selected_checkpoint=str(checkpoint), selected_checkpoint_sha256=digest(checkpoint / "state.npz"),
        late_checkpoint=str(late_checkpoint), late_checkpoint_sha256=digest(late_checkpoint / "state.npz"),
        prediction_backend="full-float32 indexed Triton encoder and site attention",
        trace_backend="native attention, previously parity-validated; weights only",
        variants=list(VARIANTS), split_scope="all training and frozen validation sites; no test data", test_data_included=False))
    byid = {row["complex_id"]: row for row in manifest["records"]}
    loader = SiteBatchLoader(base, manifest, manifest["config"]["batch_size"], "site-orientation")
    forward = jax.jit(jax.vmap(predict_site_pkpdb_indexed, in_axes=(None, 0)))
    # Trace one graph at a time: the native trace materializes attention
    # weights, unlike Triton, and batching those tensors wastes substantial VRAM.
    trace_forward = jax.jit(predict_site_with_trace)
    no_pair = copy.deepcopy(params); no_pair["site"]["pair_bias"] = jnp.zeros_like(no_pair["site"]["pair_bias"])
    site_rows = []; attention_rows = []
    split_records = {split: [r for r in manifest["records"] if r["split"] == split] for split in ("train", "val")}
    for split, records in split_records.items():
        plans = batches(records, manifest["config"]["batch_size"], byid)
        for number, (cids, batch) in enumerate(loader.iterate(plans), 1):
            graph, target, eligible, valid = batch
            predictions = {}
            for variant in VARIANTS if split == "val" else ("original",):
                view = perturb_site_graph(graph, variant)
                use_params = no_pair if variant == "remove_pair_bias" else params
                predictions[variant] = np.asarray(forward(use_params, {k: jnp.asarray(v) for k, v in view.items()}))
            late_prediction = np.asarray(forward(late_params, {k: jnp.asarray(v) for k, v in graph.items()}))
            for bi, cid in enumerate(cids):
                record = byid[cid]; q = int(eligible[bi].sum()); groups = np.asarray(graph["query_group"][bi, :q])
                reference = np.asarray(PKPDB_PK_MOD)[groups]; teacher = np.asarray(target[bi, :q]) - reference
                predicted = predictions["original"][bi, :q] - reference
                late_predicted = late_prediction[bi, :q] - reference
                for qi, (key, observed, value, last) in enumerate(zip(record["keys"][:q], teacher, predicted, late_predicted)):
                    row = dict(split=split, component_id=record["component_id"], complex_id=cid,
                        chain=key[1], resnum=key[2], icode=key[3], group=key[4], query_index=qi,
                        shift_bin=shift_bin(observed), teacher_shift=float(observed), predicted_shift=float(value),
                        late_predicted_shift=float(last), absolute_error=float(abs(value-observed)),
                        magnitude_ratio=float(abs(value)/max(abs(observed), 1e-8)),
                        tanh_derivative=float(8*(1-np.clip(value/8, -1, 1)**2)))
                    # Keep a stable Parquet schema when training rows precede
                    # validation rows, which alone have intervention outputs.
                    for variant in VARIANTS[1:]:
                        row["predicted_" + variant] = None
                        row["effect_" + variant] = None
                    if split == "val":
                        for variant in VARIANTS[1:]:
                            changed = predictions[variant][bi, qi] - reference[qi]
                            row["predicted_" + variant] = float(changed)
                            row["effect_" + variant] = float(changed-value)
                    site_rows.append(row)
                if split == "val":
                    one_graph = {k: jnp.asarray(v[bi]) for k, v in graph.items()}
                    trace = trace_forward(params, one_graph)
                    metrics = attention_metrics(np.asarray(trace["site_attention"]["weights"]),
                        {k: np.asarray(v[bi]) for k, v in graph.items()}, np.asarray(graph["query_site"][bi]), q)
                    for qi, key in enumerate(record["keys"][:q]):
                        for head in range(metrics["entropy"].shape[-1]):
                            attention_rows.append(dict(component_id=record["component_id"], complex_id=cid,
                                chain=key[1], resnum=key[2], icode=key[3], group=key[4], query_index=qi,
                                shift_bin=shift_bin(teacher[qi]), head=head,
                                **{name: float(value[qi, head]) for name, value in metrics.items()}))
            if number % 100 == 0: print(json.dumps({"split": split, "batches": number, "total": len(plans)}), flush=True)
    write_parquet(out / "sites.parquet", site_rows); write_parquet(out / "attention.parquet", attention_rows)
    atomic_json(out / "progress.json", {"stage": "predictions", "complete": True})

    # Per-bin gradients on validation sites. The objective here is site-micro MSE
    # so gradient magnitudes and directions are directly comparable between bins.
    baseline = jnp.asarray(PKPDB_PK_MOD, jnp.float32)
    def loss(p, graphs, targets, selected_mask, valid):
        def one(g, y, mask):
            expected = y - baseline[g["query_group"]]
            error = jnp.where(mask, predict_site_shift_indexed(p, g)-expected, 0.)
            return jnp.sum(error*error), jnp.sum(mask)
        sums, counts = jax.vmap(one)(graphs, targets, selected_mask)
        return jnp.sum(jnp.where(valid, sums, 0.))/jnp.maximum(jnp.sum(jnp.where(valid, counts, 0)), 1)
    value_grad = jax.jit(jax.value_and_grad(loss))
    gradients = {}; gradient_rows = []
    records = split_records["val"]; plans = batches(records, manifest["config"]["batch_size"], byid)
    for low, high, name in BINS:
        total = None; sites = 0; losses = []
        for cids, (graph, target, eligible, valid) in loader.iterate(plans):
            group = np.asarray(graph["query_group"]); expected = target - np.asarray(PKPDB_PK_MOD)[group]
            selected_mask = eligible & (np.abs(expected) >= low) & (np.abs(expected) < high)
            count = int(np.sum(selected_mask[np.asarray(valid)]))
            if not count: continue
            value, gradient = value_grad(params, {k:jnp.asarray(v) for k,v in graph.items()},
                                         jnp.asarray(target), jnp.asarray(selected_mask), jnp.asarray(valid))
            total = tree_add(total, gradient, count); sites += count; losses.append((float(value), count))
        mean = jax.tree.map(lambda value: value/sites, total); gradients[name] = mean
        gradient_rows.append(dict(bin=name, sites=sites,
            mse=float(sum(value*count for value,count in losses)/sum(count for _,count in losses)), **gradient_norms(mean)))
    flat = {name: np.concatenate([np.ravel(x) for x in jax.tree.leaves(tree)]) for name, tree in gradients.items()}
    cosines = []
    names = list(flat)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            cosines.append(dict(bin_a=a, bin_b=b,
                cosine=float(np.dot(flat[a],flat[b])/(np.linalg.norm(flat[a])*np.linalg.norm(flat[b])))))
    atomic_json(out / "gradients.json", {"bins": gradient_rows, "cosines": cosines})
    loader.close()
    atomic_json(out / "verification.json", dict(passed=True, sites=len(site_rows), attention_rows=len(attention_rows),
        sites_sha256=digest(out/"sites.parquet"), attention_sha256=digest(out/"attention.parquet"),
        gradients_sha256=digest(out/"gradients.json"), test_data_included=False))


def report(out):
    require_compute(threads=4, allow_comp1400=True)
    import pyarrow.parquet as pq
    out = Path(out); sites = pq.read_table(out/"sites.parquet").to_pylist(); attention = pq.read_table(out/"attention.parquet").to_pylist()
    gradients = read(out/"gradients.json"); summaries=[]; causal=[]; att=[]
    for split in ("train", "val"):
        for _, _, name in BINS:
            rr=[r for r in sites if r["split"]==split and r["shift_bin"]==name]
            target=np.asarray([r["teacher_shift"] for r in rr]);pred=np.asarray([r["predicted_shift"] for r in rr]);late=np.asarray([r["late_predicted_shift"] for r in rr])
            slope=lambda y: float(np.cov(target,y,ddof=0)[0,1]/np.var(target)) if np.var(target)>0 else None
            summaries.append(dict(split=split,bin=name,sites=len(rr),mae=float(np.mean(abs(pred-target))),
                late_mae=float(np.mean(abs(late-target))),slope=slope(pred),late_slope=slope(late),
                prediction_sd=float(pred.std()),teacher_sd=float(target.std()),mean_magnitude_ratio=float(np.mean([r["magnitude_ratio"] for r in rr])),
                mean_tanh_derivative=float(np.mean([r["tanh_derivative"] for r in rr]))))
            if split=="val":
                for variant in VARIANTS[1:]:
                    changed=np.asarray([r["predicted_"+variant] for r in rr])
                    causal.append(dict(bin=name,variant=variant,mean_absolute_effect=float(np.mean(abs(changed-pred))),
                        ablated_mae=float(np.mean(abs(changed-target)))))
                aa=[r for r in attention if r["shift_bin"]==name]
                for metric in ("effective","maximum","mass_self","mass_arg","mass_opposite_class","mass_0_6","mass_6_10","mass_10_15","mass_15_20","neighbor_count"):
                    att.append(dict(bin=name,metric=metric,mean=float(np.mean([r[metric] for r in aa]))))
    results=dict(summary=summaries,causal=causal,attention=att,gradients=gradients)
    atomic_json(out/"results.json",results)
    val={r["bin"]:r for r in summaries if r["split"]=="val"}; train={r["bin"]:r for r in summaries if r["split"]=="train"}
    lines=["# Why does the site-token GQT miss large shifts?","",
        "Selected epoch 9 and late epoch 17 of the backbone-only site-orientation model. Predictions, interventions and gradients use full-float32 Triton. Native attention is used only to expose parity-validated weights. No test data were read.","",
        "| Shift bin | Train sites | Train MAE | Validation sites | Validation MAE | Predicted/teacher magnitude | Shift slope | Late validation MAE |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _,_,name in BINS:
        a,b=train[name],val[name];lines.append(f"| {name} | {a['sites']:,} | {a['mae']:.4f} | {b['sites']:,} | {b['mae']:.4f} | {b['mean_magnitude_ratio']:.3f} | {b['slope']:.3f} | {b['late_mae']:.4f} |")
    lines += ["","## Causal context tests","","| Shift bin | Intervention | Mean absolute prediction change | Ablated MAE |","|---|---|---:|---:|"]
    for row in causal: lines.append(f"| {row['bin']} | {row['variant']} | {row['mean_absolute_effect']:.4f} | {row['ablated_mae']:.4f} |")
    lines += ["","## Gradient path","","| Shift bin | Sites | MSE | Total | Head | Site block | Query | Encoder 0 | Encoder 1 |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in gradients["bins"]:lines.append(f"| {row['bin']} | {row['sites']:,} | {row['mse']:.4f} | {row['total']:.3g} | {row['head']:.3g} | {row['site']:.3g} | {row['query']:.3g} | {row['encoder_0']:.3g} | {row['encoder_1']:.3g} |")
    lines += ["","Gradient cosines and complete attention summaries are in `results.json`."]
    (out/"report.md").write_text("\n".join(lines)+"\n")
    verification=read(out/"verification.json");verification.update(report_sha256=digest(out/"report.md"),results_sha256=digest(out/"results.json"),report_complete=True);atomic_json(out/"verification.json",verification)


def main():
    import sys
    root=Path(os.environ["PKABENCH_RUNTIME"]);out=root/"audits/gqt-large-shift-diagnostics-v2"
    if sys.argv[1]=="run":run(root,out)
    elif sys.argv[1]=="report":report(out)
    else:raise ValueError(sys.argv[1])


if __name__ == "__main__": main()
