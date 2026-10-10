"""Frozen, explicitly nonphysical topology-corruption audit; no retraining."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
import jax
import numpy as np
from jaxpropka.topology import load_topology
from pkanet.model import PKPDB_PK_MOD
from pkabench.ogqt_rotation_invariance import _read_cif
from pkabench.runtime import atomic_json, digest, require_compute
from pkatrain.gqt_auxiliary_pilot import _primary_metrics, _aux_metrics, read
from pkatrain.gqt_paired_pinder import Loader, _bucket_n, _prefetched
from pkatrain.gqt_query_norm_pilot import setup
from pkatrain.site_graph_data import frames
from pkatrain.trainer import load_checkpoint

VERSION = "edge-corruption-v1"


def coordinate_context(runtime, cid):
    folder = runtime / "pretraining/pinder-pkai-v1/entries" / cid
    topology = load_topology(_read_cif(folder / "AB.cif.gz"), gap_policy="cap", freeze_disulfides=True)
    mapping = {}
    for site in read(folder / "sites.json"):
        chain, partner = site["chain"], site["partner"]
        if chain in mapping and mapping[chain] != partner: raise AssertionError("ambiguous chain partner")
        mapping[chain] = partner
    chains = [key.chain for key in topology.keys]
    if set(chains) - set(mapping): raise AssertionError((cid, "unmapped chains", set(chains)-set(mapping)))
    ca, frame, valid = frames(topology.backbone)
    sequence = np.zeros(len(ca), int); counters = {}
    for index, chain in enumerate(chains):
        sequence[index] = counters.get(chain, 0); counters[chain] = sequence[index] + 1
    return {"ca": ca, "frame": frame, "valid": valid, "chain": np.asarray(chains),
            "partner": np.asarray([mapping[c] for c in chains]), "sequence": sequence,
            "keys": [(k.chain, k.number, k.insertion) for k in topology.keys],
            "source_sha256": digest(folder / "AB.cif.gz"), "sites_sha256": digest(folder / "sites.json")}


def pair_features(context, ri, rj, site):
    delta = context["ca"][rj] - context["ca"][ri]
    distance = np.sqrt(np.sum(delta * delta) + 1e-8)
    fi, fj = context["frame"][ri], context["frame"][rj]
    di = delta @ fi / distance * context["valid"][ri]
    rbf = np.exp(-((distance - np.linspace(0, 20, 16)) / 1.5) ** 2)
    same = context["chain"][ri] == context["chain"][rj]
    if site:
        dj = -delta @ fj / distance * context["valid"][rj]
        orientation = (fi.T @ fj).ravel() * (context["valid"][ri] & context["valid"][rj])
        separation = np.clip((context["sequence"][rj]-context["sequence"][ri])/32, -1, 1) if same else 0
        feature = np.concatenate((rbf, di, dj, orientation, [same, ri == rj, separation]))
    else: feature = np.concatenate((rbf, di, [same]))
    switch = 1. if distance < 18 else .5 * (1 + np.cos(np.pi*np.clip((distance-18)/2, 0, 1)))
    return feature.astype(np.float32), float(switch), float(distance)


def audit_context(graph, bi, context):
    counts = {}
    for prefix in ("", "site_"):
        active = graph[prefix+"edge_mask"][bi, 0]
        neighbors = graph[prefix+"neighbors"][bi, 0]
        residues = graph["site_residue"][bi, 0] if prefix else np.arange(len(active))
        n = int(graph["site_mask" if prefix else "node_mask"][bi, 0].sum())
        partner = context["partner"][residues[:n]]
        same = partner[:, None] == partner[neighbors[:n]]
        expected = active[:n] & same
        if not np.array_equal(expected, graph[prefix+"edge_mask"][bi, 1, :n]):
            raise AssertionError((prefix, "free branch is not partner-correct"))
        ii, jj = np.where(active[:n])
        # A deterministic geometric spot check catches misaligned coordinate order.
        for index in np.linspace(0, len(ii)-1, min(16, len(ii)), dtype=int):
            i, k = ii[index], jj[index]; j = neighbors[i, k]
            feature, switch, _ = pair_features(context, residues[i], residues[j], bool(prefix))
            np.testing.assert_allclose(feature, graph[prefix+"edge"][bi, 0, i, k], atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(switch, graph[prefix+"switch"][bi, 0, i, k], atol=2e-5, rtol=2e-5)
        counts[prefix or "residue"] = int((active[:n] & ~same).sum())
    return counts


def perturb(graph, ids, contexts, level, mode, rate, draw):
    changed = {name: np.array(value, copy=True) for name, value in graph.items()}
    prefix = "site_" if level == "site" else ""
    edits = []
    for bi, cid in enumerate(ids):
        context = contexts[cid]
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"17|{cid}|{level}|{draw}".encode()).digest()[:8], "little"))
        mask = changed[prefix+"edge_mask"][bi]
        neighbors = changed[prefix+"neighbors"][bi]
        n = int(graph["site_mask" if prefix else "node_mask"][bi, 0].sum())
        residues = graph["site_residue"][bi, 0, :n] if prefix else np.arange(n)
        partner = context["partner"][residues]
        cross = mask[0, :n] & (partner[:, None] != partner[neighbors[0, :n]])
        cross &= changed[prefix+"switch"][bi, 0, :n] > 0
        slots = np.argwhere(cross); requested = int(np.ceil(rate * len(slots)))
        done = 0; distances = []
        if mode in ("remove_bound", "leak_free"):
            # Select undirected contacts and change both directions together.
            contacts = sorted({tuple(sorted((int(i), int(neighbors[0, i, k])))) for i, k in slots})
            order = rng.permutation(len(contacts)); wanted = int(np.ceil(rate*len(contacts)))
            for number in order[:wanted]:
                i, j = contacts[number]
                for source, destination in ((i,j),(j,i)):
                    kk = np.flatnonzero(cross[source] & (neighbors[0, source] == destination))
                    if len(kk) != 1: raise AssertionError("native graph must be reciprocal and unique")
                    mask[0 if mode == "remove_bound" else 1, source, kk[0]] = mode == "leak_free"
                    done += 1
        else:
            # Far endpoints have consistent geometry. Only the explicit forced
            # arms bypass the real distance switch; they are not physical contacts.
            candidates = {}
            for i in range(n):
                distances_i = np.linalg.norm(context["ca"][residues] - context["ca"][residues[i]], axis=1)
                candidates[i] = np.flatnonzero((partner != partner[i]) & (distances_i > 20) & (distances_i <= 40))
            if mode == "swap_forced": choices = slots[rng.permutation(len(slots))]
            else:
                choices = np.argwhere(~mask[0, :n]); choices = choices[rng.permutation(len(choices))]
            for i, k in choices:
                if done >= requested: break
                available = candidates[int(i)]
                if not len(available): continue
                used = neighbors[0, i, mask[0, i]]
                available = available[~np.isin(available, used)]
                if not len(available): continue
                j = int(rng.choice(available))
                feature, switch, distance = pair_features(context, residues[i], residues[j], bool(prefix))
                neighbors[0, i, k] = j
                changed[prefix+"edge"][bi, 0, i, k] = feature
                changed[prefix+"switch"][bi, 0, i, k] = switch if mode == "add_zero_gate" else 1.
                mask[0, i, k] = True
                done += 1; distances.append(distance)
        edits.append({"complex_id": cid, "eligible_directed_cross_edges": len(slots), "requested_directed": requested,
                      "realized_directed": done, "far_distance_mean": float(np.mean(distances)) if distances else None})
    # The unaffected branch, feature tensors, and original mmap arrays stay immutable.
    branch = 0 if mode == "leak_free" else 1
    for name in graph: np.testing.assert_array_equal(changed[name][:, branch], graph[name][:, branch])
    return changed, edits


def run(runtime):
    require_compute(threads=8, gpu_benchmark=True, allow_comp1400=True)
    started = time.monotonic(); runtime = Path(runtime)
    base, root, manifest, params, state, engine = setup(runtime, "separate")
    output = root / VERSION; output.mkdir(exist_ok=False)
    verification = read(root / "separate/verification.json")
    checkpoint = root / "separate/checkpoints" / f"epoch-{verification['best']['epoch']:03d}"
    params, _, _ = load_checkpoint(checkpoint, (params, state))
    records = [r for r in manifest["records"] if r["split"] == "val"]
    by_id = {r["id"]: r for r in records}; grouped = defaultdict(list)
    for row in records: grouped[_bucket_n(row["n"])].append(row["id"])
    plans = [ids[i:i+manifest["batch_sizes"][bucket]] for bucket, ids in sorted(grouped.items(), key=lambda x:int(x[0]))
             for i in range(0, len(ids), manifest["batch_sizes"][bucket])]
    conditions = [("native", None, None, 0., 0)]
    for level in ("residue", "site"):
        for mode in ("remove_bound", "leak_free", "add_forced", "swap_forced", "add_zero_gate"):
            for rate in (.01, .05, .10):
                for draw in range(3): conditions.append((f"{level}:{mode}:{rate}:{draw}",level,mode,rate,draw))
    atomic_json(output / "protocol.json", {"version": VERSION, "seed":17, "checkpoint_sha256":digest(checkpoint/"state.npz"),
        "checkpoint_epoch": verification["best"]["epoch"], "manifest_sha256":digest(base/"manifest.json"),
        "code_sha256":digest(Path(__file__)), "conditions":conditions, "validation_complexes":len(records),
        "retraining":False, "test_data_included":False,
        "interpretation":"Original labels retained: sensitivity to corrupted information, not a physical counterfactual.",
        "rates":"fraction of native positive-switch interpartner directed contacts; removals/restorations reciprocal, rounded by undirected contact count",
        "false_links":"20-40 A interpartner endpoints; exact endpoint geometry; add/swap forced arms explicitly override switch=1; zero-gate addition is a negative control",
        "capacity":"addition limited to existing padding; no truncation; realized counts reported",
        "swaps":"directed substitutions preserve each row degree; do not enforce reciprocal swaps"})
    results = defaultdict(list); edit_rows = []; context_audit = []; loader = Loader(base, manifest)
    for number, (ids, batch) in enumerate(_prefetched(loader, plans), 1):
        graph, targets, mask, burial, interface, metadata = batch
        contexts = {cid:coordinate_context(runtime,cid) for cid in ids}
        for bi,cid in enumerate(ids):
            context = contexts[cid]
            for key, residue in zip(by_id[cid]["keys"], graph["query_residue"][bi,0]):
                if tuple(key[:3]) != context["keys"][residue]: raise AssertionError("query/coordinate alignment")
            context_audit.append({"complex_id":cid,"chains":len(set(context["chain"])),
                                  "source_sha256":context["source_sha256"],"sites_sha256":context["sites_sha256"],
                                  "contacts":audit_context(graph,bi,context)})
        baseline = None
        for name,level,mode,rate,draw in conditions:
            changed, edits = (graph,[]) if name == "native" else perturb(graph,ids,contexts,level,mode,rate,draw)
            prediction = jax.tree.map(np.asarray, engine.predictions(params, changed))
            if name == "native": baseline = prediction
            if mode == "add_zero_gate":
                for head in prediction: np.testing.assert_allclose(prediction[head], baseline[head], rtol=2e-5, atol=2e-5)
            edit_rows.extend({"condition":name,**edit} for edit in edits)
            for bi,cid in enumerate(ids):
                for qi,key in enumerate(by_id[cid]["keys"]):
                    if not mask[bi,qi]: continue
                    ref = float(np.asarray(PKPDB_PK_MOD)[graph["query_group"][bi,0,qi]])
                    expected = targets[bi,:,qi]-ref; shift = prediction["shift"][bi,:,qi]
                    results[name].append({"complex_id":cid,**dict(zip(("chain","resnum","icode","group"),key)),
                        "state_error":float(np.mean(np.abs(shift-expected))),
                        "paired_error":float(shift[0]-shift[1]-expected[0]+expected[1]),
                        "burial_target":float((burial[bi,qi]*manifest["normalization"]["burial"]-.4)/.6),
                        "burial_prediction":float(prediction["burial"][bi,1,qi]),
                        "interface_target":float((interface[bi,qi]*manifest["normalization"]["interface"]-.05)/.95),
                        "interface_target_raw":float(interface[bi,qi]*manifest["normalization"]["interface"]),
                        "interface_prediction":float(prediction["interface"][bi,0,qi]),
                        "interface_prediction_raw":float(.05+.95*prediction["interface"][bi,0,qi]),
                        "interface":bool(metadata["interface"][bi,qi]),"distance":float(metadata["partner_distance_A"][bi,qi]),
                        "rsa_free":float(metadata["rsa_free"][bi,qi]),
                        **{f"delta_{head}":float(np.mean(np.abs(prediction[head][bi,:,qi]-baseline[head][bi,:,qi]))) for head in prediction}})
        progress = {"batch":number,"batches":len(plans),"seconds":time.monotonic()-started,"complexes_audited":len(context_audit)}
        atomic_json(output/"progress.json",progress); print(json.dumps(progress),flush=True)
    loader.close(); atomic_json(output/"partner-mask-audit.json",{"passed":True,"complexes":context_audit})
    with (output/"edits.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(edit_rows[0])); writer.writeheader(); writer.writerows(edit_rows)
    summary={}; mean=read(base/"common-warmup/verification.json")["train_burial_mean"]
    for name,rows in results.items():
        summary[name]={"primary":_primary_metrics(rows),"auxiliary":_aux_metrics(rows,mean),
            "sensitivity":{head:{"mean":float(np.mean([r[f'delta_{head}'] for r in rows])),
                "p99":float(np.quantile([r[f'delta_{head}'] for r in rows],.99))} for head in ("shift","burial","interface")}}
    atomic_json(output/"summary.json",summary)
    lines=["# Frozen oGQT edge corruption audit","",f"Seed 17, epoch {verification['best']['epoch']}; {len(records)} validation complexes. No retraining or test-set use.","",
        "Free-mask partner alignment and coordinate-feature checks passed. Rates refer to native interpartner contacts, not all edges. Forced far links bypass the 20 Å gate and are deliberately nonphysical.","",
        "| Level | Intervention | Rate | State MAE | Interface paired MAE | Burial MAE | Interface AP |", "|---|---|---:|---:|---:|---:|---:|"]
    native=summary["native"]
    def metrics(value):
        return [value["primary"]["state_mae"],value["primary"]["interface_paired_mae"],value["auxiliary"]["burial"]["pooled_mae"],value["auxiliary"]["interface"]["predicted_ranking"]["average_precision"]]
    lines.append("| — | Native | — | "+" | ".join(f"{v:.5f}" for v in metrics(native))+" |")
    for level in ("residue","site"):
        for mode in ("remove_bound","leak_free","add_forced","swap_forced","add_zero_gate"):
            for rate in (.01,.05,.10):
                values=np.asarray([metrics(summary[f"{level}:{mode}:{rate}:{draw}"]) for draw in range(3)])
                lines.append(f"| {level} | {mode} | {rate:.0%} | "+" | ".join(f"{v:.5f} ± {s:.5f}" for v,s in zip(values.mean(0),values.std(0,ddof=1)))+" |")
    lines.extend(["","± denotes variability across three corruption draws, not a confidence interval. Additions are limited by padding; see edits.csv for requested/realized counts. Swaps preserve directed row degree but are not reciprocal. Zero-gate additions passed prediction equivalence checks.","",f"Wall time: {time.monotonic()-started:.1f} seconds. Full metrics and prediction sensitivities: summary.json."])
    (output/"report.md").write_text("\n".join(lines)+"\n")
    atomic_json(output/"verification.json",{"passed":True,"wall_seconds":time.monotonic()-started,"conditions":len(conditions),"partner_masks_passed":True,"zero_gate_controls_passed":True})


if __name__ == "__main__": run(os.environ["PKABENCH_RUNTIME"])
