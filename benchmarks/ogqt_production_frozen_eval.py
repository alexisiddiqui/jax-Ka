#!/usr/bin/env python3
"""Evaluate the exported production oGQT weights on the frozen 400-complex cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute


REVISION = "7c3e3e97582f7f3162525add8b265824b0af29a8"
PARAMS_SHA256 = "29eb740008244e5069f54ff81ebfbeb13e8fb1bf5df0fba9abe367ae255c5fcc"
MODEL = "oGQT production 50K"


def read(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value):
    return str(value).lower() == "true"


def key(row):
    return (row["complex_id"], row["chain"], int(row["resnum"]), row.get("icode", ""), row["group"])


def metrics(values):
    x = np.asarray(values, np.float64)
    return {"sites": len(x), "mae": float(np.abs(x).mean()), "rmse": float(np.sqrt(np.mean(x*x))),
            "bias": float(x.mean())}


def hh_integral(pka, lo=6.1, hi=7.4):
    ln10 = math.log(10.0)
    primitive = lambda ph: ph - np.logaddexp(0.0, ln10 * (ph-pka)) / ln10
    return float(primitive(hi)-primitive(lo))


def check_graph_store(root, manifest, store):
    """Prove that the reused frozen mmap graph equals the current production builder."""
    from pkatrain.production_graphs import _pinder_one
    row = next(r for r in manifest["records"] if r["split"] == "val")
    rebuilt, _ = _pinder_one(root, row["id"], "val")
    frozen = store.raw(row["id"])
    checked = []
    for name, expected in rebuilt.items():
        if name not in frozen:
            continue
        actual = np.asarray(frozen[name])
        if actual.shape != expected.shape or not np.array_equal(actual, expected, equal_nan=True):
            delta = None
            if actual.shape == expected.shape and np.issubdtype(actual.dtype, np.number):
                delta = float(np.nanmax(np.abs(actual.astype(np.float64)-expected.astype(np.float64))))
            raise AssertionError((row["id"], name, actual.shape, expected.shape, delta))
        checked.append(name)
    required = {"nodes", "neighbors", "edge", "switch", "query_residue", "query_group",
                "site_residue", "site_type", "site_neighbors", "site_edge", "site_switch", "query_site",
                "branch_edge_mask", "branch_site_edge_mask"}
    if not required <= set(checked):
        raise AssertionError(("missing graph checks", sorted(required-set(checked))))
    return {"complex_id": row["id"], "fields": sorted(checked)}


def infer(root, source, output, smoke=False):
    require_compute(threads=2, gpu_benchmark=True, allow_comp1400=True)
    import jax
    import jax.numpy as jnp
    import jax_triton as jt

    # The production kernel changed only the jax-triton call spelling when it
    # moved from 0.3 to 0.4. Keep this benchmark on the validated JAX/CUDA
    # environment while translating that external API boundary explicitly.
    triton_signature = inspect.signature(jt.triton_call)
    triton_compat = "native"
    if "out_shape" in triton_signature.parameters and "out_type" not in triton_signature.parameters:
        native_triton_call = jt.triton_call

        def compatible_triton_call(*args, **kwargs):
            kwargs["out_shape"] = kwargs.pop("out_type")
            backend_options = kwargs.pop("backend_options", {})
            if "enable_fp_fusion" in backend_options:
                kwargs["enable_fp_fusion"] = backend_options["enable_fp_fusion"]
            return native_triton_call(*args, **kwargs)

        jt.triton_call = compatible_triton_call
        triton_compat = "jax-triton-0.3-api-adapter"
    from pkanet.model import PKPDB_PK_MOD
    from pkanet.ogqt import predict_shift
    from pkatrain.gqt_paired_pinder import Loader, PairedMMap
    from pkatrain.production_train import load_exported

    root, source, output = Path(root), Path(source), Path(output)
    if (source / ".source-revision").read_text().strip() != REVISION:
        raise AssertionError("unexpected jax-Ka main revision")
    model_dir = source / "models/ogqt-production-50k"
    if digest(model_dir / "params.npz") != PARAMS_SHA256:
        raise AssertionError("production oGQT parameter hash mismatch")
    model_meta = read(model_dir / "model.json")
    if model_meta["params_sha256"] != PARAMS_SHA256 or model_meta["parameter_count"] != 67770:
        raise AssertionError("production oGQT metadata mismatch")
    graph_dir = root / "training/ogqt-pinder-factorial-v1"
    manifest = read(graph_dir / "manifest.json")
    records = [row for row in manifest["records"] if row["split"] == "val"]
    if len(records) != 400 or len({r["cluster_id"] for r in records}) != 400:
        raise AssertionError("expected frozen 400-cluster validation cohort")
    if smoke:
        records = records[:1]

    mmap = PairedMMap(graph_dir / "mmap-v1", manifest["records"])
    graph_check = check_graph_store(root, manifest, mmap)
    mmap.close()
    loader = Loader(graph_dir, manifest)
    params = load_exported(model_dir)

    @jax.jit
    def predict(p, graph):
        batch, branches = graph["nodes"].shape[:2]
        flat = jax.tree.map(lambda value: value.reshape((batch*branches,) + value.shape[2:]), graph)
        shift = jax.vmap(predict_shift, in_axes=(None, 0))(p, flat).reshape(batch, branches, -1)
        return shift + PKPDB_PK_MOD[graph["query_group"]]

    by_bucket = defaultdict(list)
    for row in records:
        bucket = next(int(n) for n in sorted(map(int, manifest["capacities"])) if row["n"] <= int(n))
        by_bucket[bucket].append(row)
    predictions = []
    try:
        for bucket in sorted(by_bucket):
            rows = by_bucket[bucket]
            for start in range(0, len(rows), 8):
                chunk = rows[start:start+8]
                graph, _, mask, _, _, _ = loader.batch([row["id"] for row in chunk])
                values = np.asarray(predict(params, jax.tree.map(jnp.asarray, graph)))
                for i, row in enumerate(chunk):
                    if not np.all(mask[i, :row["q"]]):
                        raise AssertionError((row["id"], "invalid query mask"))
                    for site, bound, free in zip(row["keys"], values[i, 0, :row["q"]],
                                                  values[i, 1, :row["q"]], strict=True):
                        predictions.append({"complex_id": row["id"], "cluster_id": row["cluster_id"],
                            "ctype": row["ctype"], "chain": site[0], "resnum": site[1], "icode": site[2],
                            "group": site[3], "predicted_ab": float(bound), "predicted_free": float(free)})
    finally:
        loader.close()

    output.mkdir(parents=True, exist_ok=True)
    pred_path = output / ("smoke-predictions.csv" if smoke else "predictions.csv")
    with pred_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)

    teacher = root / "audits/pinder-pkai-pypka-v1"
    lookup = {key(row): row for row in predictions}
    state_rows, pair_rows = [], []
    teacher_state_rows = read_csv(teacher / "matched-state.csv")
    teacher_state = {}
    for row in teacher_state_rows:
        teacher_state[(*key(row), row["state"])] = float(row["pypka"])
        pred = lookup.get(key(row))
        if pred is None:
            continue
        estimate = pred["predicted_ab"] if row["state"] == "AB" else pred["predicted_free"]
        error = estimate-float(row["pypka"])
        state_rows.append({**row, "model": MODEL, "prediction": estimate, "error": error})
    for row in read_csv(teacher / "matched-paired.csv"):
        pred = lookup.get(key(row))
        if pred is None:
            continue
        estimate = pred["predicted_ab"]-pred["predicted_free"]
        error = estimate-float(row["pypka_shift"])
        pair_rows.append({**row, "model": MODEL, "prediction_shift": estimate, "shift_error": error})
    interface = [row for row in pair_rows if truth(row["interface"])]
    result = {"state": metrics([r["error"] for r in state_rows]),
              "paired": metrics([r["shift_error"] for r in pair_rows]),
              "interface_paired": metrics([r["shift_error"] for r in interface])}

    by_complex = defaultdict(lambda: [0.0, 0.0, 0])
    for row in pair_rows:
        pred = lookup[key(row)]
        site_key = key(row)
        reference = (hh_integral(teacher_state[(*site_key, "AB")])
                     - hh_integral(teacher_state[(*site_key, row["free_state"]) ]))
        estimate = hh_integral(pred["predicted_ab"])-hh_integral(pred["predicted_free"])
        item = by_complex[row["complex_id"]]; item[0] += reference; item[1] += estimate; item[2] += 1
    linkage_diff = np.asarray([x[0]-x[1] for x in by_complex.values()])
    linkage = {"complexes": len(by_complex), "site_pairs": len(pair_rows),
        "reference_mean": float(np.mean([x[0] for x in by_complex.values()])),
        "prediction_mean": float(np.mean([x[1] for x in by_complex.values()])),
        "mean_reference_minus_prediction": float(linkage_diff.mean()),
        "mean_absolute_difference": float(np.abs(linkage_diff).mean())}
    prefix = "smoke-" if smoke else ""
    for name, rows in (("matched-state.csv", state_rows), ("matched-paired.csv", pair_rows)):
        with (output / f"{prefix}{name}").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload = {"passed": True, "smoke": smoke, "model": MODEL, "complexes_processed": len(records),
        "triton_compatibility": triton_compat,
        "prediction_sites": len(predictions), "results": result, "linkage": linkage,
        "graph_store_equivalence": graph_check, "test_data_included": False}
    atomic_json(output / f"{prefix}summary.json", payload)
    atomic_json(output / f"{prefix}manifest.json", {"jax_ka_revision": REVISION,
        "params_sha256": PARAMS_SHA256, "model_json_sha256": digest(model_dir / "model.json"),
        "graph_manifest_sha256": digest(graph_dir / "manifest.json"),
        "graph_verification_sha256": digest(graph_dir / "mmap-v1/verification.json"),
        "teacher_state_sha256": digest(teacher / "matched-state.csv"),
        "teacher_pair_sha256": digest(teacher / "matched-paired.csv"),
        "code_sha256": digest(Path(__file__)), "predictions_sha256": digest(pred_path),
        "model_metadata": model_meta, "test_data_included": False})
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    infer(args.runtime, args.source, args.output, args.smoke)


if __name__ == "__main__":
    main()
