#!/usr/bin/env python3
"""Check the jax-triton 0.3 API adapter against native oGQT attention."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json, require_compute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_compute(threads=2, gpu_benchmark=True, allow_comp1400=True)

    import jax
    import jax.numpy as jnp
    import jax_triton as jt

    signature = inspect.signature(jt.triton_call)
    native_call = jt.triton_call
    if "out_shape" not in signature.parameters or "out_type" in signature.parameters:
        raise RuntimeError(f"expected jax-triton 0.3 API, found {signature}")

    def compatible(*values, **kwargs):
        kwargs["out_shape"] = kwargs.pop("out_type")
        backend = kwargs.pop("backend_options", {})
        kwargs["enable_fp_fusion"] = backend.get("enable_fp_fusion", True)
        return native_call(*values, **kwargs)

    jt.triton_call = compatible
    from pkanet.ogqt import predict_shift, predict_with_trace
    from pkatrain.gqt_paired_pinder import Loader
    from pkatrain.production_train import load_exported

    graph_dir = args.runtime / "training/ogqt-pinder-factorial-v1"
    manifest = json.loads((graph_dir / "manifest.json").read_text())
    record = next(row for row in manifest["records"] if row["split"] == "val")
    loader = Loader(graph_dir, manifest)
    try:
        batch, *_ = loader.batch([record["id"]])
    finally:
        loader.close()
    graph = jax.tree.map(lambda value: jnp.asarray(value[0, 0]), batch)
    params = load_exported(args.source / "models/ogqt-production-50k")
    indexed = np.asarray(jax.jit(predict_shift)(params, graph))
    reference = np.asarray(jax.jit(lambda p, g: predict_with_trace(p, g)["predicted_shift"])(params, graph))
    difference = np.abs(indexed-reference)
    result = {"passed": bool(np.allclose(indexed, reference, rtol=5e-6, atol=5e-6)),
              "complex_id": record["id"], "sites": int(indexed.size),
              "max_absolute_difference": float(difference.max()),
              "mean_absolute_difference": float(difference.mean()),
              "rtol": 5e-6, "atol": 5e-6}
    if not result["passed"]:
        raise AssertionError(result)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "triton-compatibility.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
