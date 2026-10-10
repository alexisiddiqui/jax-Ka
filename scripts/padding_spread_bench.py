"""Padding spread benchmark (2026-10-10): the same production batches loaded with the zero-fill padding (spread=False)
and with production_loading.spread_padding (spread=True). Per dataset and bucket (batch 16, real training ids):
gradient-step time (median of 10 after compilation), loss equality, and the relative global-norm gradient
difference, against a repeat of the zero-fill step (float nondeterminism of the atomics) as the noise floor.

  python scripts/padding_spread_bench.py OUT.json [--batch 16]
"""
import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pkanet.ogqt import initialize as initialize_ogqt
from pkatrain.gqt_multitask_replay import JointEngine
from pkatrain.production_graphs import output, read
from pkatrain.production_loading import MANIFEST, PinderSource, PkpdbSource, normalization, select
from pkatrain.production_train import ARCHITECTURE, SEED, _with_policy


def timed(fn, args, repeats=10):
    jax.block_until_ready(fn(*args)); times = []
    for _ in range(repeats):
        t = time.perf_counter(); value = fn(*args); jax.block_until_ready(value); times.append(time.perf_counter() - t)
    return value, float(np.median(times))


def flat(tree): return jnp.concatenate([x.ravel() for x in jax.tree.leaves(tree)])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("out"); parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args(); root = Path(os.environ["PKABENCH_RUNTIME"])
    manifests = {d: read(output(root, d) / MANIFEST) for d in ("pinder", "pkpdb")}
    params = initialize_ogqt(jax.random.PRNGKey(SEED), **ARCHITECTURE); engine = JointEngine(params); rows = []
    for dataset in ("pinder", "pkpdb"):
        manifest = _with_policy(manifests[dataset], args.batch)
        make = (lambda spread: PinderSource(manifest, norms=normalization(manifests["pinder"], 0.1), spread=spread)) if dataset == "pinder" \
            else (lambda spread: PkpdbSource(manifest, spread=spread))
        sources = {False: make(False), True: make(True)}; policy = sources[False].policy
        train = select(manifests[dataset], "train", 0.1)
        for bucket in manifests[dataset]["capacities"]:
            ids = [r["id"] for r in train if policy.bucket(r["n"]) == bucket][:args.batch]
            if not ids: continue
            result = {}
            for spread in (False, True):
                host = sources[spread].load(ids)
                if dataset == "pinder":
                    graphs, targets, mask, wb, wi, _, valid = host; device = jax.device_put((graphs, targets, mask, wb, wi, valid))
                    fn = lambda *a: engine.paired_value_grad(params, *a); (value, gradient), seconds = timed(fn, device); loss = value[0]
                else:
                    device = jax.device_put(host); fn = lambda *a: engine.pkpdb_value_grad(params, *a)
                    (loss, gradient), seconds = timed(fn, device)
                result[spread] = (float(loss), flat(gradient), seconds)
                if not spread: repeat = flat(fn(*device)[1])
            norm = float(jnp.linalg.norm(result[False][1]))
            row = {"dataset": dataset, "bucket": bucket, "structures": len(ids),
                   "zero_fill_seconds": result[False][2], "spread_seconds": result[True][2], "speedup": result[False][2] / result[True][2],
                   "loss_zero_fill": result[False][0], "loss_spread": result[True][0], "loss_equal": result[False][0] == result[True][0],
                   "grad_rel_diff": float(jnp.linalg.norm(result[False][1] - result[True][1])) / norm,
                   "grad_rel_diff_repeat_zero_fill": float(jnp.linalg.norm(result[False][1] - repeat)) / norm}
            rows.append(row); print(json.dumps(row), flush=True)
        for source in sources.values(): source.close()
    Path(args.out).write_text(json.dumps({"batch": args.batch, "device": jax.devices()[0].device_kind, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
