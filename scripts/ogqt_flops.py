"""Analytic training FLOPs per epoch for production oGQT runs (2026-10-11; scaling plan, FLOPs beside wall-clock).

Counts the multiply-accumulates (MACs) of one forward pass over the padded tensors the GPU actually processes: every
batch padded to `batch` graphs at its bucket's capacities (manifest "capacities": residues N, residue neighbours K,
sites S, site neighbours SK), PINDER as two branches (AB, free) and one pKPDB batch per PINDER batch. Per graph:
- embed: N x node_dim x W
- each residue block application (attend_indexed): q/k/v/o 4 N W^2, feed-forward 2 N W F, attention 2 N K W, edge bias N K 20 H
- loop re-injection (loops >= 2): N W^2 per extra loop
- site query block (attend, context = residues): q/o 2 S W^2, k/v on all residues 2 N W^2, feed-forward 2 S W F,
  attention 2 S K W, edge bias S K 20 H
- site-site block: 4 S W^2 + 2 S W F + 2 S SK W + S SK (34 W + W H)
- heads: S W (shift) + S W (burial)
Training FLOPs = 6 x MACs (forward 2 FLOPs per MAC, backward twice the forward), the 6ND convention. Norms, softmax,
dropout and the optimizer are left out. Attention work is in the Triton kernel, which XLA's cost analysis cannot see,
hence the analytic count.

  python scripts/ogqt_flops.py RUN [RUN ...]       (PKABENCH_RUNTIME set; reads the runs' protocol.json)
"""
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

from pkatrain.loading import BucketPolicy
from pkatrain.production_graphs import output, read
from pkatrain.production_loading import MANIFEST, select
from pkatrain.production_train import BATCH, run_dir

HEADS, EDGE, SITE_EDGE = 4, 20, 34


def graph_macs(cap, w, f, node_dim, depth, loops):
    n, k, _, s, sk = cap
    block = 4 * n * w * w + 2 * n * w * f + 2 * n * k * w + n * k * EDGE * HEADS
    macs = n * node_dim * w + block * depth * loops + (loops - 1) * n * w * w
    macs += 2 * s * w * w + 2 * n * w * w + 2 * s * w * f + 2 * s * k * w + s * k * EDGE * HEADS
    macs += 4 * s * w * w + 2 * s * w * f + 2 * s * sk * w + s * sk * (SITE_EDGE * w + w * HEADS)
    return macs + 2 * s * w


def bucket_batches(manifest, fraction, batch):
    policy = BucketPolicy.from_json(manifest["bucket_policy"]); counts = Counter(policy.bucket(r["n"]) for r in select(manifest, "train", fraction))
    return {b: math.ceil(c / batch) for b, c in counts.items()}


def main():
    root = Path(os.environ["PKABENCH_RUNTIME"]); manifests = {d: read(output(root, d) / MANIFEST) for d in ("pinder", "pkpdb")}
    for run in sys.argv[1:]:
        protocol = read(run_dir(root, run) / "protocol.json"); a = protocol["architecture"]; batch = protocol.get("batch", BATCH)
        w, f, node_dim = a["width"], a["ff"], a.get("node_dim", 24); depth, loops = a.get("depth", 2), a.get("loops", 1)
        pair = bucket_batches(manifests["pinder"], protocol["fraction"], batch); n_batches = sum(pair.values())
        pk = bucket_batches(manifests["pkpdb"], protocol["fraction"], batch); pk_total = sum(pk.values())
        cap = {d: manifests[d]["capacities"] for d in manifests}
        macs = sum(c * batch * 2 * graph_macs(cap["pinder"][b], w, f, node_dim, depth, loops) for b, c in pair.items())
        # pKPDB: n_batches drawn from the stream, bucket mix as one pass over the pKPDB pool
        macs += n_batches / pk_total * sum(c * batch * graph_macs(cap["pkpdb"][b], w, f, node_dim, depth, loops) for b, c in pk.items())
        history = [json.loads(l) for l in (run_dir(root, run) / "history.jsonl").read_text().splitlines()] \
            if (run_dir(root, run) / "history.jsonl").exists() else []
        train_s = sorted(h["train_seconds"] for h in history)
        print(json.dumps({"run": run, "width": w, "node_dim": node_dim, "depth": depth, "loops": loops, "pinder_batches": n_batches,
                          "train_pflop_per_epoch": round(6 * macs / 1e15, 3),
                          "median_train_s_per_epoch": train_s[len(train_s) // 2] if train_s else None}), flush=True)


if __name__ == "__main__":
    main()
