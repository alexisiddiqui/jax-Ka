"""Padding census of the production GQT batches (2026-10-10): per dataset and size bucket, how much of each padded
tensor is real, and how many neighbour indices land on row 0 (the zero fill), which the backward scatters
(Triton atomic_add of dK/dV; XLA scatter-add of the query-attention K/V gathers) accumulate into.

Per structure (PINDER: both branches) on the padded capacity of its bucket:
  nodes        real residues / n capacity
  edge slots   real residue edges / (n x k); split into masked slots of real nodes and slots of padded nodes
  sites        real sites / s capacity; site edge slots real / (s x sk)
  query gather rows of the site-token query attention (s x k gathers of neighbors[site_residue]): padded sites
               repeat residue 0's neighbour row
  row-0 hits   share of the (n x k) residue K/V gather indices and of the (s x sk) site gather indices equal to 0,
               versus real edges that end on residue/site 0
Batch fill: real structures / batch slots over the epoch-1 plan at the given batch size and fraction.

  python scripts/padding_census.py OUT.json [--batch 16] [--fraction 0.1] [--per-bucket 64]
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from pkatrain.production_graphs import output, read
from pkatrain.production_loading import MANIFEST, PinderSource, PkpdbSource, normalization, select
from pkatrain.production_train import _with_policy, epoch_plans


def census(graph):
    """One structure-branch graph (unbatched, padded) -> counts."""
    n, k = graph["neighbors"].shape; s, sk = graph["site_neighbors"].shape
    node = graph["node_mask"].astype(bool); edge = graph["edge_mask"].astype(bool)
    site = graph["site_mask"].astype(bool); site_edge = graph["site_edge_mask"].astype(bool)
    nb = graph["neighbors"]; snb = graph["site_neighbors"]; sres = graph["site_residue"]
    query_rows = nb[sres]                              # (s, k) query-attention gathers
    return {"n_cap": n, "k_cap": k, "s_cap": s, "sk_cap": sk, "q_cap": graph["query_residue"].shape[0],
            "nodes": int(node.sum()), "edges": int(edge.sum()),
            "masked_slots_real_nodes": int((~edge[node]).sum()), "slots_padded_nodes": int((~node).sum() * k),
            "sites": int(site.sum()), "site_edges": int(site_edge.sum()),
            "row0_residue_gathers": int((nb == 0).sum()), "row0_residue_real": int(((nb == 0) & edge).sum()),
            "row0_site_gathers": int((snb == 0).sum()), "row0_site_real": int(((snb == 0) & site_edge).sum()),
            "query_gathers": int(s * k), "query_gathers_real": int(graph["edge_mask"][sres][site].sum()),
            "query_row0": int((query_rows == 0).sum()), "padded_sites_on_residue0": int(((sres == 0) & ~site).sum())}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("out"); parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--fraction", type=float, default=0.1); parser.add_argument("--per-bucket", type=int, default=64)
    args = parser.parse_args(); root = Path(os.environ["PKABENCH_RUNTIME"])
    manifests = {d: read(output(root, d) / MANIFEST) for d in ("pinder", "pkpdb")}
    sources = {"pinder": PinderSource(_with_policy(manifests["pinder"], 1), norms=normalization(manifests["pinder"], args.fraction)),
               "pkpdb": PkpdbSource(_with_policy(manifests["pkpdb"], 1))}
    report = {"batch": args.batch, "fraction": args.fraction, "per_bucket_sample": args.per_bucket, "datasets": {}}
    rng = np.random.default_rng(0)
    for dataset, source in sources.items():
        records = select(manifests[dataset], "train", args.fraction); by_bucket = defaultdict(list)
        for r in records: by_bucket[source.policy.bucket(r["n"])].append(r)
        rows = {}
        for bucket in sorted(by_bucket, key=int):
            members = by_bucket[bucket]; sample = [members[i] for i in rng.permutation(len(members))[:args.per_bucket]]
            total = defaultdict(int)
            for r in sample:
                graphs = source.load([r["id"]])[0]
                branches = [{k: v[0, b] for k, v in graphs.items()} for b in range(2)] if dataset == "pinder" else [{k: v[0] for k, v in graphs.items()}]
                for g in branches:
                    for key, value in census(g).items(): total[key] += value
                    total["graphs"] += 1
            g = total["graphs"]; cap = lambda key: total[key] / g
            n, k, s, sk = cap("n_cap"), cap("k_cap"), cap("s_cap"), cap("sk_cap")
            rows[bucket] = {"structures": len(members), "sampled": len(sample), "capacity_n_k_q_s_sk": manifests[dataset]["capacities"][bucket],
                "node_fill": total["nodes"] / total["n_cap"],
                "edge_slot_fill": total["edges"] / (g * n * k),
                "edge_slots_masked_on_real_nodes": total["masked_slots_real_nodes"] / (g * n * k),
                "edge_slots_on_padded_nodes": total["slots_padded_nodes"] / (g * n * k),
                "mean_real_neighbors": total["edges"] / max(total["nodes"], 1),
                "site_fill": total["sites"] / total["s_cap"], "site_edge_slot_fill": total["site_edges"] / (g * s * sk),
                "query_gather_fill": total["query_gathers_real"] / total["query_gathers"],
                "residue_gathers_on_row0": total["row0_residue_gathers"] / (g * n * k),
                "residue_gathers_on_row0_real": total["row0_residue_real"] / (g * n * k),
                "residue_row0_hits_per_graph": total["row0_residue_gathers"] / g,
                "site_gathers_on_row0": total["row0_site_gathers"] / (g * s * sk),
                "site_row0_hits_per_graph": total["row0_site_gathers"] / g,
                "query_gathers_on_row0": total["query_row0"] / total["query_gathers"],
                "padded_sites_per_graph": (total["s_cap"] - total["sites"]) / g}
            print(json.dumps({dataset: {bucket: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[bucket].items()}}}), flush=True)
        report["datasets"][dataset] = rows
    pair, pk = epoch_plans(manifests, args.fraction, 1, args.batch)
    report["batch_fill"] = {"pinder": sum(map(len, pair)) / (len(pair) * args.batch), "pkpdb": sum(map(len, pk)) / (len(pk) * args.batch)}
    print(json.dumps({"batch_fill": report["batch_fill"]}))
    for source in sources.values(): source.close()
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
