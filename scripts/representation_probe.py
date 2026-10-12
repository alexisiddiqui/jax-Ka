"""Representation probe and residual test for candidate backbone features (2026-10-12).

For a trained production run (selected checkpoint), on the PINDER (AB branch) and pKPDB validation sets:
1. Probe: can a ridge regression from the final site embedding (what the linear pKa head reads) recover the feature?
   Held-out R^2, two folds split by structure. High R^2: the model already computes it, so an explicit input adds
   little.
2. Residual test: does the feature explain the model's errors (predicted - teacher shift)? Ridge on the feature x
   titratable-group interactions (with squares), held-out R^2 gain over group means alone. Only a feature the probe
   cannot recover and that explains residuals is worth a training run.
3. Burial reference: the model's own burial head (sigmoid of aux_burial on the site embedding) against the true
   target 1 - clip(RSA) (PINDER: bound-state RSA of the AB branch, rsa-bound-v1; pKPDB: store RSA), and, per feature,
   the held-out R^2 from predicted burial alone next to the full-embedding probe (how much of the feature is burial).

Features per query site, from backbone-v1 (numpy, no side chains):
- dihedrals: sin/cos phi, psi of the site residue (production_inputs.backbone_dihedrals).
- hse: half-sphere exposure up / down (CA within 13 A on the virtual-CB side / the other side), virtual-CB count 10 A.
- own_hbond: DSSP energies of the site residue's best backbone donor (its N-H) and acceptor (its C=O) H-bond.
- env_hbond: within 8 A of the site's virtual CB, backbone N-H groups that are free (no DSSP H-bond below -0.5
  kcal/mol), free N-H pointing at the virtual CB, bonded N-H, free C=O.
- edge_dist: shortest N(site)-O(j) and O(site)-N(j) distances (j not the site or its bonded neighbours).
DSSP: E = 0.084 x 332 (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN) kcal/mol, H = N + unit(C_{i-1} - O_{i-1}), no H on proline
or chain starts, pairs at least 2 apart in sequence (or on different chains) with CA-CA < 9 A, E floored at -9.9.

  python scripts/representation_probe.py OUT.json RUN [RUN ...]   (PKABENCH_RUNTIME set; GPU job)
"""
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np

GROUPS = {"dihedrals": ("phi_sin", "phi_cos", "psi_sin", "psi_cos"), "hse": ("hse_up", "hse_down", "cb_count_10"),
          "own_hbond": ("own_donor_E", "own_acceptor_E"),
          "env_hbond": ("free_nh_8", "free_nh_pointing_8", "bonded_nh_8", "free_co_8"), "edge_dist": ("min_N_O", "min_O_N")}


def virtual_cb(n, ca, c):
    b = ca - n; c_ = c - ca; a = np.cross(b, c_)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c_ + ca


def site_features(table, aa, residues):
    """Feature rows for the given residue indices of one structure (real nodes only in `table`)."""
    from pkatrain.production_inputs import PEPTIDE_BOND_A, backbone_dihedrals
    n, ca, c, o = (table[:, k:k + 3] for k in (0, 3, 6, 9)); chain = table[:, 12]; seq = table[:, 13]; m = len(table)
    prev = np.roll(np.arange(m), 1)
    has_prev = (chain[prev] == chain) & (seq[prev] == seq - 1) & (np.linalg.norm(c[prev] - n, axis=1) < PEPTIDE_BOND_A)
    co = c[prev] - o[prev]; h = n + co / np.maximum(np.linalg.norm(co, axis=1, keepdims=True), 1e-6)
    has_h = has_prev & (aa != 12)  # 12 = P in ALPHABET
    r = lambda x, y: np.maximum(np.linalg.norm(x[:, None] - y[None], axis=-1), 0.5)
    energy = 0.084 * 332 * (1 / r(n, o) + 1 / r(h, c) - 1 / r(h, o) - 1 / r(n, c))  # [donor i, acceptor j]
    near = (np.abs(seq[:, None] - seq[None]) >= 2) | (chain[:, None] != chain[None])
    ok = has_h[:, None] & near & (r(ca, ca) < 9.0)
    energy = np.where(ok, np.maximum(energy, -9.9), 0.0)
    donor = energy.min(1); acceptor = energy.min(0)
    cb = virtual_cb(n, ca, c); angles = backbone_dihedrals(table, xp=np)
    rows = []
    for s in residues:
        up = (ca - ca[s]) @ (cb[s] - ca[s]); d_ca = np.linalg.norm(ca - ca[s], axis=1); within = (d_ca < 13.0) & (np.arange(m) != s)
        to_cb = np.linalg.norm(n - cb[s], axis=1); other = np.arange(m) != s
        free_nh = has_h & (donor > -0.5) & (to_cb < 8.0) & other
        pointing = free_nh & (np.einsum("ij,ij->i", cb[s] - n, h - n) > 0)
        bonded_nh = has_h & (donor <= -0.5) & (to_cb < 8.0) & other
        free_co = (acceptor > -0.5) & (np.linalg.norm(o - cb[s], axis=1) < 8.0) & other
        apart = (np.abs(seq - seq[s]) >= 2) | (chain != chain[s])
        rows.append({"phi_sin": angles[s, 0], "phi_cos": angles[s, 1], "psi_sin": angles[s, 2], "psi_cos": angles[s, 3],
                     "hse_up": float(np.sum(within & (up > 0))), "hse_down": float(np.sum(within & (up <= 0))),
                     "cb_count_10": float(np.sum((np.linalg.norm(cb - cb[s], axis=1) < 10.0) & other)),
                     "own_donor_E": float(donor[s]), "own_acceptor_E": float(acceptor[s]),
                     "free_nh_8": float(free_nh.sum()), "free_nh_pointing_8": float(pointing.sum()), "bonded_nh_8": float(bonded_nh.sum()),
                     "free_co_8": float(free_co.sum()),
                     "min_N_O": float(np.min(np.linalg.norm(o[apart] - n[s], axis=1), initial=20.0)),
                     "min_O_N": float(np.min(np.linalg.norm(n[apart] - o[s], axis=1), initial=20.0))})
    return rows


def collect(root, run, dataset):
    import jax
    from pkanet.model import PKPDB_PK_MOD, linear
    from pkatrain.loading import LoaderConfig
    from pkatrain.production_graphs import output, read
    from pkatrain.production_inputs import BACKBONE_WIDTH, BackboneTable, ConcatTable
    from pkatrain.production_loading import MANIFEST, PinderSource, PkpdbSource, select
    from pkatrain.production_sidechains import SidechainTable
    from pkatrain.production_aux import RSA_BOUND_TABLE, ContactTable
    from pkatrain.production_train import _with_policy, run_dir, selected_model
    params, (embeddings, _) = selected_model(root, run); protocol = read(run_dir(root, run) / "protocol.json")
    sidechains = "sidechains" in protocol; arch = protocol["architecture"]
    table = "orientation_edges" in arch or "dihedral_features" in arch or "hbond_features" in arch or "geometry_auxiliary" in protocol or "chi_auxiliary" in protocol
    manifest = read(output(root, dataset) / MANIFEST); tables = ([SidechainTable(root, dataset)] if sidechains else []) + [BackboneTable(root, dataset)]
    side = tables[0] if len(tables) == 1 else ConcatTable(*tables); config = LoaderConfig()
    source = PinderSource(_with_policy(manifest), config=config, norms=protocol["pinder_weight_normalization"], sidechains=side) if dataset == "pinder" \
        else PkpdbSource(_with_policy(manifest), config=config, mask="train_mask", sidechains=side, aux=True)
    reference = np.asarray(PKPDB_PK_MOD); rsa_bound = ContactTable(root, RSA_BOUND_TABLE) if dataset == "pinder" else None

    @jax.jit
    def forward(graphs):
        def one(graph):
            if not table: graph = {**graph, "nodes": graph["nodes"][:, :-BACKBONE_WIDTH]}
            tokens, query = embeddings(params, graph); site = tokens[query]
            return site, 8 * jax.numpy.tanh(linear(params["head"], site)[:, 0]), jax.nn.sigmoid(linear(params["aux_burial"], site)[:, 0])
        return jax.vmap(one)(graphs)

    records = select(manifest, "val"); plans = source.policy.plans(records, np.random.default_rng(0)); out = []
    for ids in plans:
        batch = source.load(ids)
        if dataset == "pinder":
            graphs, targets, mask = batch[0], batch[1], batch[2]
            graphs = jax.tree.map(lambda x: x[:, 0], graphs); targets = targets[:, 0]
        else:
            graphs, targets, mask, rsa = batch[0], batch[1], batch[2], batch[3]
        tokens, shift, burial = (np.asarray(x) for x in forward(graphs))
        for slot, cid in enumerate(ids):
            graph = {k: np.asarray(v[slot]) for k, v in graphs.items()}; real = graph["node_mask"]
            active = np.asarray(mask[slot], bool); query = graph["query_site"]; residues = graph["site_residue"][query][active]
            group = graph["query_group"][active]; expected = targets[slot][active] - reference[group]
            feats = site_features(graph["nodes"][real][:, -BACKBONE_WIDTH:], graph["nodes"][real][:, :20].argmax(1), residues)
            if dataset == "pinder":
                values = rsa_bound(cid); site_rsa = np.full(len(active), np.nan); site_rsa[:len(values)] = values; site_rsa = site_rsa[active]
            else: site_rsa = np.where(np.asarray(rsa[slot]) >= 0, rsa[slot], np.nan)[active]
            for k in range(len(residues)):
                feats[k]["burial_true"] = float(1 - np.clip(site_rsa[k], 0, 1)) if np.isfinite(site_rsa[k]) else np.nan
                feats[k]["burial_pred"] = float(burial[slot][active][k])
                out.append((cid, int(group[k]), float(shift[slot][active][k] - expected[k]), tokens[slot][active][k], feats[k]))
    source.close(); return out


def ridge_r2(x, y, folds, alpha=1.0):
    """Held-out R^2 (two folds), standardized inputs, intercept."""
    pred = np.zeros_like(y)
    for f in (0, 1):
        tr, te = folds != f, folds == f
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-8; xs = (x - mu) / sd; xt = np.c_[xs[tr], np.ones(tr.sum())]
        w = np.linalg.solve(xt.T @ xt + alpha * np.eye(xt.shape[1]), xt.T @ y[tr]); pred[te] = np.c_[xs[te], np.ones(te.sum())] @ w
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def analyse(rows):
    folds = np.asarray([zlib.crc32(cid.encode()) % 2 for cid, *_ in rows]); group = np.asarray([g for _, g, *_ in rows])
    residual = np.asarray([r for _, _, r, *_ in rows]); emb = np.stack([e for *_, e, _ in rows])
    feats = {k: np.asarray([f[k] for *_, f in rows], float) for k in rows[0][4]}
    onehot = np.eye(9)[group]; base = ridge_r2(onehot, residual, folds)
    result = {"sites": len(rows), "residual_mae": float(np.mean(np.abs(residual))), "residual_r2_group_means": base, "groups": {}}
    from scipy.stats import spearmanr
    known = np.isfinite(feats["burial_true"]); pred = feats["burial_pred"][:, None]
    result["burial"] = {"sites_with_rsa": int(known.sum()), "probe_r2_true_burial": round(ridge_r2(emb[known], feats["burial_true"][known], folds[known]), 3),
                        "head_r2_true_burial": round(ridge_r2(pred[known], feats["burial_true"][known], folds[known]), 3),
                        "head_spearman_true_burial": round(float(spearmanr(pred[known, 0], feats["burial_true"][known])[0]), 3)}
    for name, keys in GROUPS.items():
        x = np.stack([feats[k] for k in keys], axis=1)
        probe = {k: ridge_r2(emb, feats[k], folds) for k in keys}
        from_burial = {k: ridge_r2(pred, feats[k], folds) for k in keys}
        design = np.concatenate([onehot] + [onehot * x[:, [i]] for i in range(x.shape[1])] + [onehot * x[:, [i]] ** 2 for i in range(x.shape[1])], axis=1)
        result["groups"][name] = {"probe_r2": {k: round(v, 3) for k, v in probe.items()}, "probe_r2_mean": round(float(np.mean(list(probe.values()))), 3),
                                  "burial_r2": {k: round(v, 3) for k, v in from_burial.items()}, "burial_r2_mean": round(float(np.mean(list(from_burial.values()))), 3),
                                  "residual_r2_gain": round(ridge_r2(design, residual, folds) - base, 4),
                                  "feature_means": {k: round(float(feats[k].mean()), 3) for k in keys}}
    return result


def main():
    root = Path(os.environ["PKABENCH_RUNTIME"]); out = Path(sys.argv[1]); report = {}
    for run in sys.argv[2:]:
        report[run] = {}
        for dataset in ("pinder", "pkpdb"):
            report[run][dataset] = analyse(collect(root, run, dataset))
            r = report[run][dataset]
            print(json.dumps({"run": run, "dataset": dataset, "burial": r["burial"],
                              **{g: {"probe": v["probe_r2_mean"], "from_pred_burial": v["burial_r2_mean"], "probe_each": v["probe_r2"], "burial_each": v["burial_r2"]}
                                 for g, v in r["groups"].items()}}), flush=True)
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
