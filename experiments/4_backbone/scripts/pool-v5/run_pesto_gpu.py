"""hotspot-oracle-v1 (2026-10-30): PeSTo (LBM-EPFL/PeSTo @ ba651aa, model i_v4_1_2021-09-07_11-21) per-residue
protein-protein interface probability for every exported crystal chain (export_chains.py zip), on one GPU.

Procedure = work/pesto-compare/run_pesto.py (itself apply_model.ipynb): read_pdb, clean_structure, tag_hetatm_chains,
split_by_chain, filter_non_atomic_subunits, remove_duplicate_tagged_subunits, concatenate_chains, encode_structure,
encode_features(...)[0] (element one-hot), 64 nearest neighbours, collate (ids + 1, sink at 0), sigmoid of output
channel 0 = protein interface. Each chain is predicted alone. Differences: parsing runs in CPU worker processes; the
64-NN topology is computed on the GPU in row chunks with extract_topology's own elementwise ops (norm of X[j] - X[i],
D + max(D) * (D < 0.01)) instead of one dense (N, N, 3) tensor; only ids_topk is used by the model. (torch.cdist is NOT
used: its matmul path changes ~5% of neighbour sets and PeSTo outputs by up to 0.5.)
Original residue number / insertion code ride through PeSTo's renumbering ('orig' field) and map each prediction to
its graph node; residues PeSTo did not score stay NaN; a chain whose keys are not unique is an error (all NaN).

  python run_pesto_gpu.py CHAINS_ZIP PESTO_DIR OUT_PREFIX      -> OUT_PREFIX.npz (ids, offsets, pesto) + OUT_PREFIX.tsv
"""
import json, os, sys, time, zipfile
from collections import deque
import numpy as np

W = {}


def _init(zpath, pesto):
    import torch as pt
    pt.set_num_threads(1)
    for p in (os.path.join(pesto, "model/save/i_v4_1_2021-09-07_11-21"), pesto):
        if p not in sys.path: sys.path.insert(0, p)
    W["zip"] = zipfile.ZipFile(zpath); W["tmp"] = os.path.join(os.environ.get("PESTO_LOCAL", "/tmp"), f"chain-{os.getpid()}.pdb")


def prepare(task):
    """(cid, chain) -> (cid, chain, X float32 (N,3), qe bool (N,30), residue index per atom (N,), keys per residue) | error."""
    cid, chain = task
    try:
        from src.structure_io import read_pdb
        from src.structure import (clean_structure, tag_hetatm_chains, split_by_chain, filter_non_atomic_subunits,
                                   remove_duplicate_tagged_subunits, concatenate_chains)
        from src.data_encoding import onehot, std_elements
        with open(W["tmp"], "wb") as f: f.write(W["zip"].read(f"{cid}/{chain}.pdb"))
        s = read_pdb(W["tmp"]); s["orig"] = np.array([f"{r}|{i}" for r, i in zip(s["resid"].tolist(), s["icode"].tolist())])
        s = tag_hetatm_chains(clean_structure(s)); st = concatenate_chains(remove_duplicate_tagged_subunits(filter_non_atomic_subunits(split_by_chain(s))))
        uniq, first, inv = np.unique(st["resid"], return_index=True, return_inverse=True); keys = st["orig"][first].tolist()
        qe = onehot(st["element"], std_elements)
        return cid, chain, st["xyz"].astype(np.float32), qe, inv.astype(np.int64), keys, None
    except Exception as e:  # noqa: BLE001
        return cid, chain, None, None, None, None, f"{type(e).__name__}: {str(e)[:160]}"


def topk_ids(pt, X, k=64, rows=1024):
    """extract_topology(X, k)[0] + 1 (collate's sink offset), computed in row chunks with the same elementwise ops
    (R = X[j] - X[i], D = norm(R), D + max(D) * (D < 0.01)) so the distances, and hence the neighbour sets, are identical."""
    n = X.shape[0]; knn = min(k, n); dmax = pt.zeros((), device=X.device)
    for s in range(0, n, rows): dmax = pt.maximum(dmax, pt.norm(X.unsqueeze(0) - X[s:s + rows].unsqueeze(1), dim=2).max())
    out = pt.empty((n, knn), dtype=pt.long, device=X.device)
    for s in range(0, n, rows):
        D = pt.norm(X.unsqueeze(0) - X[s:s + rows].unsqueeze(1), dim=2); D = D + dmax * (D < 1e-2).float()
        out[s:s + rows] = pt.topk(D, knn, dim=1, largest=False)[1]
    ids = pt.zeros((n, k), dtype=pt.long, device=X.device); ids[:, :knn] = out + 1; return ids


def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    zpath, pesto, prefix = sys.argv[1:4]
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist(); infos = {}
        for nm in names:
            if nm.endswith("/residues.json"): info = json.loads(z.read(nm)); infos[info["id"]] = info
    ids = sorted(infos); tasks = [(cid, c) for cid in ids if "error" not in infos[cid] for c in infos[cid]["chains"]]
    workers = int(os.environ.get("PESTO_WORKERS", "48"))
    pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(zpath, pesto))
    import torch as pt
    for p in (os.path.join(pesto, "model/save/i_v4_1_2021-09-07_11-21"), pesto): sys.path.insert(0, p)
    from config import config_model
    from model import Model
    dev = pt.device(os.environ.get("PESTO_DEVICE", "cuda")); model = Model(config_model)
    model.load_state_dict(pt.load(os.path.join(pesto, "model/save/i_v4_1_2021-09-07_11-21/model_ckpt.pt"), map_location="cpu")); model = model.eval().to(dev)
    print(json.dumps({"complexes": len(ids), "chains": len(tasks), "export_errors": sum("error" in infos[c] for c in ids), "workers": workers,
                      "device": pt.cuda.get_device_name(0) if dev.type == "cuda" else str(dev), "torch": pt.__version__}), flush=True)
    score = {cid: np.full(infos[cid]["n"], np.nan, np.float32) for cid in ids}; status = {cid: [] for cid in ids}
    rows = {}
    for cid in ids:
        info = infos[cid]
        if "error" in info: status[cid].append(f"export: {info['error']}"); continue
        for code, c in enumerate(info["chains"]):
            r = np.flatnonzero(np.asarray(info["code"]) == code)
            rows[(cid, c)] = {f"{info['resnum'][i]}|{info['icode'][i]}": int(i) for i in r}
    t = time.time(); window = deque(); it = iter(tasks); done = 0; unmapped = 0
    def submit():
        for task in it:
            window.append(pool.submit(prepare, task))
            if len(window) >= 4 * workers: return
    submit()
    while window:
        cid, chain, X, qe, inv, keys, err = window.popleft().result(); submit()
        if err is None:
            try:
                if len(set(keys)) != len(keys): raise AssertionError("PeSTo residue keys not unique")
                with pt.no_grad():
                    Xg = pt.from_numpy(X).to(dev); nres = len(keys); M = pt.zeros((X.shape[0], nres), dtype=pt.float, device=dev)
                    M[pt.arange(X.shape[0], device=dev), pt.from_numpy(inv).to(dev)] = 1.0
                    z = model(Xg, topk_ids(pt, Xg), pt.from_numpy(qe.astype(np.float32)).to(dev), M)
                    p = pt.sigmoid(z[:, 0]).cpu().numpy().astype(np.float32)
                m = rows[(cid, chain)]; hit = 0
                for key, v in zip(keys, p):
                    i = m.get(key)
                    if i is not None: score[cid][i] = v; hit += 1
                miss_graph = len(m) - hit; extra = len(keys) - hit; unmapped += miss_graph
                status[cid].append("ok" if miss_graph == 0 and extra == 0 else f"{chain}: {miss_graph} graph residues unscored, {extra} PeSTo residues unmapped")
            except Exception as e:  # noqa: BLE001
                status[cid].append(f"{chain}: {type(e).__name__}: {str(e)[:160]}")
        else: status[cid].append(f"{chain}: {err}")
        done += 1
        if done % 2000 == 0: print(json.dumps({"chains": done, "of": len(tasks), "s": round(time.time() - t)}), flush=True)
    pool.shutdown()
    offsets = np.concatenate([[0], np.cumsum([infos[c]["n"] for c in ids])]).astype(np.int64)
    np.savez(prefix + ".partial.npz", ids=np.asarray(ids), offsets=offsets, pesto=np.concatenate([score[c] for c in ids]) if ids else np.zeros(0, np.float32))
    os.replace(prefix + ".partial.npz", prefix + ".npz")
    with open(prefix + ".tsv", "w") as f:
        f.write("complex_id\tsplit\tn\tpesto_status\tpesto_scored\n")
        for c in ids:
            st = status[c]; ok = all(s == "ok" for s in st) and len(st) > 0
            f.write(f"{c}\t{infos[c]['split']}\t{infos[c]['n']}\t{'ok' if ok else '; '.join(s for s in st if s != 'ok')}\t{int(np.isfinite(score[c]).sum())}\n")
    bad = sum(not all(s == "ok" for s in status[c]) for c in ids)
    print(json.dumps({"complexes": len(ids), "chains": done, "complexes_not_ok": bad, "graph_residues_unscored": unmapped,
                      "residues": int(offsets[-1]), "nan": int(sum(np.isnan(score[c]).sum() for c in ids)), "s": round(time.time() - t)}), flush=True)


if __name__ == "__main__":
    main()
