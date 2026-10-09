"""pkPDB build ($PKPDB_BUILD, default pkpdb-5k-v3): burial field and absolute-pKa loss weight per mapped site, in entries/<pdb>/environment.json
{"version": "resolved-v2", "sites": [...]} (separate file: sites.json is hash-verified by pilot.json). Rows align with sites.json order:
  rsa          relative residue SASA of the deposited structure restricted to the build's selected protein chains
               (defects.json sequences; conformers resolved as in pkpdb_mask_all; canonical residues, heavy atoms, label chains with author numbering,
               as pkpdb_mask_all), biotite Shrake-Rupley, ProtOr radii, Tien 2013 max ASA,
  w_burial     pkabench.site_weights.burial_weight(rsa),
  chain_distance_A  residue-level minimum heavy-atom distance to any other selected protein chain of the entry (label chains,
               homomer copies included), as PINDER's partner_distance_A with "partner" = every other chain; null for one-chain entries,
  w_interface  pkabench.site_weights.interface_weight(chain_distance_A); null for one-chain entries.
pKPDB has single-state labels only (no Siamese task); the distance fields are for weighting/reporting the absolute loss
(user decision 2026-10-09). resolved-v1 (rsa, w_burial only) is rewritten. Usage: pkpdb_env.py <task index> <n tasks>."""
import sys, json, gzip
from pathlib import Path
import numpy as np, biotite.structure as struc
from biotite.structure.io import pdbx
from pkabench.prep import CANONICAL
from pkabench.conformers import resolve
from pkabench.site_weights import burial_weight, interface_weight
from scipy.spatial import cKDTree
import os
R = Path("/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench"); OUT = R / "pretraining" / os.environ.get("PKPDB_BUILD", "pkpdb-5k-v3")
MAX = dict(ALA=129, ARG=274, ASN=195, ASP=193, CYS=167, GLN=225, GLU=223, GLY=104, HIS=224, ILE=197, LEU=201, LYS=236,
           MET=224, PHE=240, PRO=159, SER=155, THR=172, TRP=285, TYR=263, VAL=174)
k, n = int(sys.argv[1]), int(sys.argv[2]); done = 0
for rec in json.loads((OUT / "pilot.json").read_text())["records"][k::n]:
    pdb = rec["pdb_id"]; d = OUT / "entries" / pdb
    if (d / "environment.json").exists():
        old = json.loads((d / "environment.json").read_text())
        if isinstance(old, dict) and old.get("version") == "resolved-v2": continue  # list = pre-resolution; resolved-v1 lacks distances
    chains = [s["chain"] for s in json.loads((d / "defects.json").read_text())["sequences"]]
    with gzip.open(R / "pretraining/pkpdb-v1/structures" / pdb[1:3] / f"{pdb}.cif.gz", "rt") as f: cif = pdbx.CIFFile.read(f)
    cif, _ = resolve(cif, chains)  # same conformer resolution as pkpdb_mask_all, so label and author views align
    label = pdbx.get_structure(cif, model=1, altloc="occupancy", use_author_fields=False)
    author = pdbx.get_structure(cif, model=1, altloc="occupancy", use_author_fields=True)
    a = label.copy(); a.res_id = author.res_id.copy(); a.ins_code = author.ins_code.copy()
    a = a[np.isin(a.chain_id, chains) & np.isin(a.res_name, list(CANONICAL)) & ~np.isin(np.char.upper(a.element), ["H", "D"])]
    s = struc.sasa(a, vdw_radii="ProtOr"); s = np.where(np.isfinite(s), s, 0.)
    per = struc.apply_residue_wise(a, s, np.sum); st = struc.get_residue_starts(a)
    key = lambda i: (str(a.chain_id[i]), int(a.res_id[i]), str(a.ins_code[i]).strip())
    rsa = {key(i): float(v) / MAX.get(str(a.res_name[i]), 200) for i, v in zip(st, per)}
    dist = {}
    for c in np.unique(a.chain_id):
        mine = a.chain_id == c
        if mine.all(): continue
        dd = cKDTree(a.coord[~mine]).query(a.coord[mine])[0]
        for i, x in zip(np.flatnonzero(mine), dd): rk = key(i); dist[rk] = min(dist.get(rk, np.inf), float(x))
    rows = []
    for site in json.loads((d / "sites.json").read_text()):
        sk = (site["label_chain"], site["resnum"], site["icode"]); x = rsa.get(sk); dd = dist.get(sk)
        rows.append(dict(chain=site["chain"], resnum=site["resnum"], icode=site["icode"], group=site["group"],
                         rsa=None if x is None else round(x, 4), w_burial=None if x is None else round(float(burial_weight(x)), 4),
                         chain_distance_A=None if dd is None else round(dd, 2), w_interface=None if dd is None else round(float(interface_weight(dd)), 4)))
    tmp = d / "environment.json.tmp"; tmp.write_text(json.dumps(dict(version="resolved-v2", sites=rows))); tmp.rename(d / "environment.json"); done += 1
print("task", k, "wrote", done, flush=True)
