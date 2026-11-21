"""M0 (b): pack the per-state goldens (m0/rec/<sid>.npz, one StateRecord each) into shards of ~50 states, complete
AB+A+B triples kept together, + a manifest with sha256 and gate summaries.
usage: python m0_pack.py <states.json> <rec_dir> <out_dir> [per_shard=50]"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from jaxpb import records as RC
states = json.load(open(sys.argv[1])); rec_dir, out = sys.argv[2], sys.argv[3]
per = int(sys.argv[4]) if len(sys.argv) > 4 else 50
os.makedirs(out, exist_ok=True)
cids = list(dict.fromkeys(s["cid"] for s in states))
have = {s["state_id"] for s in states if os.path.exists(os.path.join(rec_dir, s["state_id"] + ".npz"))}
shards = [[]]
for cid in cids:
    sids = [s["state_id"] for s in states if s["cid"] == cid and s["state_id"] in have]
    if len(shards[-1]) + len(sids) > per:
        shards.append([])
    shards[-1] += sids
man = {"shards": [], "missing": sorted({s["state_id"] for s in states} - have), "states": {}}
for k, sids in enumerate(shards):
    recs = [RC.load_states(os.path.join(rec_dir, sid + ".npz"))[0] for sid in sids]
    for r in recs:
        g = json.loads(r.ref["gate"])
        man["states"][r.state_id] = dict(shard=k, role=r.role, n_atoms=r.n_atoms, n_sites=r.n_sites, n_taus=r.n_taus,
                                         size_class=r.size_class, bitwise_equal=g.get("bitwise_equal"), pka_maxabs=g.get("pka_maxabs"),
                                         pb_wall_s=g.get("pb_wall_sum"), max_taus_site=int(np.diff(r.site_tau_off).max()),
                                         max_charged=int(max(np.count_nonzero(r.tau_charges(t)) for t in range(r.n_taus))),
                                         max_model_atoms=int(np.diff(r.model_off).max()))
    p = os.path.join(out, f"m0_goldens_{k:02d}.npz")
    sha = RC.save_states(p, recs)
    man["shards"].append(dict(path=os.path.basename(p), sha256=sha, n_states=len(recs), bytes=os.path.getsize(p), states=sids))
    print(p, len(recs), sha, flush=True)
v = list(man["states"].values())
man["summary"] = dict(n_states=len(v), n_triples=sum(1 for c in cids if all(f"{c}_{s}" in have for s in ("AB", "A", "B"))),
                      n_abag=sum(1 for x in v if x["role"] == "antibody_antigen"), all_bitwise=all(x["bitwise_equal"] for x in v),
                      max_pka_diff=max(x["pka_maxabs"] or 0 for x in v), taus=sum(x["n_taus"] for x in v),
                      classes={c: sum(1 for x in v if x["size_class"] == c) for c in range(5)},
                      max_taus_site=max(x["max_taus_site"] for x in v), max_charged=max(x["max_charged"] for x in v),
                      max_model_atoms=max(x["max_model_atoms"] for x in v), pb_wall_h=sum(x["pb_wall_s"] for x in v) / 3600)
json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(json.dumps(man["summary"], indent=1))
