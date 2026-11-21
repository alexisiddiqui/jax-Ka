"""M0 (a): golden pack of the 5 pilot states: StateRecords from the jaxpb-pilot-v1 DelPhi recordings (rec.npz) +
saved mc-energies/result/gate JSONs + the jaxpb-pilot-v2 outputs (final.npz, reordered to MC order) as v2_* arrays.
usage: python m0_goldens.py <v1_ref_dir> <v2_out_dir> <pilot_json_dir> <out.npz>"""
import sys, json, os
import numpy as np
from jaxpb import records as RC

STATES = [("d0d5c619bd23b13d_B", "antibody_antigen"), ("d0d5c619bd23b13d_A", "antibody_antigen"),
          ("24d02660df3ac695_A", "general"), ("d0d5c619bd23b13d_AB", "antibody_antigen"), ("d6004cf2ee4008d2_AB", "general")]
ref_dir, v2_dir, jdir, out = sys.argv[1:5]
if len(sys.argv) > 5:
    STATES = [x for x in STATES if x[0] in sys.argv[5].split(",")]
REC_NAME = os.environ.get("JPP_REC_NAME", "{sid}.rec.npz")      # Coulson layout: {sid}/rec.npz
recs = []; extra = {}; v2 = {k: [] for k in ("esolvM", "esolvS", "esolvM0", "sitpotM", "it", "res")}
info = {}
for sid, role in STATES:
    r = RC.from_pilot_rec(os.path.join(ref_dir, REC_NAME.format(sid=sid)), sid, role)
    for k, f in (("mc_energies", "mc-energies.json"), ("result", "result.json"), ("gate", "gate.json")):
        r.ref[k] = open(os.path.join(jdir, f"{sid}.{f}")).read()
    r.ref["pairs"] = np.zeros((0, 2), np.int64)   # not recorded by the pilot (derive from mc-energies)
    r.validate()
    V = np.load(os.path.join(v2_dir, sid + ".final.npz"))
    o = r.tau_delphi
    for k in v2:
        v2[k].append(np.asarray(V[k])[o])
    recs.append(r)
    info[sid] = dict(n_atoms=r.n_atoms, n_surface=r.n_surface_atoms, n_sites=r.n_sites, n_taus=r.n_taus, size_class=r.size_class,
                     max_taus_per_site=int(np.diff(r.site_tau_off).max()), max_site_atoms=int(np.diff(r.site_atoms_off).max()),
                     max_model_atoms=int(np.diff(r.model_off).max()),
                     max_charged_atoms=int(max(np.count_nonzero(r.tau_charges(t)) for t in range(r.n_taus))))
for k, lst in v2.items():
    a = [x.reshape(-1) for x in lst]
    cat, off = RC._ragged_concat(a)
    extra["v2_" + k] = cat; extra["v2_" + k + "__off"] = off
    extra["v2_" + k + "__shape"] = np.array([list(x.shape) + [1] * (2 - x.ndim) for x in lst])
sha = RC.save_states(out, recs, extra=extra)
info["sha256"] = sha; info["path"] = out
json.dump(info, open(out.replace(".npz", ".json"), "w"), indent=1)
print(json.dumps(info, indent=1))
