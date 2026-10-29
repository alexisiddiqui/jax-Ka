"""Check the default all-residue export against its training run (2026-10-21, user: "make this the default weights and
endpoint. Update the CLI etc to be able to handle this").

For PINDER validation complexes: AB.cif.gz -> pkanet.features.featurise (the ogqt predict CLI's cache graph) ->
pkanet.allres outputs, bound and free states, with the indexed-attention backend given on the command line. Site pKa
are matched to the training run's stored selected-epoch validation predictions (predictions-pinder-epoch-017.csv:
predicted_ab, predicted_free) by (chain, residue number, insertion code, group), the store's query keys being rebuilt
with the training topology rule (load_topology, gap cap, frozen disulfides). The export is load_default()'s
(GQT_DEFAULT_MODEL selects another pinned all-residue export; its run and epoch come from model.json). Run under
scripts/sqfs_run.sh.
  python check_allres_default.py OUT.json BACKEND(jax|triton) [N_COMPLEXES]
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from pkanet.attention_backend import set_backend

OUT = Path(sys.argv[1]); BACKEND = sys.argv[2]; LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 100
set_backend(BACKEND)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jaxpropka.parameters import GROUPS  # noqa: E402
from jaxpropka.topology import load_topology  # noqa: E402
from pkanet.allres import graph_from_features, state_graph  # noqa: E402
from pkanet.features import featurise  # noqa: E402
from pkatrain.gqt_paired_pinder import _read_cif_gz  # noqa: E402
from pkatrain.production_defaults import load_default  # noqa: E402
from pkatrain.production_graphs import PINDER, ProductionStore, output, read, source  # noqa: E402
from pkatrain.production_loading import select  # noqa: E402

ROOT = Path(os.environ["PKABENCH_RUNTIME"]); model = load_default()
run = ROOT / "training/gqt-production-v2/runs" / model.info["run"]
stored = {}
for r in csv.DictReader((run / f"predictions-pinder-epoch-{model.info['selected_epoch']:03d}.csv").open()):
    stored[(r["complex_id"], int(r["site"]))] = (float(r["predicted_ab"]), float(r["predicted_free"]))
manifest = read(output(ROOT, "pinder") / "manifest-v1.json"); store = ProductionStore(Path(manifest["store"]))
fn = {s: jax.jit(lambda g, s=s: model.outputs(state_graph(g, s))) for s in ("bound", "free")}
diffs = {"bound": [], "free": []}; unmatched = 0; matched = 0; complexes = 0; began = time.time()
for rec in select(manifest, "val")[:LIMIT]:
    cid = rec["id"]; path = source(ROOT, f"{PINDER}/entries/{cid}") / "AB.cif.gz"
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cif", dir=os.environ.get("TMPDIR")) as tmp:
        import gzip
        tmp.write(gzip.decompress(path.read_bytes())); tmp.flush()
        graph, metadata = featurise(tmp.name)
    g = {k: jnp.asarray(v) for k, v in graph_from_features(graph).items()}
    out = {s: jax.device_get(fn[s](g)) for s in fn}
    cli = {(s["chain"], int(s["number"]), s["insertion"].strip(), s["group"]): k for k, s in enumerate(metadata["sites"])}
    keys = load_topology(_read_cif_gz(path), gap_policy="cap", freeze_disulfides=True).keys; raw = store.raw(cid)
    for i, (r, grp) in enumerate(zip(raw["query_residue"], raw["query_group"])):
        k = keys[int(r)]; key = (str(k.chain), int(k.number), str(k.insertion).strip(), GROUPS[int(grp)])
        if key not in cli or (cid, i) not in stored: unmatched += 1; continue
        j = cli[key]; ab, fr = stored[(cid, i)]; matched += 1
        diffs["bound"].append(abs(float(out["bound"]["query_pka"][j]) - ab)); diffs["free"].append(abs(float(out["free"]["query_pka"][j]) - fr))
    complexes += 1
report = {"model": str(model.path), "run": model.info["run"], "backend": BACKEND, "device": str(jax.devices()[0]), "complexes": complexes, "matched_sites": matched, "unmatched_sites": unmatched,
          "seconds": round(time.time() - began, 1),
          **{f"{s}_abs_diff": {"max": float(np.max(d)), "mean": float(np.mean(d)), "frac_lt_1e-3": float(np.mean(np.asarray(d) < 1e-3)),
                               "frac_lt_1e-2": float(np.mean(np.asarray(d) < 1e-2))} for s, d in diffs.items()}}
OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=1) + "\n"); print(json.dumps(report, indent=1))
