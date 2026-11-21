"""quartet-rank-v1 Part A (2026-11-15): score_masked2.py, unchanged, on runs whose free edge head (aux_redge hidden / out) is
replaced by an external head file -- no run folder is written.

A run argument NAME=RUN@HEAD.npz loads RUN's selected checkpoint (pkatrain.production_train.selected_model via score_decoy) and
replaces params["aux_redge"]["hidden"]["w" / "b"] and ["out"]["w" / "b"] by the npz keys hidden_w / hidden_b / out_w / out_b
(edge-contrib-v1 baseline-edge.npz format, pairing-rank-v1 params.npz). The pair normalisation aux_redge["norm"], the encoder
and every other head stay RUN's. Shapes and dtypes must match exactly. Every file read for a NAME=RUN@HEAD run (protocol,
selection, pKa predictions, checkpoint) is RUN's own (score_decoy.run_dir strips the @HEAD suffix), so the pKa columns are
the control's by construction. NAME=RUN (no @) is scored as is. The grafts are recorded in OUT/graft.json: head path, sha256,
the head's own selection.json beside it (if present) and the max |head - RUN's head| per array (0 for a parity graft).

  python score_graft.py OUT_DIR STORE NAME=RUN[@HEAD.npz] [...] --modes unmasked      (score_masked2.py arguments)

The functions split_run / load_head / graft_params are importable (tests); the patching runs only as __main__.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HEAD_KEYS = ("hidden_w", "hidden_b", "out_w", "out_b")
SEP = "@"


def split_run(run):
    """'RUN@/path/head.npz' -> ('RUN', '/path/head.npz'); 'RUN' -> ('RUN', None)."""
    base, sep, head = run.partition(SEP)
    if sep and not head: raise ValueError(f"empty head path in {run!r}")
    return base, (head or None)


def load_head(path):
    with np.load(path, allow_pickle=False) as f: head = {k: np.asarray(f[k]) for k in f.files}
    if set(head) != set(HEAD_KEYS): raise ValueError(f"{path}: keys {sorted(head)} != {sorted(HEAD_KEYS)}")
    return head


def current_head(params):
    e = params["aux_redge"]
    return {"hidden_w": e["hidden"]["w"], "hidden_b": e["hidden"]["b"], "out_w": e["out"]["w"], "out_b": e["out"]["b"]}


def graft_params(params, head):
    """(params with aux_redge hidden / out replaced by head, {key: max |head - current|}); norm and everything else untouched."""
    import jax.numpy as jnp
    e = params["aux_redge"]
    if set(e) != {"norm", "hidden", "out"} or set(e["hidden"]) != {"w", "b"} or set(e["out"]) != {"w", "b"}:
        raise ValueError(f"unexpected aux_redge structure: { {k: sorted(v) if isinstance(v, dict) else None for k, v in e.items()} }")
    cur = {k: np.asarray(v) for k, v in current_head(params).items()}; diff = {}
    for k in HEAD_KEYS:
        if cur[k].shape != head[k].shape or cur[k].dtype != head[k].dtype:
            raise ValueError(f"{k}: head {head[k].shape} {head[k].dtype} != checkpoint {cur[k].shape} {cur[k].dtype}")
        diff[k] = float(np.max(np.abs(cur[k].astype(np.float64) - head[k].astype(np.float64)))) if cur[k].size else 0.0
    new = {**e, "hidden": {"w": jnp.asarray(head["hidden_w"]), "b": jnp.asarray(head["hidden_b"])},
           "out": {"w": jnp.asarray(head["out_w"]), "b": jnp.asarray(head["out_b"])}}
    return {**params, "aux_redge": new}, diff


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def graft_record(run, head_path, diff):
    sel = Path(head_path).with_name("selection.json")
    return {"run": run, "base_run": split_run(run)[0], "head": str(head_path), "head_sha256": sha256(head_path), "max_abs_diff_vs_base_head": diff,
            "head_selection": json.loads(sel.read_text()) if sel.is_file() else None}


def install(sd, out_dir):
    """Patch score_decoy's run_dir / selected_model (score_masked2 reads both as sd.<name>); returns the graft record dict."""
    records = {}; _run_dir, _selected = sd.run_dir, sd.selected_model; out_dir = Path(out_dir)

    def run_dir(root, run): return _run_dir(root, split_run(run)[0])

    def selected_model(root, run):
        base, head_path = split_run(run); params, fns = _selected(root, base)
        if head_path is None: return params, fns
        params, diff = graft_params(params, load_head(head_path)); records[run] = graft_record(run, head_path, diff)
        out_dir.mkdir(parents=True, exist_ok=True); tmp = out_dir / f"graft.json.tmp.{os.getpid()}"
        tmp.write_text(json.dumps(records, indent=1) + "\n"); tmp.replace(out_dir / "graft.json")
        print(f"[graft] {run}: max |head - base head| {json.dumps(diff)} sha256 {records[run]['head_sha256'][:16]}", flush=True)
        return params, fns
    sd.run_dir = run_dir; sd.selected_model = selected_model
    return records


if __name__ == "__main__":
    import runpy
    import tempfile
    os.environ.setdefault("GQT_PARAMETER_SCALING", "1")
    here = Path(__file__).resolve().parent; argv = sys.argv[1:]
    if len(argv) < 3: raise SystemExit(__doc__)
    for a in argv[2:]:
        if "=" in a and not a.startswith("--"):
            _, head = split_run(a.split("=", 1)[1])
            if head is not None: load_head(head)  # fail before any scoring if a head file is unreadable or malformed
    saved = sys.argv; sys.argv = [str(here / "score_decoy.py"), tempfile.mkdtemp(prefix="score_decoy_import_")]  # as score_masked2 does
    sys.path.insert(0, str(here))
    import score_decoy  # noqa: E402  (zero runs at import)
    sys.argv = saved
    install(score_decoy, argv[0])
    sys.argv = [str(here / "score_masked2.py"), *argv]
    runpy.run_path(str(here / "score_masked2.py"), run_name="__main__")
