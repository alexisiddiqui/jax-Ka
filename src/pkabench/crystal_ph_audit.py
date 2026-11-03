"""Experimental (crystallisation / sample) pH of the pool-v4 structures (2026-10-12), for a pH auxiliary target.

PINDER AB.cif.gz files carry no experimental metadata and PINDER's metadata.parquet has no pH; pKPDB's database has an
exp_ph column from the same mmCIF fields, but its export (pkas.csv) does not include it. So pH comes from the RCSB
data API (GraphQL, batched) for every PINDER and pKPDB PDB id in pool-v4, chosen by the entry's method: crystal
methods (X-ray, neutron, electron / fibre crystallography) _exptl_crystal_grow.pH (pdbx_pH_range midpoint when pH is
missing); electron microscopy _em_buffer.pH (not in the GraphQL schema: read from the RCSB mmCIF header file, streamed,
nothing saved); NMR _pdbx_nmr_exptl_sample_conditions.pH. pH < 2.5 is kept in all_ph but given source "implausible"
(106 entries deposit 0.1, apparently a concentration). A sample of the local pKPDB deposited mmCIFs
(pkpdb-v1/structures) is parsed to check the API values.

Writes one table and one summary (no per-structure files): <runtime>/audits/crystal-ph-v1/{entries.tsv, summary.json}.

  python -m pkabench.crystal_ph_audit [--batch 400] [--check 500] [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import re
import shlex
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json

URL = "https://data.rcsb.org/graphql"
QUERY = """query($ids: [String!]!) { entries(entry_ids: $ids) { rcsb_id exptl { method }
  exptl_crystal_grow { pH pdbx_pH_range method } pdbx_nmr_exptl_sample_conditions { pH }
  rcsb_entry_info { resolution_combined } } }"""
HEADER = "https://files.rcsb.org/header/{}.cif"
CRYSTAL_METHODS = ("X-RAY DIFFRACTION", "NEUTRON DIFFRACTION", "ELECTRON CRYSTALLOGRAPHY", "FIBER DIFFRACTION", "POWDER DIFFRACTION")
MINIMUM_PH = 2.5
POOL = "training/pool-v4"
TABLES = {"pinder": "pinder.tsv", "pinder-val": "pinder-val.tsv", "pkpdb": "pkpdb.tsv", "pkpdb-val": "pkpdb-val.tsv"}
NUMBER = re.compile(r"\d+(?:\.\d+)?")
EDGES = np.arange(0.0, 14.5, 0.5)


def pool_ids(root):
    """{subset: [(structure id, pdb id)]} from the pool-v4 tables."""
    out = {}
    for subset, name in TABLES.items():
        with open(Path(root) / POOL / name) as handle:
            out[subset] = [(r["id"], r["pdb_id"].lower()) for r in csv.DictReader(handle, delimiter="\t")]
    return out


def fetch(ids, batch=400, retries=4):
    results = {}
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]; body = json.dumps({"query": QUERY, "variables": {"ids": chunk}}).encode()
        for attempt in range(retries):
            try:
                request = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=120) as response: data = json.load(response)
                break
            except Exception:
                if attempt == retries - 1: raise
                time.sleep(5 * 2 ** attempt)
        if data.get("errors"): raise RuntimeError(data["errors"][:3])
        for entry in data["data"]["entries"] or []: results[entry["rcsb_id"].lower()] = entry
    return results


def category_values(text, category, item):
    """Values of _category.item from mmCIF text, key-value or loop_ form (a minimal reader: quoted tokens, ;-text
    fields as one token)."""
    lines = text.splitlines(); key = f"_{category}.{item}"; i = 0; values = []
    while i < len(lines):
        line = lines[i]
        if line.startswith(key + " ") or line == key:
            rest = line[len(key):].strip(); values.append(rest if rest else lines[i + 1].strip()); i += 1; continue
        if line.strip() == "loop_":
            names = []; i += 1
            while i < len(lines) and lines[i].startswith("_"): names.append(lines[i].split()[0]); i += 1
            tokens = []
            while i < len(lines) and not lines[i].startswith(("_", "loop_", "#", "data_")):
                if lines[i].startswith(";"):
                    block = [lines[i][1:]]; i += 1
                    while i < len(lines) and not lines[i].startswith(";"): block.append(lines[i]); i += 1
                    tokens.append("\n".join(block)); i += 1; continue
                tokens.extend(shlex.split(lines[i], posix=True)); i += 1
            if key in names and names:
                column = names.index(key); values.extend(tokens[column::len(names)])
            continue
        i += 1
    return [v.strip("'\"") for v in values]


def em_buffer_ph(pdb_id, retries=4):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(HEADER.format(pdb_id.upper()), timeout=120) as response: text = response.read().decode(errors="replace")
            break
        except Exception:
            if attempt == retries - 1: return pdb_id, None, "fetch_failed"
            time.sleep(5 * 2 ** attempt)
    try: values = [float(v) for v in category_values(text, "em_buffer", "pH") if v not in ("?", ".")]
    except ValueError: return pdb_id, None, "parse_failed"
    return pdb_id, values, "ok"


def range_midpoint(text):
    values = [float(v) for v in NUMBER.findall(text or "")]
    values = [v for v in values if 0 < v <= 14]
    return float(np.mean([min(values), max(values)])) if values else None


def summarise_entry(pdb_id, entry, em=None):
    """One row: method, the pH of the experiment that produced the structure and its source. Crystal methods: first
    finite _exptl_crystal_grow.pH, else the pH range midpoint; EM: first _em_buffer.pH (em = (values, status)); NMR:
    first sample-condition pH. Mixed-method entries take the first of crystal, EM, NMR that has a value."""
    if entry is None: return dict(pdb_id=pdb_id, method="", ph="", source="not_in_rcsb", all_ph="", ph_range="", resolution="")
    methods = {e["method"] for e in entry.get("exptl") or []}; method = ";".join(sorted(methods))
    grow = entry.get("exptl_crystal_grow") or []
    crystal = [float(g["pH"]) for g in grow if g.get("pH") is not None]
    ranges = [g["pdbx_pH_range"] for g in grow if g.get("pdbx_pH_range")]
    nmr = [float(c["pH"]) for c in entry.get("pdbx_nmr_exptl_sample_conditions") or [] if c.get("pH") is not None]
    buffer, status = em if em is not None else ([], "not_fetched")
    buffer = buffer or []
    candidates = []
    if methods & set(CRYSTAL_METHODS):
        if crystal: candidates.append((crystal[0], "crystal_grow.pH"))
        elif ranges and range_midpoint(ranges[0]) is not None: candidates.append((range_midpoint(ranges[0]), "crystal_grow.pH_range"))
    if "ELECTRON MICROSCOPY" in methods and buffer: candidates.append((buffer[0], "em_buffer.pH"))
    if any("NMR" in m for m in methods) and nmr: candidates.append((nmr[0], "nmr_sample.pH"))
    ph, source = candidates[0] if candidates else (None, "missing" if status in ("ok", "not_fetched") else f"em_{status}")
    if ph is not None and ph < MINIMUM_PH: source = "implausible"
    resolution = (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []
    return dict(pdb_id=pdb_id, method=method, ph="" if ph is None or source == "implausible" else f"{ph:g}", source=source,
                all_ph=";".join(f"{v:g}" for v in crystal + buffer + nmr), ph_range="|".join(ranges),
                resolution=f"{resolution[0]:g}" if resolution else "")


def local_ph(path):
    """_exptl_crystal_grow.pH from a deposited mmCIF (key-value form only; loops return None), read up to _atom_site."""
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("_exptl_crystal_grow.pH "):
                value = line.split(None, 1)[1].strip(); return None if value in ("?", ".") else float(value)
            if line.startswith("_atom_site."): return None
    return None


def stats(values):
    v = np.asarray(values, float)
    if not len(v): return {"n": 0}
    q = np.quantile(v, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    rounded = Counter(np.round(v, 1).tolist())
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()), "max": float(v.max()),
            "quantiles": dict(zip(["1", "5", "10", "25", "50", "75", "90", "95", "99"], map(float, q))),
            "fraction_below": {str(t): float(np.mean(v < t)) for t in (4, 5, 6, 7, 7.5, 8, 9)},
            "histogram_0.5": {"edges": EDGES.tolist(), "counts": np.histogram(v, EDGES)[0].tolist()},
            "top_values": [[k, c] for k, c in rounded.most_common(15)]}


def run(root, batch=400, check=500, workers=8):
    root = Path(root); out = root / "audits" / "crystal-ph-v1"; out.mkdir(parents=True, exist_ok=True); began = time.time()
    pools = pool_ids(root); unique = sorted({p for rows in pools.values() for _, p in rows})
    entries = fetch(unique, batch)
    em_ids = [p for p in unique if p in entries and any(e["method"] == "ELECTRON MICROSCOPY" for e in entries[p].get("exptl") or [])]
    with ThreadPoolExecutor(workers) as pool: em = {p: (values, status) for p, values, status in pool.map(em_buffer_ph, em_ids)}
    rows = {p: summarise_entry(p, entries.get(p), em.get(p)) for p in unique}
    fields = ["pdb_id", "method", "ph", "source", "all_ph", "ph_range", "resolution"]
    tmp = out / "entries.tsv.tmp"
    with open(tmp, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t"); writer.writeheader(); writer.writerows(rows[p] for p in unique)
    os.replace(tmp, out / "entries.tsv")
    per_subset = {}
    for subset, members in pools.items():
        structures = [rows[p] for _, p in members]; pdbs = sorted({p for _, p in members})
        per_subset[subset] = {
            "structures": len(structures), "pdb_ids": len(pdbs),
            "source_by_structure": dict(Counter(r["source"] for r in structures)),
            "method_by_structure": dict(Counter(r["method"] for r in structures).most_common(6)),
            "multiple_distinct_ph_by_pdb": sum(len(set(rows[p]["all_ph"].split(";")) - {""}) > 1 for p in pdbs),
            "ph_by_structure": stats([float(r["ph"]) for r in structures if r["ph"]]),
            "ph_by_pdb": stats([float(rows[p]["ph"]) for p in pdbs if rows[p]["ph"]]),
            "crystal_ph_only_by_structure": stats([float(r["ph"]) for r in structures if r["source"] == "crystal_grow.pH"])}
    # spot check against local deposited mmCIFs (pKPDB set)
    structures = root / "pretraining" / "pkpdb-v1" / "structures"
    local = [p for p in unique if (structures / p[1:3] / f"{p}.cif.gz").exists()]
    sample = random.Random(17).sample(local, min(check, len(local))); agree = disagree = local_missing = 0; examples = []
    for p in sample:
        value = local_ph(structures / p[1:3] / f"{p}.cif.gz"); api = rows[p]
        if value is None: local_missing += 1; continue
        first = api["all_ph"].split(";")[0] if api["source"] == "crystal_grow.pH" else ""
        if first and abs(float(first) - value) < 1e-6: agree += 1
        else: disagree += 1; examples.append([p, value, api["all_ph"], api["source"]])
    summary = {"version": "crystal-ph-v1", "definition": __doc__.split("\n\n")[0], "pool": POOL, "unique_pdb_ids": len(unique),
               "returned_by_rcsb": len(entries),
               "em_headers": {"fetched": len(em_ids), "status": dict(Counter(status for _, status in em.values())),
                              "with_buffer_ph": sum(bool(values) for values, _ in em.values())},
               "minimum_ph": MINIMUM_PH, "subsets": per_subset,
               "local_check": {"sampled": len(sample), "agree": agree, "disagree": disagree, "local_missing_or_loop": local_missing,
                               "disagree_examples": examples[:20]},
               "seconds": round(time.time() - began, 1)}
    atomic_json(out / "summary.json", summary); return summary


def main(argv=None):
    p = argparse.ArgumentParser(prog="pkabench.crystal_ph_audit")
    p.add_argument("--batch", type=int, default=400); p.add_argument("--check", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)
    print(json.dumps(run(Path(os.environ["PKABENCH_RUNTIME"]), a.batch, a.check, a.workers), indent=1))


if __name__ == "__main__":
    main()
