"""Versioned long tables; site identities are never inferred from row ordering."""
import os
from pathlib import Path
import tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PH = np.linspace(-2, 16, 73)
KEY = ("complex_id", "chain", "resnum", "icode", "group")
GROUPS = ("ASP", "GLU", "HIS", "CYS", "TYR", "LYS", "ARG", "NTERM", "CTERM")
NULL_PKA = dict(zip(GROUPS, (3.8, 4.5, 6.5, 9., 10., 10.5, 12.5, 8., 3.2)))
STATUSES = {"ok", "out_of_range", "not_titrating", "not_reported", "failed"}
S = pa.string(); F = pa.float64(); I = pa.int64(); B = pa.bool_()
SITE_KEY = [("complex_id", S), ("chain", S), ("resnum", I), ("icode", S), ("group", S)]
SCHEMAS = {
    "structures": pa.schema([("complex_id", S), ("pdb_id", S), ("assembly", S),
        ("partner_A_chains", pa.list_(S)), ("partner_B_chains", pa.list_(S)),
        ("n_residues", I), ("homomeric", B), ("antibody", B), ("resolution", F),
        ("exp_method", S), ("crystallization_ph", F), ("provenance", S),
        ("content_sha256", S), ("split", S), ("component_id", S)]),
    "sites": pa.schema(SITE_KEY + [("restype", S), ("partner", S),
        ("residue_delta_sasa", F), ("functional_delta_sasa", F), ("functional_atoms_complete", B),
        ("min_partner_distance", F), ("in_interface_zone", B), ("is_break_terminus", B), ("was_completed", B),
        ("coordinates_observed", B), ("defect_clearance", F), ("supervision_mask", B),
        ("supervision_mask_10", B), ("supervision_mask_15", B), ("supervision_mask_20", B),
        ("supervision_mask_adaptive_10", B), ("supervision_mask_adaptive_15", B), ("supervision_mask_adaptive_20", B),
        ("alternate_conformation_uncertain", B)]),
    "predictions": pa.schema(SITE_KEY + [("state", S), ("method", S), ("method_version", S),
        ("config_sha256", S), ("pka", F), ("status", S), ("curve", pa.list_(pa.float32())),
        ("curve_source", S), ("intrinsic_pka", F)]),
    "pairs": pa.schema([("complex_id", S), ("state", S), ("site_i", S), ("site_j", S), ("w", F)]),
    "rejections": pa.schema([("candidate_id", S), ("stage", S), ("code", S), ("detail", S)]),
}


def key(row):
    return tuple(row[x] for x in KEY)


def write_table(path, name, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if name == "predictions":
        seen = set()
        for row in rows:
            identity = key(row) + (row["state"], row["method"])
            if identity in seen:
                raise ValueError(f"duplicate prediction {identity}")
            seen.add(identity)
            if row["status"] not in STATUSES or row["group"] not in GROUPS:
                raise ValueError("unknown prediction status/group")
            if row["status"] == "ok" and (row.get("pka") is None or not np.isfinite(row["pka"])):
                raise ValueError("ok requires a finite midpoint")
            curve = row.get("curve")
            if curve is not None:
                curve = np.asarray(curve)
                if curve.shape != (73,) or not np.isfinite(curve).all() or np.any((curve < 0) | (curve > 1)):
                    raise ValueError("invalid protonated-fraction curve")
                if row.get("curve_source") not in ("native", "hh"):
                    raise ValueError("curve source required")
    table = pa.Table.from_pylist(rows, schema=SCHEMAS[name])
    table = table.replace_schema_metadata({b"pkabench_schema": b"2", b"ph_grid": b"-2:16:0.25"})
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pending-"); os.close(fd)
    try:
        pq.write_table(table, tmp); os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_table(path):
    return pq.read_table(path).to_pylist()
