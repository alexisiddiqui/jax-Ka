"""Compare PINDER's stored 10 A interface zone with residue dSASA > 10 A2."""
import concurrent.futures
import json
import os
from collections import Counter
from pathlib import Path

from pkabench.runtime import atomic_json, require_compute

MAX_ASA = dict(ALA=129, ARG=274, ASN=195, ASP=193, CYS=167, GLN=225, GLU=223, GLY=104,
    HIS=224, ILE=197, LEU=201, LYS=236, MET=224, PHE=240, PRO=159, SER=155, THR=172,
    TRP=285, TYR=263, VAL=174)
ALIASES = {"NTR": "NTERM", "CTR": "CTERM"}


def _keys(rows):
    return {(str(c), int(n), str(i), ALIASES.get(str(g), str(g))) for c, n, i, g, value in rows
            if value is not None}


def one(folder):
    folder = Path(folder); sites = json.loads((folder / "sites.json").read_text())
    labels = json.loads((folder / "labels.json").read_text())["pkai"]
    available = {state: _keys(rows) for state, rows in labels.items()}
    residue_type = {}
    for site in sites:
        if site["group"] in MAX_ASA:
            residue_type[(site["chain"], site["resnum"], site["icode"])] = site["group"]
    count = Counter()
    for site in sites:
        key = (site["chain"], site["resnum"], site["icode"], site["group"])
        paired = key in available["AB"] and key in available[site["partner"]]
        cohorts = ["all"]
        if site["train_mask"]: cohorts.append("train")
        if site["eval_mask"]: cohorts.append("eval")
        if site["train_mask"] and paired: cohorts.append("train_labelled_pkai")
        if site["eval_mask"] and paired: cohorts.append("eval_labelled_pkai")
        stored = bool(site["interface"])
        distance = bool(site["partner_distance_A"] <= 10)
        restype = site["group"] if site["group"] in MAX_ASA else residue_type.get(key[:3])
        known = restype is not None and site["rsa_free"] is not None and site["rsa_bound"] is not None
        dsasa = known and (site["rsa_free"] - site["rsa_bound"]) * MAX_ASA[restype] > 10
        for cohort in cohorts:
            count[cohort+"|sites"] += 1; count[cohort+"|stored_interface"] += stored
            count[cohort+"|distance_recomputed"] += distance
            count[cohort+"|distance_flag_disagreement"] += stored != distance
            if not known: count[cohort+"|dsasa_unknown"] += 1; continue
            count[cohort+"|dsasa_known"] += 1; count[cohort+"|dsasa"] += dsasa
            count[cohort+"|both"] += stored and dsasa
            count[cohort+"|stored_only"] += stored and not dsasa
            count[cohort+"|dsasa_only"] += dsasa and not stored
    return count


def main():
    runtime = Path(os.environ["PKABENCH_RUNTIME"]); threads = int(os.environ["SLURM_CPUS_PER_TASK"])
    require_compute(threads=threads, allow_comp1400=True)
    base = runtime / "pretraining/pinder-pkai-v1/entries"
    folders = [path for path in base.iterdir() if (path / "sites.json").exists()]
    total = Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(threads, 16)) as pool:
        for index, result in enumerate(pool.map(one, folders, chunksize=32), 1):
            total.update(result)
            if index % 10000 == 0: print(json.dumps({"entries": index, "total": len(folders)}), flush=True)
    cohorts = {}
    for cohort in ("all", "train", "eval", "train_labelled_pkai", "eval_labelled_pkai"):
        cohorts[cohort] = {name: total[cohort+"|"+name] for name in
            ("sites", "stored_interface", "distance_recomputed", "distance_flag_disagreement", "dsasa",
             "dsasa_known", "dsasa_unknown", "both", "stored_only", "dsasa_only")}
    pkpdb = {}
    for name in ("pkpdb-5k-v3", "pkpdb-full-v1"):
        pilot = json.loads((runtime / "pretraining" / name / "pilot.json").read_text())
        verify = json.loads((runtime / "pretraining" / name / "verification.json").read_text())
        pkpdb[name] = {"structures": verify["structures"], "raw_sites": pilot["raw_sites"],
            "clean_sites": pilot["clean_sites"], "interface_filter_used": False}
    out = runtime / "audits/interface-definition-v1"; out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "summary.json", {"pinder": cohorts, "pkpdb": pkpdb,
        "distance_definition": "stored interface was generated from in_interface_zone (minimum partner distance <= 10 A); later partner_distance_A is audited separately",
        "dsasa_definition": "(rsa_free-rsa_bound)*Tien2013_max_ASA > 10 A2; RSA stored to 4 decimals",
        "unknown_dsasa": "terminal sites on residues without another titratable token; reported separately",
        "entries": len(folders), "test_data_included": True})


if __name__ == "__main__": main()
