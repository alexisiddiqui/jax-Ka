"""Read-only campaign audits; counterfactual prep never changes frozen inputs."""
from collections import Counter
import io
import json
import os
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdbx
from scipy.spatial import cKDTree

from .prep import CANONICAL, Rejection, clean_components, prepare_pair
from .runtime import atomic_json, digest, require_compute
from .schema import read_table


def component_inventory(atoms, cif, selected):
    """Inventory every forbidden residue, including distance to selected heavy atoms."""
    heavy = ~np.isin(np.char.upper(atoms.element), ["H", "D"])
    pair = np.isin(atoms.chain_id, selected) & heavy
    tree = cKDTree(atoms.coord[pair]) if pair.any() else None
    entries = []
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    for start, end in zip(starts[:-1], starts[1:]):
        residue = atoms[start:end]
        try:
            clean_components(residue, cif)
        except Rejection as exc:
            coords = residue.coord[heavy[start:end]]
            distance = float(tree.query(coords)[0].min()) if tree is not None and len(coords) else None
            entries.append({"chain": str(residue.chain_id[0]), "resnum": int(residue.res_id[0]),
                "name": str(residue.res_name[0]), "code": exc.code,
                "in_selected_chain": str(residue.chain_id[0]) in selected,
                "min_pair_distance": distance, "start": int(start), "end": int(end)})
    return entries


def audit_prep(campaign, out):
    require_compute(); campaign = Path(campaign); out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((campaign / "manifest.json").read_text())
    rejected = {r["candidate_id"]: r for r in read_table(campaign / "rejections.parquet")}
    accepted = {r["complex_id"] for r in read_table(campaign / "structures.parquet")}
    results = []
    for row in manifest["candidates"]:
        cid = row["complex_id"]; source = campaign / "structures" / cid / "source.cif"
        cif = pdbx.CIFFile.read(source)
        atoms = pdbx.get_structure(cif, model=1, altloc="occupancy", use_author_fields=False, include_bonds=True)
        selected = row["partner_A_chains"] + row["partner_B_chains"]
        components = component_inventory(atoms, cif, selected)
        poly = cif.block.get("entity_poly")
        peptide_entities = set() if poly is None else {str(e) for e, t in zip(poly["entity_id"].as_array(str), poly["type"].as_array(str)) if "polypeptide" in str(t)}
        asym = cif.block.get("struct_asym")
        protein_chains = sorted(set(atoms.chain_id[np.isin(atoms.res_name, list(CANONICAL))]))
        if asym is not None and peptide_entities:
            protein_chains = sorted(set(str(c) for c, e in zip(asym["id"].as_array(str), asym["entity_id"].as_array(str)) if str(e) in peptide_entities) & set(atoms.chain_id))
        record = {"complex_id": cid, "pdb_id": row["pdb_id"], "selected_chains": selected,
            "source_sha256": digest(source), "original": "accepted" if cid in accepted else rejected.get(cid, {}).get("code", "pipeline_error"),
            "protein_chains": protein_chains, "strict_binary_eligible": set(protein_chains) == set(selected),
            "components": components, "counterfactuals": {}}
        # Filtering atom_site retains original label/author IDs, occupancy, and CCD metadata.
        # pair_only is intentionally unsafe as production: ligands often use separate chains.
        for mode in ("pair_only_upper_bound", "pair_plus_nearby_forbidden_10A"):
            keep_chains = set(selected)
            if mode.endswith("10A"):
                keep_chains.update(c["chain"] for c in components if c["min_pair_distance"] is not None and c["min_pair_distance"] <= 10)
            filtered = pdbx.CIFFile.read(source)
            category = filtered.block["atom_site"]
            mask = np.isin(category["label_asym_id"].as_array(str), list(keep_chains))
            new_category = pdbx.CIFCategory()
            for name in category:
                new_category[name] = pdbx.CIFColumn(category[name].as_array(str)[mask])
            filtered.block["atom_site"] = new_category
            work = out / cid / mode; work.mkdir(parents=True)
            try:
                structure, sites = prepare_pair(filtered, {**row, "source_sha256": digest(source)}, work,
                    manifest["prep_config"].get("completion_executable"), audit_allow_subcomplex=True)
                result = {"status": "accepted", "n_residues": structure["n_residues"], "n_sites": len(sites)}
            except Rejection as exc:
                result = {"status": "rejected", "code": exc.code, "detail": str(exc)}
            except Exception as exc:
                import traceback
                result = {"status": "pipeline_error", "detail": str(exc), "traceback": traceback.format_exc()}
            record["counterfactuals"][mode] = result
        results.append(record)
        atomic_json(out / "progress.json", {"done": len(results), "total": len(manifest["candidates"])})
    summary = {"campaign": str(campaign), "manifest_sha256": digest(campaign / "manifest.json"),
        "audit_sha256": digest(Path(__file__)), "job": os.environ["SLURM_JOB_ID"], "node": os.environ["SLURMD_NODENAME"],
        "candidates": len(results), "original_histogram": dict(Counter(r["original"] for r in results)),
        "strict_binary_candidates": sum(r["strict_binary_eligible"] for r in results),
        "original_accepted_strict_binary": sum(r["strict_binary_eligible"] and r["original"] == "accepted" for r in results),
        "counterfactual_histograms": {mode: dict(Counter(r["counterfactuals"][mode].get("code", r["counterfactuals"][mode]["status"]) for r in results)) for mode in ("pair_only_upper_bound", "pair_plus_nearby_forbidden_10A")},
        "warning": "Diagnostic counterfactuals only. No policy change or production eligibility; 10 A proximity is not proof of chemical independence.",
        "candidates_detail": results}
    atomic_json(out / "audit.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "candidates_detail"}, indent=2))


def audit_sources(out):
    require_compute(); out = Path(out); out.mkdir(parents=True, exist_ok=False)
    receipts = []
    for repo in ("mms-fcul/PypKa-Server-Back", "mms-fcul/PypKa-Server-Front"):
        receipt = {"repository": repo}
        try:
            with urllib.request.urlopen(f"https://api.github.com/repos/{repo}/commits?per_page=1", timeout=45) as stream:
                sha = json.load(stream)[0]["sha"]
            url = f"https://codeload.github.com/{repo}/tar.gz/{sha}"
            with urllib.request.urlopen(url, timeout=90) as stream:
                raw = stream.read(30 * 1024 * 1024 + 1)
            if len(raw) > 30 * 1024 * 1024: raise ValueError("source archive exceeds audit limit")
            name = repo.split("/")[-1]; archive = out / f"{name}.tar.gz"; archive.write_bytes(raw)
            target = out / name
            with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
                for member in tar.getmembers():
                    relative = Path(*Path(member.name).parts[1:])
                    if not member.isfile() or ".." in relative.parts or relative.is_absolute(): continue
                    dest = target / relative; dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(tar.extractfile(member).read())
            receipt.update(commit=sha, url=url, sha256=digest(archive), status="downloaded")
        except Exception as exc:
            receipt.update(status="error", error=str(exc))
        receipts.append(receipt)
    atomic_json(out / "receipts.json", receipts)
    print(json.dumps(receipts, indent=2))


def audit_teacher(source_root, out):
    """Fetch only precomputed records; /pkas and /pKAI may launch jobs and are avoided."""
    import ast
    from .gates import compare_deposit
    from .adapters.base import TEACHER
    require_compute(); source_root = Path(source_root); out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    path = source_root / "PypKa-Server-Back/server/const.py"
    module = ast.parse(path.read_text())
    settings = next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PKPDB_PARAMS" for t in n.targets))
    atomic_json(out / "server-pkpdb-params.json", settings)
    report = {"source": str(path), "source_sha256": digest(path),
        "source_receipts": str(source_root / "receipts.json"), "source_comparison": compare_deposit(settings, TEACHER),
        "historical_ser_thr_titration": settings.get("ser_thr_titration"), "declared_historical_version": settings.get("version"),
        "evidence_limit": "Server PKPDB_PARAMS is a shared metadata constant, not a per-simulation sim_settings export. Missing values are not inferred from defaults.",
        "queries": []}
    for pdb_id in ("4lzt", "1ubq"):
        url = f"https://api.pypka.org/query/{pdb_id}"
        receipt = {"url": url}
        try:
            with urllib.request.urlopen(url, timeout=40) as response:
                raw = response.read(2 * 1024 * 1024 + 1)
                receipt.update(status=response.status, resolved_url=response.url)
            if len(raw) > 2 * 1024 * 1024: raise ValueError("response exceeds audit limit")
            saved = out / f"{pdb_id}.json"; saved.write_bytes(raw)
            data = json.loads(raw); params = data.get("params")
            if isinstance(params, str): params = ast.literal_eval(params)
            receipt.update(file=str(saved), sha256=digest(saved), sites=len(data.get("pKas", [])),
                groups=sorted({r[1] for r in data.get("pKas", [])}))
            if isinstance(params, dict):
                receipt.update(settings=params, agrees_with_source=params == settings, comparison=compare_deposit(params, TEACHER))
            else: receipt["missing"] = "No settings in response"
        except Exception as exc:
            receipt["error"] = str(exc)
        report["queries"].append(receipt)
    report["G2"] = "unresolved"
    atomic_json(out / "audit.json", report)
    atomic_json(Path(os.environ['PKABENCH_RUNTIME']) / 'sources/teacher-audit.json',
        {'path': str((out / 'audit.json').resolve()), 'sha256': digest(out / 'audit.json')})
    print(json.dumps(report, indent=2))


def audit_findings(campaign, prep_audit, out):
    """Collect counterfactuals and corrected teacher validation without rerunning methods."""
    from .linkage import linkage
    from .schema import GROUPS
    require_compute(); campaign = Path(campaign); prep_audit = Path(prep_audit)
    audit = json.loads(prep_audit.read_text()); original = Path(audit['campaign'])
    before = json.loads((original/'manifest.json').read_text())
    after = json.loads((campaign/'manifest.json').read_text())
    same = [r['complex_id'] for r in before['candidates']] == [r['complex_id'] for r in after['candidates']]
    if not same: raise ValueError('corrective campaign changed sampled identities')
    gates = json.loads((campaign/'gates.json').read_text())
    teacher = json.loads(Path(gates['G1']['evidence']).read_text())
    predictions = read_table(Path(gates['G1']['evidence']).with_suffix('.parquet'))
    sites = read_table(campaign/'structures'/gates['G1']['complex_id']/'sites.parquet')
    states = {}
    for state in ('AB', 'A', 'B'):
        import re
        result = json.loads((Path(teacher['workdir'])/state/'result.json').read_text())
        resolved = Path(teacher['workdir'])/state/'resolved-config.txt'
        flags = re.findall(r"'ser_thr_titration': (True|False)\b", resolved.read_text())
        if len(flags) != 1: raise ValueError('missing or ambiguous resolved SER/THR setting')
        states[state] = {'ser_thr_titration': flags[0] == 'True', 'resolved_config_sha256': digest(resolved),
            'unexpected_groups': sorted({r['group'] for r in result['rows']} - set(GROUPS)),
            'native_curves': all(r.get('curve_source') == 'native' and len(r.get('curve', [])) == 73 for r in result['rows'])}
    validation = all(s['ser_thr_titration'] is False and not s['unexpected_groups'] and s['native_curves'] for s in states.values())
    chemistry = []
    for row in audit['candidates_detail']:
        result = row['counterfactuals']['pair_plus_nearby_forbidden_10A']
        if row['original'] != 'accepted' and result['status'] == 'accepted':
            chemistry.append({k: row[k] for k in ('complex_id', 'pdb_id', 'strict_binary_eligible', 'components')})
    accepted = read_table(campaign/'structures.parquet')
    summary = {"same_50_candidate_identities": same, "corrected_campaign": str(campaign),
        "prep_audit": str(prep_audit), "prep_audit_sha256": digest(prep_audit),
        "strict_binary_candidates": audit['strict_binary_candidates'],
        "corrected_curation": json.loads((campaign/'curation_report.json').read_text()),
        "corrected_rejection_histogram": dict(Counter(r['code'] for r in read_table(campaign/'rejections.parquet'))),
        "original_accepted_removed": [r['pdb_id'] for r in audit['candidates_detail'] if r['original'] == 'accepted' and not r['strict_binary_eligible']],
        "nearby_component_policy_recoveries": chemistry,
        "accepted": [{k: r[k] for k in ('complex_id','pdb_id','n_residues')} for r in accepted],
        "teacher_correction_validation": {'status': 'pass' if validation else 'fail', 'states': states,
            'complex_id': gates['G1']['complex_id'], 'linkage': linkage(predictions, sites)['status'],
            'missing_curve_rows': [{k: r[k] for k in ('chain', 'resnum', 'icode', 'group', 'state', 'status')}
                for r in predictions if r['curve'] is None or r['status'] in ('failed', 'not_reported')]},
        "G2": gates['G2'], "production_allowed": False,
        "recommendation": "Keep chemical exclusions. Establish a curated binary protein-only candidate pool before scaling; obtain historical per-simulation settings/version provenance or explicitly separate the new teacher from pKPDB.",
        "job": os.environ['SLURM_JOB_ID'], "node": os.environ['SLURMD_NODENAME']}
    atomic_json(Path(out), summary)
    print(json.dumps(summary, indent=2))
    if not validation: raise RuntimeError('corrected teacher validation failed')
