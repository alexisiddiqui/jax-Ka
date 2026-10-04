"""Canonical heavy-atom preparation, immutable states, and rejection accounting."""
import csv
import io
import json
from pathlib import Path
import subprocess
import tarfile
import numpy as np
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile
from jaxpropka.topology import load_topology
from jaxpropka.geometry import _template
from jaxpropka.parameters import THREE
from .annotate import annotate, SITE_ATOMS
from .runtime import atomic_json, config_hash, digest, require_compute
from .schema import GROUPS, write_table

CANONICAL = set(THREE)
NONMETALS = {"H", "D", "C", "N", "O", "P", "S", "SE", "F", "CL", "BR", "I"}


class Rejection(ValueError):
    def __init__(self, code, detail, stage="prep"):
        super().__init__(detail); self.code = code; self.stage = stage


def topology(atoms):
    try:
        return load_topology(atoms, gap_policy="cap", freeze_disulfides=True)
    except ValueError as exc:
        messages = {"missing backbone": "missing_backbone", "ambiguous repeated": "ambiguous_residue_key",
            "duplicate atom": "ambiguous_residue_key", "possible cyclic": "cyclic_peptide",
            "ambiguous disulfide": "ambiguous_disulfide", "unsupported covalent": "covalent_crosslink"}
        for prefix, code in messages.items():
            if str(exc).startswith(prefix): raise Rejection(code, str(exc)) from exc
        raise


def write_cif(path, atoms):
    file = pdbx.CIFFile(); pdbx.set_structure(file, atoms); file.write(path)


def read_cif(path):
    return pdbx.get_structure(pdbx.CIFFile.read(path), model=1, altloc="occupancy", use_author_fields=True, include_bonds=True)


def export_pdb(atoms, path):
    """Renumber observed heavy atoms only; retain original chain boundaries."""
    import string
    import biotite.structure as struc
    out = atoms.copy(); mapping = {}; number = 0
    chains = list(dict.fromkeys(map(str, atoms.chain_id)))
    codes = string.ascii_uppercase + string.ascii_lowercase + string.digits
    if len(chains) > len(codes) or len(atoms) > 99999: raise ValueError("PDB format capacity exceeded")
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    segment = -1; previous = None
    for s, e in zip(starts[:-1], starts[1:]):
        number += 1
        if number > 9999: raise ValueError("PDB residue capacity exceeded")
        residue = atoms[s:e]
        new_segment = previous is None or str(previous.chain_id[0]) != str(residue.chain_id[0])
        if not new_segment:
            carbon = previous.coord[previous.atom_name == 'C']; nitrogen = residue.coord[residue.atom_name == 'N']
            new_segment = len(carbon)!=1 or len(nitrogen)!=1 or float(np.linalg.norm(carbon[0]-nitrogen[0]))>2.0
        if new_segment: segment += 1
        if segment >= len(codes): raise ValueError('PDB chain-segment capacity exceeded')
        chain = codes[segment]; previous = residue
        mapping[(chain, number)] = (str(atoms.chain_id[s]), int(atoms.res_id[s]), str(atoms.ins_code[s]).strip())
        out.chain_id[s:e] = chain; out.res_id[s:e] = number; out.ins_code[s:e] = ""
    file = PDBFile(); file.set_structure(out); file.write(path)
    check = PDBFile.read(path).get_structure(model=1)
    if len(check) != len(out) or not np.array_equal(check.atom_name, out.atom_name) or not np.allclose(check.coord, out.coord, atol=0.00051, rtol=0):
        raise ValueError("PDB export altered heavy atoms")
    return mapping


def complete(atoms, executable, work):
    """Use only newly rebuilt heavy atoms; reject renamed/moved observed atoms."""
    inp = work / "completion-input.pdb"; out = work / "completion-output.pdb"
    mapping = export_pdb(atoms, inp)
    result = subprocess.run([executable, "--ff=PARSE", "--keep-chain", "--nodebump", "--noopt",
        "--pdb-output", str(out), str(inp), str(work / "completion.pqr")], capture_output=True, text=True, timeout=300)
    (work / "completion.log").write_text(result.stdout + result.stderr)
    if result.returncode or not out.exists(): raise ValueError("PDB2PQR completion failed")
    fixed = PDBFile.read(out).get_structure(model=1)
    fixed = fixed[~np.isin(np.char.upper(fixed.element), ["H", "D"])]
    for i in range(len(fixed)):
        chain, num, ins = mapping[(str(fixed.chain_id[i]), int(fixed.res_id[i]))]
        fixed.chain_id[i] = chain; fixed.res_id[i] = num; fixed.ins_code[i] = ins
    def atomkey(a): return (str(a.chain_id), int(a.res_id), str(a.ins_code).strip(), str(a.atom_name))
    observed = {atomkey(a): a for a in atoms}; rebuilt = {atomkey(a): a for a in fixed}
    if len(rebuilt) != len(fixed) or not observed.keys() <= rebuilt.keys(): raise ValueError("completion removed/renamed atoms")
    for key, old in observed.items():
        new = rebuilt[key]
        if old.res_name != new.res_name or not np.allclose(old.coord, new.coord, atol=0.0011, rtol=0):
            raise ValueError("completion changed residue identity/observed coordinates")
    # Restore unrounded original coordinates after the PDB interchange.
    for i, atom in enumerate(fixed):
        if atomkey(atom) in observed: fixed.coord[i] = observed[atomkey(atom)].coord
    if set(fixed.res_name) - CANONICAL: raise ValueError("completion introduced noncanonical residues")
    return fixed


def missing_atoms(t):
    missing = {}
    for i, k in enumerate(t.keys):
        residue = t.residue(i); template, _, _ = _template(str(residue.res_name[0]))
        absent = set(template) - {"OXT"} - set(map(str, residue.atom_name))
        if absent: missing[i] = sorted(absent)
    return missing


def clean_components(atoms, cif):
    """Reject forbidden components before partner-only geometry is extracted."""
    import biotite.structure as struc
    keep = np.ones(len(atoms), bool)
    categories = cif.block.get("chem_comp")
    types = {} if categories is None else dict(zip(categories["id"].as_array(str), categories["type"].as_array(str)))
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    for s, e in zip(starts[:-1], starts[1:]):
        name = str(atoms.res_name[s]); elements = set(np.char.upper(atoms.element[s:e])) - {"H", "D"}
        if name in {"HOH", "WAT", "H2O", "DOD"} or (e-s == 1 and name in {"NA", "K", "CL"} and elements <= {"NA", "K", "CL"}):
            keep[s:e] = False; continue
        if name in CANONICAL: continue
        kind = types.get(name, "").upper()
        if elements - NONMETALS: code = "metal"
        elif "SACCHARIDE" in kind: code = "glycan"
        elif "PEPTIDE" in kind: code = "nonstandard_residue"
        else: code = "ligand"
        raise Rejection(code, f"component {name}, CCD type {kind}")
    return atoms[keep & ~np.isin(np.char.upper(atoms.element), ["H", "D"])]


def validate_partner_selection(atoms, cif, row):
    """FoldBench v1 requires a binary protein assembly, not an extracted subcomplex."""
    selected = row["partner_A_chains"] + row["partner_B_chains"]
    if len(selected) != 2 or len(set(selected)) != 2:
        raise Rejection("multi_partner", "FoldBench requires one distinct chain per partner", "selection")
    protein_chains = set(map(str, atoms.chain_id[np.isin(atoms.res_name, list(CANONICAL))]))
    poly = cif.block.get("entity_poly"); asym = cif.block.get("struct_asym")
    if poly is not None and asym is not None:
        entities = {str(e) for e, t in zip(poly["entity_id"].as_array(str), poly["type"].as_array(str)) if "polypeptide" in str(t)}
        protein_chains = {str(c) for c, e in zip(asym["id"].as_array(str), asym["entity_id"].as_array(str)) if str(e) in entities} & set(atoms.chain_id)
    if protein_chains != set(selected):
        raise Rejection("multi_partner", json.dumps({"selected": selected, "protein_chains": sorted(protein_chains)}), "selection")


def prepare_pair(cif, row, out, completion_executable=None, *, audit_allow_subcomplex=False, allow_distal_unlabelled=False, pdb2pqr_acceptance=False, enforce_geometry=True):
    atoms = pdbx.get_structure(cif, model=1, altloc="occupancy", use_author_fields=False, include_bonds=True)
    author_atoms = pdbx.get_structure(cif, model=1, altloc="occupancy", use_author_fields=True, include_bonds=True)
    if len(atoms)!=len(author_atoms) or not np.array_equal(atoms.coord,author_atoms.coord):
        raise Rejection("ambiguous_residue_key","author/label atom mapping disagrees")
    label_positions={}
    for atom, author in zip(atoms,author_atoms):
        label_positions[(str(atom.chain_id),int(author.res_id),str(author.ins_code).strip())]=int(atom.res_id)
    # Assembly copies keep label_asym IDs; residue numbering/insertion uses author identity.
    atoms.res_id=author_atoms.res_id.copy(); atoms.ins_code=author_atoms.ins_code.copy()
    # FoldBench names label_asym IDs. Preserve author mappings separately.
    category = cif.block["atom_site"]
    provenance = {"source_sha256": row["source_sha256"], "altloc": "highest occupancy; first tie",
        "author_mapping": {name: category[name].as_array(str).tolist() for name in
            ("label_asym_id", "auth_asym_id", "label_seq_id", "auth_seq_id", "pdbx_PDB_ins_code") if name in category}}
    provenance["altloc_source"]={name:category[name].as_array(str).tolist() for name in ("id","label_alt_id","occupancy") if name in category}
    if not audit_allow_subcomplex:
        validate_partner_selection(atoms, cif, row)
    # Chemical exclusions remain assembly-wide; do not silently discard bound components.
    atoms = clean_components(atoms, cif)
    partners = {"A": row["partner_A_chains"], "B": row["partner_B_chains"]}
    selected = set(partners["A"] + partners["B"])
    if set(partners["A"]) & set(partners["B"]) or not selected <= set(atoms.chain_id):
        raise Rejection("multi_partner", "invalid partner chain selection", "selection")
    atoms = atoms[np.isin(atoms.chain_id, list(selected))]
    # Match precision of PDB-only methods without moving any observed atom afterward.
    atoms.coord = np.round(atoms.coord, 3)
    precompleted=set()
    if pdb2pqr_acceptance:
        import biotite.structure as struc
        starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
        for s,e in zip(starts[:-1],starts[1:]):
            residue=atoms[s:e]; template,_,_=_template(str(residue.res_name[0]))
            if set(template)-{'OXT'}-set(map(str,residue.atom_name)):
                precompleted.add((str(residue.chain_id[0]),int(residue.res_id[0]),str(residue.ins_code[0]).strip()))
        if not completion_executable: raise ValueError('PDB2PQR acceptance requires its executable')
        try:
            atoms=complete(atoms,completion_executable,out)
        except (ValueError,KeyError,subprocess.TimeoutExpired) as exc:
            raise Rejection('pdb2pqr_failed',str(exc)) from exc
        provenance['pdb2pqr_acceptance']={'status':'passed','representation':'repaired observed residues; no missing-segment modelling; explicit physical segments',
            'observed_heavy_atoms_preserved':True}
    t = topology(atoms)
    label_by_key={str(k):label_positions[(k.chain,k.number,k.insertion)] for k in t.keys}
    for gap in t.metadata["gaps"]:
        n=label_by_key[gap["before"]]-label_by_key[gap["after"]]-1
        gap["missing_residues"]=max(0,n) if label_by_key[gap["after"]]>0 and label_by_key[gap["before"]]>0 else None
    if len(t.keys) > 1500: raise Rejection("size_cap", str(len(t.keys)), "selection")
    observed, _, _ = annotate(t, partners)
    gap_keys = {k for g in t.metadata["gaps"] for k in (g["after"], g["before"])}
    if not pdb2pqr_acceptance and any(str(k) in gap_keys and observed[i]["min_partner_distance"] <= 10 for i,k in enumerate(t.keys)):
        raise Rejection("interface_gap", json.dumps(t.metadata["gaps"]))
    chain_partner = {c: p for p, chains in partners.items() for c in chains}
    key_partner = {str(k): chain_partner[k.chain] for k in t.keys}
    if any(key_partner[a] != key_partner[b] for a,b in t.metadata["disulfide_pairs"]):
        raise Rejection("interpartner_disulfide", "covalent bond crosses partner boundary")
    missing = missing_atoms(t); completed = {str(k) for k in t.keys if (k.chain,k.number,k.insertion) in precompleted}; fallback = False
    if not pdb2pqr_acceptance and any(observed[i]["min_partner_distance"] <= 10 for i in missing):
        raise Rejection("interface_missing_sidechain", json.dumps({str(t.keys[i]): v for i,v in missing.items()}))
    if missing and not pdb2pqr_acceptance:
        original_atoms = atoms.copy(); original_topology = t
        if completion_executable:
            try:
                atoms = complete(atoms, completion_executable, out)
                completed = {str(t.keys[i]) for i in missing}
                t = topology(atoms)
                if missing_atoms(t): raise ValueError("completion left missing heavy atoms")
            except (ValueError, KeyError, subprocess.TimeoutExpired) as exc:
                provenance["completion_failure"] = str(exc); fallback = True
                atoms = original_atoms; t = original_topology
        else: fallback = True
        if fallback:
            completed = set()
            for i in missing:
                if str(t.residue(i).res_name[0]) in GROUPS and not allow_distal_unlabelled:
                    raise Rejection("missing_titratable_sidechain", f"G5 fallback: {t.keys[i]}")
    residues, delta, summary = annotate(t, partners)
    if enforce_geometry and summary["half_sum_buried_area"] < 500: raise Rejection("buried_area", json.dumps(summary), "selection")
    if enforce_geometry and summary["interface_residues"] < 10: raise Rejection("interface_residues", json.dumps(summary), "selection")
    sites = []
    for i, r in enumerate(residues):
        residue = t.residue(i); name = str(residue.res_name[0]); groups = []
        if name in GROUPS: groups.append(name)
        if t.nterm[i] or (str(t.keys[i]) in gap_keys and t.previous[i] < 0): groups.append("NTERM")
        if t.cterm[i] or (str(t.keys[i]) in gap_keys and t.following[i] < 0): groups.append("CTERM")
        s,e = t.starts[i:i+2]
        for group in groups:
            required = SITE_ATOMS[group]; is_complete = set(required) <= set(residue.atom_name)
            sites.append({"complex_id": row["complex_id"], **r, "group": group, "restype": name,
                "functional_delta_sasa": float(delta[s:e][np.isin(residue.atom_name, required)].sum()) if is_complete else None,
                "functional_atoms_complete": is_complete, "in_interface_zone": r["min_partner_distance"] <= 10,
                "is_break_terminus": group == "NTERM" and str(t.keys[i]) in gap_keys and not t.nterm[i] or group == "CTERM" and str(t.keys[i]) in gap_keys and not t.cterm[i],
                "was_completed": str(t.keys[i]) in completed})
    for state, chains in {"AB": list(selected), **partners}.items():
        write_cif(out / f"{state}.cif", t.atoms[np.isin(t.atoms.chain_id, chains)])
    provenance.update(topology=t.metadata, annotation=summary, geometry_gate_enforced=enforce_geometry, completion_fallback=fallback,
        completed=sorted(completed), coordinate_precision_angstrom=0.001)
    hashes = {state: digest(out/f"{state}.cif") for state in ("AB", "A", "B")}
    atomic_json(out/"provenance.json", {"partners": partners, "states": hashes, **provenance})
    sequences = {p: [str(t.residue(i).res_name[0]) for i,k in enumerate(t.keys) if k.chain in chains] for p,chains in partners.items()}
    structure = {"complex_id": row["complex_id"], "pdb_id": row["pdb_id"], "assembly": "1",
        "partner_A_chains": partners["A"], "partner_B_chains": partners["B"], "n_residues": len(t.keys),
        "homomeric": sequences["A"] == sequences["B"], "antibody": None,
        "provenance": json.dumps(provenance), "content_sha256": config_hash(hashes), "split": None, "component_id": None}
    return structure, sites


def prepare_campaign(manifest, archive, out, count=50, seed=20261003, completion_executable=None):
    require_compute(); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/"manifest.json").exists(): raise ValueError("campaign manifest already exists; choose a new run directory")
    candidates = []
    for index, row in enumerate(csv.DictReader(Path(manifest).open())):
        if row["interface_chain_type_1"] != "protein" or row["interface_chain_type_2"] != "protein": continue
        identity = (row["pdb_id"], *sorted((row["interface_chain_id_1"], row["interface_chain_id_2"])))
        if any(c["identity"] == identity for c in candidates): continue
        candidates.append({"identity": identity, "manifest_index": index, "pdb_id": row["pdb_id"],
            "partner_A_chains": [row["interface_chain_id_1"]], "partner_B_chains": [row["interface_chain_id_2"]],
            "complex_id": config_hash(identity)[:16]})
    if len(candidates) < count: raise ValueError("not enough unique candidate pairs")
    rng = np.random.default_rng(seed)
    chosen = [candidates[i] for i in sorted(rng.choice(len(candidates), count, replace=False))]
    frozen = {"seed": seed, "archive_sha256": digest(archive), "manifest_sha256": digest(manifest), "candidates": chosen}
    frozen["prep_config"]={"size_cap":1500,"interface_zone":10,"interface_delta_sasa":10,"min_buried_area":500,
        "min_interface_residues":10,"completion_executable":completion_executable,"fallback":"titratable_only",
        "component_scope":"whole_assembly_before_pair_selection","coordinate_precision":.001}
    frozen["prep_config"]["partner_scope"] = "exactly_two_protein_chains_in_source"
    frozen["prep_implementation"]={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name("annotate.py"))}
    atomic_json(out/"manifest.json", frozen)
    structures = []; sites = []; rejections = []; failures = []
    with tarfile.open(archive) as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        for row in chosen:
            work = out/"structures"/row["complex_id"]; work.mkdir(parents=True, exist_ok=True)
            try:
                raw = tar.extractfile(members[row["pdb_id"]+".cif"]).read()
                source = work/"source.cif"; source.write_bytes(raw); row["source_sha256"] = digest(source)
                structure, annotated = prepare_pair(pdbx.CIFFile.read(io.StringIO(raw.decode())), row, work, completion_executable)
                structures.append(structure); sites.extend(annotated)
                write_table(work/"sites.parquet", "sites", annotated)
            except Rejection as exc:
                rejections.append({"candidate_id": row["complex_id"], "stage": exc.stage, "code": exc.code, "detail": str(exc)})
            except Exception as exc:
                # Programming/I/O errors must not masquerade as scientific rejections.
                import traceback
                failures.append({"candidate_id": row["complex_id"], "error": str(exc), "traceback": traceback.format_exc()})
    for name, rows in (("structures", structures), ("sites", sites), ("rejections", rejections)):
        write_table(out/f"{name}.parquet", name, rows)
    atomic_json(out/"curation_report.json", {"candidates": count, "accepted": len(structures),
        "acceptance": len(structures)/count, "rejected": len(rejections), "pipeline_errors": failures,
        "progression_allowed": len(structures)/count >= .5 and not failures})
    if failures: raise RuntimeError("preparation pipeline errors; see curation_report.json")
