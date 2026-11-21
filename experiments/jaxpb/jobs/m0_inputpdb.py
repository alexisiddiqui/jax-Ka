"""Regenerate input.pdb for one campaign state exactly as pkabench adapters/base.py did (export_pdb(read_cif(cif))),
copied from jaxpb-pilot-v1 prep.py (itself from ~/jax-Ka/src/pkabench/prep.py @95091c0). Gate: mapping equals the saved
mapping.json. Runner env (biotite). usage: python m0_inputpdb.py <campaign_state_dir> <work_dir>"""
import json, sys, string
from pathlib import Path
import numpy as np
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile
import biotite.structure as struc


def read_cif(path):
    return pdbx.get_structure(pdbx.CIFFile.read(path), model=1, altloc="occupancy", use_author_fields=True, include_bonds=True)


def export_pdb(atoms, path):
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
            new_segment = len(carbon) != 1 or len(nitrogen) != 1 or float(np.linalg.norm(carbon[0] - nitrogen[0])) > 2.0
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


src = Path(sys.argv[1]); work = Path(sys.argv[2]); work.mkdir(parents=True, exist_ok=True)
req = json.loads((src / "request.json").read_text())
mapping = export_pdb(read_cif(req["cif"]), work / "input.pdb")
mine = [{"chain": c, "resnum": n, "original": list(v)} for (c, n), v in mapping.items()]
ok = mine == json.loads((src / "mapping.json").read_text())
print("mapping_equal", ok)
if not ok:
    raise SystemExit("mapping mismatch")
