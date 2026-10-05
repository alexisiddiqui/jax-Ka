"""Experimental pilot v2: source receipts and isolated-chain structural preflight."""
import json
import os
from pathlib import Path
import urllib.request
from collections import Counter

from pkabench.runtime import require_compute, atomic_json, digest

require_compute()
runtime = Path(os.environ['PKABENCH_RUNTIME'])
source = runtime / 'experimental/pilot-v1'
out = runtime / 'experimental/pilot-v2'
out.mkdir(exist_ok=True)
(out / 'sources').mkdir(exist_ok=True)
receipts = []
for pmid in ('7626612', '9132009', '2065058', '3173493', '35837736'):
    url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}%20AND%20SRC:MED&format=json&resultType=core'
    path = out / 'sources' / f'{pmid}.json'
    try:
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(url, timeout=45).read())
        data = json.loads(path.read_text())['resultList']['result']
        receipts.append(dict(pmid=pmid, url=url, sha256=digest(path), metadata=data))
    except Exception as exc:
        receipts.append(dict(pmid=pmid, url=url, error=repr(exc)))
atomic_json(out / 'primary_metadata.json', receipts)

import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdbx
from pkabench.conformers import resolve
from pkabench.supervision import inventory
from pkabench.prep import CANONICAL, complete, topology, write_cif, export_pdb
from pkabench.glycan_buffer_policy import classify_components, masks
from pkabench.annotate import SITE_ATOMS
from pkabench.schema import write_table

labels = json.loads((source / 'lead_residue_candidates.json').read_text())
structures = []
for pdb in sorted({r['pdb_id'] for r in labels} | {'1BNI'}):
    label_pdb = '1A2P' if pdb == '1BNI' else pdb
    root = out / 'structures' / pdb
    root.mkdir(parents=True, exist_ok=True)
    original_path = source / 'sources' / f'{pdb}-cif.raw'
    try:
        if not original_path.exists():
            original_path = out / 'sources' / f'{pdb}-cif.raw'
            if not original_path.exists():
                original_path.write_bytes(urllib.request.urlopen(f'https://files.rcsb.org/download/{pdb}.cif', timeout=45).read())
        cif = pdbx.CIFFile.read(original_path)
        cat = cif.block['atom_site']
        author_chain = next(r['chain'] for r in labels if r['pdb_id'] == label_pdb)
        hit = (cat['auth_asym_id'].as_array(str) == author_chain) & (cat['label_seq_id'].as_array(str) != '.') & (cat['label_seq_id'].as_array(str) != '?')
        chains = sorted(set(cat['label_asym_id'].as_array(str)[hit]))
        assert len(chains) == 1, chains
        chain = chains[0]
        cif, conformers = resolve(cif, chains)
        cif.write(root / 'resolved-source.cif')
        atomic_json(root / 'conformers.json', conformers)
        partners = {'A': chains, 'B': []}
        evidence = inventory(cif, partners)
        if pdb != label_pdb:
            original_cif = pdbx.CIFFile.read(source / 'sources' / f'{label_pdb}-cif.raw')
            old_sequences = original_cif.block['entity_poly']['pdbx_seq_one_letter_code_can'].as_array(str)
            assert evidence['sequences'][0]['sequence'] in {''.join(s.split()) for s in old_sequences}, 'Alternative sequence differs'
        atomic_json(root / 'input_atom_mask.json', evidence)
        atoms = pdbx.get_structure(cif, model=1, altloc='occupancy', use_author_fields=False)
        authors = pdbx.get_structure(cif, model=1, altloc='occupancy', use_author_fields=True)
        # Component chemistry is checked in the full deposited context. Empty
        # partner B is only an annotation placeholder for this monomer preflight.
        removals, components = classify_components(atoms, cif, partners)
        for c in removals:
            if c['policy_class'] == 'buffer':
                c.update(train_radius_A=15., eval_radius_A=20.)
        atomic_json(root / 'removed_components.json', removals)
        omitted = sorted(set(atoms.chain_id[np.isin(atoms.res_name, list(CANONICAL))]) - set(chains))
        atoms.res_id = authors.res_id.copy()
        atoms.ins_code = authors.ins_code.copy()
        atoms = atoms[np.isin(atoms.chain_id, chains) & np.isin(atoms.res_name, list(CANONICAL)) & ~np.isin(np.char.upper(atoms.element), ['H','D'])]
        write_cif(root / 'observed.cif', atoms)
        fixed = complete(atoms, str(runtime / 'envs/pypka/bin/pdb2pqr30'), root)
        t = topology(fixed)
        write_cif(root / 'AB.cif', fixed)
        mapping = export_pdb(fixed, root / 'input.pdb')
        atomic_json(root / 'pdb_mapping.json', [{'chain':c,'resnum':n,'original':v} for (c,n),v in mapping.items()])
        atom_inventory = {tuple(r['key']): r for r in evidence['atoms']}
        allsites = []
        for i, key in enumerate(t.keys):
            residue = t.residue(i)
            name = str(residue.res_name[0])
            groups = [name] if name in SITE_ATOMS else []
            if i == 0: groups.append('NTERM')
            if i == len(t.keys)-1: groups.append('CTERM')
            for group in groups:
                allsites.append(dict(complex_id=pdb, chain=key.chain, resnum=key.number,
                    icode=key.insertion, group=group, restype=name, partner='A',
                    functional_atoms_complete=set(SITE_ATOMS[group]) <= set(residue.atom_name),
                    is_break_terminus=bool(group in ('NTERM','CTERM') and [key.chain,key.number,key.insertion] in evidence['artificial_terminal_keys']), residue_delta_sasa=0.0,
                    was_completed=bool(atom_inventory[(key.chain,key.number,key.insertion)]['missing_atoms'])))
        write_table(root / 'sites.parquet', 'sites', allsites)
        component_masks = masks(allsites, fixed, removals)
        atomic_json(root / 'component_masks.json', component_masks)
        structures.append(dict(pdb_id=pdb, label_pdb_id=label_pdb, status='prepared', chain=chain, author_chain=author_chain,
            n_residues=len(t.keys), omitted_deposited_protein_chains=omitted,
            source_sha256=digest(original_path), prepared_sha256=digest(root / 'AB.cif'),
            topology_gaps=t.metadata['gaps'], sequences=evidence['sequences'],
            scope='isolated deposited chain; experimental solution construct must still be verified',
            component_count=len(removals), sites=len(allsites)))
    except Exception as exc:
        import traceback
        (root / 'error.txt').write_text(traceback.format_exc())
        structures.append(dict(pdb_id=pdb, status='failed', error=repr(exc)))
atomic_json(out / 'structure_preflight.json', structures)
print(json.dumps(structures, indent=2), flush=True)

# Apply the existing natural-gap overlay, which consumes geometry/inventory and
# does not depend on having two partners. Interface fields are deliberately zero.
from pkabench.anchor_tiers import apply
write_table(out / 'structures.parquet', 'structures', [{'complex_id':r['pdb_id'], 'pdb_id':r['pdb_id']} for r in structures if r['status']=='prepared'])
if not (out / 'natural-gap-tiers-final').exists():
    apply(out, out / 'natural-gap-tiers-final')
atomic_json(out / 'preparation_receipt.json', dict(job=os.environ['SLURM_JOB_ID'],
    code_sha256=digest(Path(__file__)), source_labels_sha256=digest(source / 'lead_residue_candidates.json'),
    policy='Shared completion, conformer, component and natural-gap code; isolated-chain adapter; no fitted parameters',
    implementation={name:digest(Path(os.environ['PKABENCH_SOURCE'])/'src/pkabench'/name) for name in
        ('prep.py','conformers.py','supervision.py','glycan_buffer_policy.py','anchor_tiers.py')}))
