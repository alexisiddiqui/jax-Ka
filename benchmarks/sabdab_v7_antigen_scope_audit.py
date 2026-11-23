"""Compare retained SAbDab antigens with actual deposited polymer coordinates."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import re

import numpy as np
from biotite.structure.io import pdbx

from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.sabdab_units import load_prepared
from pkabench.conformers import resolve
from jaxpropka.parameters import THREE,ALPHABET


def read(path):return json.loads(Path(path).read_text())


def runs(values):
    result=[]
    for value in sorted(values):
        if not result or value!=result[-1][-1]+1:result.append([value])
        else:result[-1].append(value)
    return [dict(start=r[0],end=r[-1],length=len(r)) for r in result]


def inspect(case,unit,row):
    source=Path(case['deposited'])
    with gzip.open(source,'rt') as stream:file=pdbx.CIFFile.read(stream)
    block=file.block;cat=block['atom_site'];n=cat.row_count
    def field(name,default=''):
        return cat[name].as_array(str) if name in cat else np.full(n,default)
    poly=block['entity_poly'];protein={str(e) for e,t in zip(poly['entity_id'].as_array(str),poly['type'].as_array(str)) if str(t)=='polypeptide(L)'}
    mask=(field('auth_asym_id')==case['author_antigen_chain']) & np.isin(field('label_entity_id'),list(protein))
    chains=set(field('label_asym_id')[mask])
    if len(chains)!=1:raise ValueError('Antigen author ID is absent or ambiguous in original deposition')
    chain=next(iter(chains));entities=set(field('label_entity_id')[mask])
    if len(entities)!=1:raise ValueError('Ambiguous antigen polymer entity')
    entity=next(iter(entities));column='pdbx_seq_one_letter_code_can' if 'pdbx_seq_one_letter_code_can' in poly else 'pdbx_seq_one_letter_code'
    seqs={str(e):''.join(re.sub(r'\([^)]*\)','X',str(s)).split()).upper() for e,s in zip(poly['entity_id'].as_array(str),poly[column].as_array(str))}
    sequence=seqs[entity]
    # Restrict before conformer resolution: unrelated author chains do not
    # participate in this targeted coordinate comparison.
    selected=pdbx.CIFCategory()
    for name in cat:selected[name]=pdbx.CIFColumn(cat[name].as_array(str)[mask])
    block['atom_site']=selected
    file,conformers=resolve(file,[chain]);cat=file.block['atom_site'];n=cat.row_count
    label=cat['label_seq_id'].as_array(int);authors=cat['auth_seq_id'].as_array(int)
    insertions=cat['pdbx_PDB_ins_code'].as_array(str) if 'pdbx_PDB_ins_code' in cat else np.full(n,'')
    insertions=np.asarray(['' if x in ('.','?',' ') else str(x).strip() for x in insertions])
    names=cat['label_atom_id'].as_array(str);resnames=cat['label_comp_id'].as_array(str)
    elements=cat['type_symbol'].as_array(str)
    xyz=np.column_stack([cat['Cartn_'+axis].as_array(float) for axis in 'xyz'])
    observed={};by_author={}
    for i in range(n):
        if str(elements[i]).upper() in ('H','D'):continue
        position=int(label[i]);key=(case['author_antigen_chain'],int(authors[i]),str(insertions[i]))
        if not 1<=position<=len(sequence):raise ValueError('Original polymer position is outside deposited sequence')
        residue=observed.setdefault(position,dict(author_key=key,resname=str(resnames[i]),atoms={}))
        if residue['author_key']!=key or residue['resname']!=resnames[i] or names[i] in residue['atoms']:
            raise ValueError('Ambiguous original deposited residue/atom identity')
        residue['atoms'][str(names[i])]=xyz[i]
        if key in by_author and by_author[key]!=position:raise ValueError('Original author residue maps to multiple polymer positions')
        by_author[key]=position
    topology,meta=load_prepared(unit);aa=dict(zip(THREE,ALPHABET))
    retained={};coordinate_differences=[];residue_checks=[]
    for i,mapping in enumerate(meta['residue_map']):
        if mapping['side']!='B':continue
        key=(mapping['source_author_chain'],mapping['source_author_number'],mapping['source_author_insertion'])
        if key[0]!=case['author_antigen_chain']:raise ValueError('Unexpected additional antigen in selected case')
        if key not in by_author:raise ValueError('ML retained antigen residue absent from original deposition')
        position=by_author[key];original=observed[position];a=topology.residue(i)
        if position!=mapping['source_label_position']:raise ValueError('ML/deposited polymer position mapping differs')
        if str(a.res_name[0])!=original['resname']:raise ValueError('ML/deposited retained residue chemistry differs')
        atoms={str(atom.atom_name):atom.coord.astype(float) for atom in a}
        if atoms.keys()!=original['atoms'].keys():raise ValueError('ML/deposited retained heavy-atom inventories differ')
        differences=[float(np.linalg.norm(atoms[name]-original['atoms'][name])) for name in atoms]
        coordinate_differences.extend(differences)
        residue_checks.append(dict(source_position=position,author_key=list(key),resname=original['resname'],
            heavy_atoms=len(atoms),max_unaligned_displacement_A=max(differences)))
        retained[position]=aa[str(a.res_name[0])]
    ag_chains=row['agchains'].split('/');index=ag_chains.index(case['author_antigen_chain'])
    published=row['ag_index_list'].split('/')[index]
    if published=='__FULL_CHAIN__':raise ValueError('This audit expects explicit published retained antigen indices')
    expected=json.loads(published)
    expected_keys=[(case['author_antigen_chain'],int(k[0]),str(k[1]).strip()) for _,k in expected]
    actual_keys=[observed[p]['author_key'] for p in sorted(retained)]
    if expected_keys!=actual_keys or ''.join(retained[p] for p in sorted(retained))!=row['agresolvedseqs'].split('/')[index]:
        raise ValueError('Published retained antigen keys/sequence differ from ML coordinates')
    if sequence!=row['agexpectedseqs'].split('/')[index]:raise ValueError('Published/deposited full antigen sequences differ')
    if max(coordinate_differences)>.002:raise ValueError('ML retained antigen coordinates differ from original reference frame')
    observed_positions=set(observed);retained_positions=set(retained);full=set(range(1,len(sequence)+1))
    omitted_observed=observed_positions-retained_positions;unobserved=full-observed_positions
    annotations=[];unobs=block.get('pdbx_unobs_or_zero_occ_residues')
    if unobs is not None:
        for i in range(unobs.row_count):
            a=str(unobs['auth_asym_id'].as_array(str)[i]) if 'auth_asym_id' in unobs else ''
            l=str(unobs['label_asym_id'].as_array(str)[i]) if 'label_asym_id' in unobs else ''
            if a==case['author_antigen_chain'] or l==chain:
                annotations.append({name:str(unobs[name].as_array(str)[i]) for name in unobs})
    annotated_positions={int(a['label_seq_id']) for a in annotations}
    if annotated_positions!=unobserved:
        raise ValueError('Deposited unobserved-residue annotation does not match missing coordinates')
    if omitted_observed and unobserved:classification='mixture_of_omitted_observed_and_unobserved_deposited_residues'
    elif omitted_observed:classification='ML_omits_residues_with_deposited_coordinates'
    else:classification='ML_retains_all_deposited_antigen_residues_with_coordinates'
    return dict(instance=case['instance'],pdb_id=case['pdb'],deposited_source=str(source),deposited_sha256=digest(source),
        author_antigen_chain=case['author_antigen_chain'],deposited_label_chain=chain,
        full_sequence_length=len(sequence),deposited_observed_residues=len(observed),ML_retained_residues=len(retained),
        retained_source_ranges=runs(retained_positions),omitted_observed_residues=len(omitted_observed),
        omitted_observed_ranges=runs(omitted_observed),unobserved_deposited_residues=len(unobserved),
        unobserved_deposited_ranges=runs(unobserved),deposition_unobserved_annotations=annotations,
        deposition_unobserved_annotation_coverage_verified=True,
        classification=classification,published_index_and_sequence_verified=True,
        retained_heavy_atoms_compared=len(coordinate_differences),
        max_unaligned_heavy_atom_displacement_A=max(coordinate_differences),residue_checks=residue_checks,
        original_observed_inventory=[dict(source_position=p,author_key=list(r['author_key']),resname=r['resname'],heavy_atoms=len(r['atoms'])) for p,r in sorted(observed.items())])


def run(args):
    require_compute(threads=1)
    if args.output.exists():raise FileExistsError(args.output)
    cases=read(args.plan)['cases'];units={u['unit_id'].split(':',1)[1]:u for u in read(args.inventory)['units']}
    rows={}
    with args.metadata.open() as handle:
        for row in csv.DictReader(handle):
            if row['INSTANCE'] in {c['instance'] for c in cases}:rows[row['INSTANCE']]=row
    results=[inspect(case,units[case['instance']],rows[case['instance']]) for case in cases]
    atomic_json(args.output,dict(status='deposited_antigen_scope_comparison_complete',cases=results,
        metadata_sha256=digest(args.metadata),inventory_sha256=digest(args.inventory),code_sha256=digest(__file__),
        supervision_or_release_changed=False,
        interpretation='Omitted observed residues demonstrate processing selection; residues without deposited coordinates retain unresolved-context uncertainty.'))
    print(json.dumps([{k:v for k,v in r.items() if k not in ('residue_checks','original_observed_inventory','deposition_unobserved_annotations')} for r in results],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','inventory','metadata','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
