"""Native teachers with published antibody scope and existing defect masks.

Deposited sequences define true termini; published IMGT intervals define the
antibody unit. Only missing positions inside that unit are sequence defects.
"""
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

from .runtime import atomic_json,digest

GROUP_ATOMS = {'ASP': ('OD1','OD2'), 'GLU': ('OE1','OE2'), 'HIS': ('ND1','NE2'),
    'CYS': ('SG',), 'TYR': ('OH',), 'LYS': ('NZ',), 'ARG': ('NE','NH1','NH2'),
    'NTERM': ('N',), 'CTERM': ('O','OXT')}
MAX_ASA = dict(ALA=129,ARG=274,ASN=195,ASP=193,CYS=167,GLN=225,GLU=223,
    GLY=104,HIS=224,ILE=197,LEU=201,LYS=236,MET=224,PHE=240,PRO=159,
    SER=155,THR=172,TRP=285,TYR=263,VAL=174)


def source_evidence(unit, topology, meta, *, antibody_crop=None, source_cif_bytes=None):
    from biotite.structure.io import pdbx
    from jaxpropka.parameters import THREE, ALPHABET
    from .conformers import resolve
    from .component_mask_policy import classify
    from .prep import missing_atoms
    raw = (gzip.decompress((Path(unit['entry'])/'source.cif.gz').read_bytes())
           if source_cif_bytes is None else source_cif_bytes)
    if not isinstance(raw,bytes):raise TypeError('Source CIF payload must be bytes')
    if hashlib.sha256(raw).hexdigest() != meta['source_cif_sha256']:
        raise ValueError('SAbDab source CIF changed')
    cif = pdbx.CIFFile.read(io.StringIO(raw.decode()))
    cif, conformers = resolve(cif, list(meta['side_by_chain']))
    poly = cif.block['entity_poly']; asym = cif.block['struct_asym']
    chain_entity = {}
    for chain, entity in zip(asym['id'].as_array(str), asym['entity_id'].as_array(str)):
        if str(chain) in chain_entity and chain_entity[str(chain)] != str(entity):
            raise ValueError('Ambiguous polymer entity')
        chain_entity[str(chain)] = str(entity)
    column = 'pdbx_seq_one_letter_code_can' if 'pdbx_seq_one_letter_code_can' in poly else 'pdbx_seq_one_letter_code'
    sequences = {str(e): ''.join(re.sub(r'\([^)]*\)', 'X', str(s)).split()).upper()
        for e,s in zip(poly['entity_id'].as_array(str), poly[column].as_array(str))}
    aa = dict(zip(THREE, ALPHABET)); missing = missing_atoms(topology)
    crop=antibody_crop if antibody_crop is not None else meta.get('antibody_crop_metadata')
    intervals={};scope=dict(status='metadata_missing_retained_full_sequence_masks',policy='deposited-full-sequence-v1')
    if crop is not None:
        from .sabdab_scope import retained_intervals
        intervals,scope=retained_intervals(crop,meta['residue_map'],
            [aa[str(topology.residue(i).res_name[0])] for i in range(topology.n_residues)],
            meta['side_by_chain'],meta['instance'])
    evidence = dict(atoms=[], sequences=[], defects=[], artificial_terminal_keys=[],
        source_scope='published antibody unit intervals; deposited antigen context and true-terminus evidence retained',
        antibody_scope=scope,intentional_omissions=[],
        conformer_selection=conformers['selection'])
    real = set(); known = set(); residue_lookup = {}
    for i, (key, mapping) in enumerate(zip(topology.keys, meta['residue_map'])):
        k = (key.chain, key.number, key.insertion); residue_lookup[k] = topology.residue(i)
        evidence['atoms'].append(dict(key=list(k), label_seq_id=mapping['source_label_position'],
            missing_atoms=missing.get(i, [])))
        if i in missing:
            evidence['defects'].append(dict(kind='missing_atoms', key=list(k),
                centres=[topology.backbone[i,1].astype(float).tolist()], extent=8.))
    for chain in topology.chain_ids:
        sequence = sequences[chain_entity[chain]]
        positions = {}
        for i, (key, mapping) in enumerate(zip(topology.keys, meta['residue_map'])):
            if key.chain != chain:
                continue
            pos = mapping['source_label_position']
            if not 1 <= pos <= len(sequence) or pos in positions:
                raise ValueError('Invalid/duplicate source polymer position')
            if sequence[pos-1] not in (aa[str(topology.residue(i).res_name[0])], 'X'):
                raise ValueError('Source sequence/residue mismatch at '+chain+':'+str(pos))
            positions[pos] = i
        ranges=intervals.get(chain,[(1,len(sequence))])
        scope_positions={p for start,end in ranges for p in range(start,end+1)}
        if not positions.keys()<=scope_positions:raise ValueError('Observed residue outside declared antibody unit')
        evidence['sequences'].append(dict(chain=chain, sequence=sequence,
            sequence_source='entity_poly_canonical',observed_residues=len(positions),retained_source_intervals=ranges))
        for pos, group in [(min(positions), 'NTERM'), (max(positions), 'CTERM')]:
            key=topology.keys[positions[pos]]; k=(key.chain,key.number,key.insertion)
            known.add((*k,group))
            if (group=='NTERM' and pos==1 or group=='CTERM' and pos==len(sequence)) and sequence[pos-1]!='X':
                real.add((*k,group))
        if chain in intervals:
            from .sabdab_scope import omitted_runs
            for omitted in omitted_runs(scope_positions,len(sequence)):
                evidence['intentional_omissions'].append(dict(chain=chain,start=omitted[0],end=omitted[-1],
                    length=len(omitted),reason='outside_verified_published_antibody_intervals',
                    scope_record_sha256=crop['sha256']))
        runs=[]
        for pos in sorted(scope_positions-positions.keys()):
            if not runs or pos!=runs[-1][-1]+1:runs.append([pos])
            else:runs[-1].append(pos)
        for run in runs:
            flanks=[p for p in (run[0]-1,run[-1]+1) if p in positions]
            kind='internal_gap' if len(flanks)==2 else 'terminal_gap'
            evidence['defects'].append(dict(kind=kind, chain=chain, start=run[0],end=run[-1],length=len(run),extent=8.,
                centres=[topology.backbone[positions[p],1].astype(float).tolist() for p in flanks],
                scope_note='unresolved within declared unit' if chain in intervals else 'deposited antigen/full-chain gap'))
            for pos in flanks:
                key=topology.keys[positions[pos]]
                evidence['artificial_terminal_keys'].append([key.chain,key.number,key.insertion])
    source_atoms=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
    # SAbDab ML files omit chem_comp. Resolve its type from the installed PDB
    # CCD, pinning the dictionary bytes, before applying the SAME component
    # classifier/radii used for the other sources. Unknown chemistry stays held.
    existing=cif.block.get('chem_comp')
    types={} if existing is None else dict(zip(existing['id'].as_array(str),existing['type'].as_array(str)))
    candidates=set(map(str,source_atoms.res_name))-set(THREE)-{'HOH','WAT','H2O','DOD'}
    unresolved=candidates-types.keys();dictionary=None
    if unresolved:
        import biotite.structure.info.ccd as ccd
        dictionary=dict(path=str(ccd._CCD_FILE),sha256=digest(ccd._CCD_FILE),resolved_types={})
        for name in sorted(unresolved):
            kind=ccd.get_from_ccd('chem_comp',name,'type')
            if kind is not None:
                types[name]=str(kind.as_array(str)[0]);dictionary['resolved_types'][name]=types[name]
    if types:
        cif.block['chem_comp']=pdbx.CIFCategory(dict(id=list(types),type=list(types.values())))
    components,unknown=classify(source_atoms,cif,list(meta['side_by_chain']))
    # Canonical free amino acids pass clean_components(), but are excluded by
    # the protein-entity selector. They still belong in omitted-context masks.
    cat=cif.block['atom_site'];poly_entities=set(poly['entity_id'].as_array(str))
    selected=np.isin(source_atoms.chain_id,list(meta['side_by_chain'])) & np.isin(cat['label_entity_id'].as_array(str),list(poly_entities))
    omitted_canonical=~selected & np.isin(source_atoms.res_name,list(THREE)) & ~np.isin(np.char.upper(source_atoms.element),['H','D'])
    if omitted_canonical.any():
        components.append(dict(cls='untyped_nonprotein',name='omitted_canonical_context',
            coordinates=source_atoms.coord[omitted_canonical].astype(float)))
    # ML CIFs can omit CCD type metadata. An untyped component gets the largest
    # existing component radius, rather than guessing it is a harmless ligand.
    for component in components:
        if component['cls'] not in {'ligand','buffer','glycan','exposed_metal','bound_metal'} or (
                component.get('name') not in types and component['cls']=='ligand'):
            component['cls']='untyped_nonprotein'
    evidence['components']=[dict(c,coordinates=c['coordinates'].tolist()) for c in components]
    evidence['component_dictionary']=dictionary
    evidence['components_without_heavy_atoms']=unknown
    return evidence,real,known,residue_lookup


def export_states(topology, meta, real, directory):
    from .prep import export_pdb
    from .pkai_termini import add_termini
    mappings={}
    for state in ['AB','A','B']:
        chains=[c for c,s in meta['side_by_chain'].items() if state=='AB' or s==state]
        path=directory/(state+'.pdb')
        mapping=export_pdb(topology.atoms[np.isin(topology.atoms.chain_id,chains)],path)
        reverse={v:k for k,v in mapping.items()}
        nterm=[reverse[k[:3]] for k in real if k[3]=='NTERM' and k[:3] in reverse]
        cterm=[reverse[k[:3]] for k in real if k[3]=='CTERM' and k[:3] in reverse]
        text,assigned=add_termini(path.read_text(),nterm,cterm);path.write_text(text)
        mappings[state]=dict(residues=[[c,n,*key] for (c,n),key in mapping.items()],
            termini=[[c,n,kind,oc,on] for (c,n),(kind,oc,on) in assigned.items()])
    return mappings


def make_sites(topology, meta, evidence, real, known, residues):
    import biotite.structure as struc
    from scipy.spatial import cKDTree
    from .site_weights import burial_weight, interface_weight
    from .pkpdb_pilot_clean import gap_context
    from .long_gap_policy import site_usable
    from .supervision import clearance
    from .component_mask_policy import POLICY,class_trees,nearest_by_class,clear_of_components
    def rsa(atoms):
        sasa=struc.sasa(atoms,vdw_radii='ProtOr')
        if not np.isfinite(sasa).all():raise ValueError('Nonfinite native SASA')
        sums=struc.apply_residue_wise(atoms,sasa,np.sum);starts=struc.get_residue_starts(atoms)
        return {(str(atoms.chain_id[i]),int(atoms.res_id[i]),str(atoms.ins_code[i]).strip()):float(v)/MAX_ASA[str(atoms.res_name[i])]
                for i,v in zip(starts,sums)}
    atoms=topology.atoms;bound=rsa(atoms);free={};distance={}
    for side in ['A','B']:
        mask=np.asarray([meta['side_by_chain'][str(c)]==side for c in atoms.chain_id])
        free.update(rsa(atoms[mask]));dist=cKDTree(atoms.coord[~mask]).query(atoms.coord[mask])[0]
        for index,value in zip(np.flatnonzero(mask),dist):
            key=(str(atoms.chain_id[index]),int(atoms.res_id[index]),str(atoms.ins_code[index]).strip())
            distance[key]=min(distance.get(key,float('inf')),float(value))
    gaps,known_sequence=gap_context(evidence,residues);trees=class_trees(evidence['components']);sites=[]
    train_radii=dict(POLICY['train_radii_A'],untyped_nonprotein=30.)
    eval_radii=dict(POLICY['eval_radii_A'],untyped_nonprotein=30.)
    incomplete={tuple(r['key']) for r in evidence['atoms'] if r['missing_atoms']}
    for i,key in enumerate(topology.keys):
        k=(key.chain,key.number,key.insertion);residue=residues[k];name=str(residue.res_name[0])
        groups=([name] if name in GROUP_ATOMS else [])+[g for g in ['NTERM','CTERM'] if (*k,g) in known]
        for group in groups:
            complete=set(GROUP_ATOMS[group])<=set(map(str,residue.atom_name))
            true_terminal=(*k,group) in real
            eligible=bool(known_sequence and complete and k not in incomplete and
                (group not in ('NTERM','CTERM') or true_terminal) and
                not (group=='CYS' and topology.disulfide[i]))
            points=residue.coord[np.isin(residue.atom_name,GROUP_ATOMS[group])]
            usable,reason=site_usable(points,eligible,gaps) if len(points) else (False,'no_functional_atoms')
            if usable and any(clearance(points,d)<15. for d in evidence['defects'] if d['kind']=='missing_atoms'):
                usable=False;reason='near_unrepaired_native_atoms'
            distances=nearest_by_class(points,trees)
            train=usable and not evidence['components_without_heavy_atoms'] and clear_of_components(distances,train_radii)
            evaluation=usable and not evidence['components_without_heavy_atoms'] and clear_of_components(distances,eval_radii)
            sites.append(dict(chain=key.chain,resnum=key.number,icode=key.insertion,group=group,restype=name,
                partner=meta['side_by_chain'][key.chain],side=0 if meta['side_by_chain'][key.chain]=='A' else 1,
                interface=distance[k]<=10.,eligible=eligible,functional_atoms_complete=complete,
                true_terminus=true_terminal if group in ('NTERM','CTERM') else None,
                is_break_terminus=bool(group in ('NTERM','CTERM') and not true_terminal),
                rsa_free=round(free[k],4),rsa_bound=round(bound[k],4),partner_distance_A=round(distance[k],2),
                w_burial=round(float(burial_weight(free[k])),4),w_interface=round(float(interface_weight(distance[k])),4),
                train_mask=bool(train),eval_mask=bool(evaluation),gap_rule=reason,
                nearest_component_by_class=distances,mask_status='curated_declared_unit_scope',
                supervision_scope_policy=evidence['antibody_scope']['policy'],
                mask_policy='long-gap-v1 + mask-all-v1 within declared antibody scope; deposited antigen gaps; untyped components 30 A'))
    return sites


def qualify_labels(sites, labels):
    if labels['errors']:raise ValueError('Teacher reported prediction errors: '+repr(labels['errors']))
    mapped={state:{tuple(r[:4]):r[4] for r in rows} for state,rows in labels['pkai'].items()}
    counts=Counter()
    for site in sites:
        key=(site['chain'],site['resnum'],site['icode'],site['group'])
        paired=key in mapped['AB'] and key in mapped[site['partner']]
        # The existing site graph deliberately excludes ARG target queries.
        graph_supported=site['group']!='ARG'
        site.update(paired_teacher_available=paired,graph_query_supported=graph_supported)
        for mask in ['train_mask','eval_mask']:
            site[mask]=bool(site[mask] and paired and graph_supported)
            counts[mask]+=site[mask];counts[mask+'_interface']+=site[mask] and site['interface']
        counts['paired_sites']+=paired
    counts['sites']=len(sites)
    return dict(counts)
