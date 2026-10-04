"""Read-only pKPDB membership, construct-aware AFDB coverage evidence."""
import json
from pathlib import Path
import time
from collections import Counter
from .runtime import atomic_json, digest, require_compute
from .download import fetch


def map_positions(sequence,mapping,af_sequence):
    """Accept linear SIFTS segments only after complete sequence identity check."""
    start=mapping['start']['residue_number']; end=mapping['end']['residue_number']
    ustart=mapping['unp_start']; uend=mapping['unp_end']
    if end-start != uend-ustart: return None
    if sequence[start-1:end]!=af_sequence[ustart-1:uend] or end>len(sequence) or uend>len(af_sequence): return None
    return {p:ustart+p-start for p in range(start,end+1)}


def audit(campaign,out,count=12):
    from biotite.structure.io import pdbx
    from .dataset_audit import chain_metadata
    require_compute(); campaign=Path(campaign); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    parent=json.loads((campaign/'manifest.json').read_text()); source=Path(parent['source_audit'])/'sources'
    choices={}
    for path in sorted((campaign/'rows').glob('*.json')):
        row=json.loads(path.read_text())
        if row['status']!='accepted': continue
        evidence=json.loads((campaign/'structures'/row['complex_id']/'input_atom_mask.json').read_text())
        gaps=[d for d in evidence['defects'] if d['kind'] in ('terminal_gap','internal_gap')]
        entry=row['pdb_id'].split('-assembly')[0]
        if gaps: choices.setdefault(entry,(row,evidence,gaps))
    selected=sorted(choices)[:count]
    atomic_json(out/'selection.json',{'entries':selected,'candidate_entries':len(choices),
        'selection':'First sorted distinct accepted PDB entries with whole-residue gaps; bounded enrichment audit, not random prevalence sample.',
        'source_campaign':str(campaign),'reprediction':False})
    receipts={}
    def get(url,name):
        path=out/'downloads'/name
        if not path.exists(): receipts[name]=fetch(url,path); atomic_json(out/'download-receipts.json',receipts); time.sleep(.4)
        return json.loads(path.read_text())
    results=[]
    for entry in selected:
        row,evidence,gaps=choices[entry]; record={'pdb_id':entry,'complex_id':row['complex_id'],'chains':[]}
        try:
            # This endpoint only queries the precomputed database; /pkas can launch pKAI.
            pk=get(f'https://api.pypka.org/query/{entry}',f'{entry}-pkpdb.json')
            record['precomputed_pkpdb']=bool(pk.get('pKas'))
            if not record['precomputed_pkpdb']:
                record['status']='not_in_precomputed_results'; results.append(record); continue
            record['prepared_coordinate_status']='unavailable_through_documented_read_only_API'
            original_path=out/'downloads'/f'{entry}-original.cif'
            receipts[original_path.name]=fetch(f'https://files.rcsb.org/download/{entry}.cif',original_path)
            atomic_json(out/'download-receipts.json',receipts)
            original=pdbx.CIFFile.read(original_path); original_cat=original.block['atom_site']
            original_auth={str(a):str(b) for a,b in zip(original_cat['label_asym_id'].as_array(str),original_cat['auth_asym_id'].as_array(str))}
            original_atoms=pdbx.get_structure(original,model=1,altloc='first',use_author_fields=False)
            original_sequences=chain_metadata(original,original_atoms,sorted(set(original_atoms.chain_id)))
            mappings=get(f'https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{entry}',f'{entry}-sifts.json').get(entry,{}).get('UniProt',{})
            cif=pdbx.CIFFile.read(source/f"{row['pdb_id']}.cif"); cat=cif.block['atom_site']
            auth={str(label):str(author) for label,author in zip(cat['label_asym_id'].as_array(str),cat['auth_asym_id'].as_array(str))}
            for chain in evidence['sequences']:
                cc=chain['chain']; cg=[g for g in gaps if g['chain']==cc]
                if not cg: continue
                deposited_auth=auth.get(cc,'')
                possible={original_auth[c['chain']] for c in original_sequences if c['sequence']==chain['sequence']
                    and (deposited_auth==original_auth[c['chain']] or (deposited_auth.startswith(original_auth[c['chain']]+'-') and deposited_auth[len(original_auth[c['chain']])+1:].isdigit()))}
                mapped_auth=next(iter(possible)) if len(possible)==1 else None
                item={'label_chain':cc,'assembly_author_chain':deposited_auth,'author_chain':mapped_auth,
                    'chain_mapping':'exact sequence plus original author-chain/copy-suffix verification' if mapped_auth else 'unresolved',
                    'gaps':[{k:g[k] for k in ('kind','start','end','length')} for g in cg],'af_matches':[]}
                for accession,annotation in mappings.items():
                    segments=[m for m in annotation['mappings'] if m['chain_id']==mapped_auth]
                    if not segments: continue
                    try:
                        predictions=get(f'https://alphafold.ebi.ac.uk/api/prediction/{accession}',f'{accession}-afdb.json')
                        # Full sequence monomer surrogate only; never substitute a complex or fragment silently.
                        for prediction in predictions:
                            seq=prediction.get('uniprotSequence','')
                            if not seq: continue
                            if prediction.get('isComplex',False) or prediction.get('uniprotStart',1)!=1 or prediction.get('uniprotEnd',len(seq))!=len(seq): continue
                            covered={}; valid=0
                            for m in segments:
                                mapped=map_positions(chain['sequence'],m,seq)
                                if mapped is not None: covered.update(mapped); valid+=1
                            if not valid: continue
                            missing=[p for g in cg for p in range(g['start'],g['end']+1)]
                            mapped_missing={p:covered[p] for p in missing if p in covered}
                            if not mapped_missing: continue
                            url=prediction.get('cifUrl'); af_id=prediction.get('entryId',accession)
                            if not url: continue
                            dest=out/'downloads'/f'{af_id}.cif'
                            if not dest.exists(): receipts[dest.name]=fetch(url,dest); atomic_json(out/'download-receipts.json',receipts)
                            af=pdbx.CIFFile.read(dest).block['atom_site']
                            positions={int(x) for x in af['label_seq_id'].as_array(str) if x not in ('.','?')}
                            item['af_matches'].append({'uniprot':accession,'afdb_entry':af_id,'coordinate_sha256':digest(dest),
                                'verified_linear_segments':valid,'mapped_missing_positions':[{'construct_position':p,'uniprot_position':u,'afdb_coordinate_present':u in positions} for p,u in mapped_missing.items()],
                                'unmapped_gap_positions':[p for p in missing if p not in covered],
                                'note':'Sequence-verified SIFTS mapping within construct; unmapped termini are not assumed equivalent. AFDB presence does not establish historical pKPDB reconstruction.'})
                    except Exception as exc: item['af_matches'].append({'uniprot':accession,'error':str(exc)})
                record['chains'].append(item)
            record['status']='coverage_checked_prepared_coordinates_unavailable'
        except Exception as exc: record.update(status='retrieval_or_mapping_error',error=str(exc))
        results.append(record); atomic_json(out/'entries'/f'{entry}.json',record)
    report={'attempted_entries':len(selected),'precomputed_entries':sum(r.get('precomputed_pkpdb',False) for r in results),
        'status_counts':dict(Counter(r['status'] for r in results)),
        'entries_with_verified_AFDB_gap_coverage':sum(any(any(m.get('mapped_missing_positions') for m in c['af_matches']) for c in r['chains']) for r in results),
        'conclusion':'AFDB overlap/coverage is supporting evidence only. Whether historical pKPDB coordinates rebuilt missing residues remains unverified without prepared coordinate retrieval.',
        'entries':results,'implementation_sha256':digest(Path(__file__))}
    atomic_json(out/'report.json',report); print(json.dumps({k:v for k,v in report.items() if k!='entries'},indent=2))
