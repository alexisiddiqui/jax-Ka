"""Exploratory retention overlay; does not change production supervision masks."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import biotite.structure as struc
from .runtime import require_compute, atomic_json, digest
from .supervision import clearance
from .annotate import SITE_ATOMS


def terminal_radius(length):
    # Intermediate lengths use the next tested dose: an explicit audit assumption.
    return next((radius for n,radius in ((1,10.),(3,15.),(5,15.),(10,20.)) if 0<length<=n),None)


def classify(eligible, clean, near, uncalibrated):
    if not eligible: return 'ineligible'
    if near: return 'near_gap'
    if uncalibrated: return 'uncalibrated'
    return 'clean' if clean else 'uncertain'


def apply(campaign,out):
    require_compute()
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .schema import read_table
    from .prep import read_cif
    campaign=Path(campaign); out=Path(out); out.mkdir(exist_ok=False,parents=True)
    rows=[]; sources={}; per_complex=[]; gap_counts=Counter()
    for structure in read_table(campaign/'structures.parquet'):
        cid=structure['complex_id']; root=campaign/'structures'/cid
        evidence=json.loads((root/'input_atom_mask.json').read_text()); atoms=read_cif(root/'AB.cif')
        starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
        residues={(str(atoms.chain_id[s]),int(atoms.res_id[s]),str(atoms.ins_code[s]).strip()):atoms[s:e] for s,e in zip(starts[:-1],starts[1:])}
        labels={(r['key'][0],r['label_seq_id']):tuple(r['key']) for r in evidence['atoms']}
        known=bool(evidence.get('sequences')) and all(s['sequence_source']=='entity_poly_canonical' for s in evidence['sequences'])
        gaps=[]
        for d in evidence['defects']:
            if d['kind'] not in ('terminal_gap','internal_gap'): continue
            internal=d['kind']=='internal_gap'
            radius=(15. if d['length']<=3 else None) if internal else terminal_radius(d['length'])
            keys=[labels[d['chain'],pos] for pos in (d['start']-1,d['end']+1) if (d['chain'],pos) in labels]
            anchor=None
            if radius is not None and len(keys)==(2 if internal else 1) and all(k in residues for k in keys):
                # Calibration uses visible backbone atoms; exclude repaired atoms.
                anchors=[]
                for key in keys:
                    a=residues[key]
                    inventory=next(r for r in evidence['atoms'] if tuple(r['key'])==key)
                    names=set(('N','CA','C','O'))-set(inventory['missing_atoms'])
                    anchors.append(a.coord[np.isin(a.atom_name,list(names))])
                anchor=np.concatenate(anchors)
                if any(len(a)!=4 for a in anchors): radius=None
            else: radius=None
            category=('short_internal_calibrated' if radius is not None else 'internal_gap_uncalibrated') if internal else 'long_tail' if d['length']>10 else 'short_tail_anchor_unavailable' if radius is None else 'short_tail_calibrated'
            gap_counts[category]+=1; gaps.append((d,radius,anchor,category))
        current=[]
        for site in read_table(root/'sites.parquet'):
            a=residues[site['chain'],site['resnum'],site['icode']]
            points=a.coord[np.isin(a.atom_name,SITE_ATOMS[site['group']])]
            eligible=bool(known and site['functional_atoms_complete'] and not site['is_break_terminus'])
            clearances=[clearance(points,d) for d,_,_,_ in gaps]
            near=False; unresolved=[]; details=[]
            for (d,radius,anchor,category),distance in zip(gaps,clearances):
                ad=float(np.linalg.norm(points[:,None]-anchor[None,:],axis=-1).min()) if radius is not None and len(points) else None
                if radius is not None and ad is not None: near |= ad<radius
                elif distance<20: unresolved.append(category)
                details.append({'chain':d['chain'],'start':d['start'],'length':d['length'],'kind':d['kind'],'anchor_distance_A':ad,'radius_A':radius,'envelope_clearance_A':distance,'category':category})
            clean=all(x>=20 for x in clearances)
            tier=classify(eligible,clean,near,bool(unresolved))
            row={k:site[k] for k in ('complex_id','chain','resnum','icode','group')}
            row.update(tier=tier,native_site_eligible=eligible,interface=site['residue_delta_sasa']>10,
                provisional_retained=tier in ('clean','uncertain'),has_near_calibrated_gap=near,
                has_uncalibrated_influence=bool(unresolved),uncalibrated_reasons='|'.join(sorted(set(unresolved))),
                old_train_mask=eligible and all(x>=10 for x in clearances),old_eval_mask=eligible and clean,
                gap_details=json.dumps(details))
            current.append(row)
        rows.extend(current); counts=Counter(r['tier'] for r in current)
        per_complex.append({'complex_id':cid,'pdb_id':structure['pdb_id'],'counts':dict(counts),'provisional_retained':counts['clean']+counts['uncertain']})
        sources[cid]={name:digest(root/name) for name in ('AB.cif','sites.parquet','input_atom_mask.json')}
    pq.write_table(pa.Table.from_pylist(rows),out/'site_tiers.parquet')
    report={'complexes':len(per_complex),'tier_counts':dict(Counter(r['tier'] for r in rows)),
        'interface_tier_counts':dict(Counter(r['tier'] for r in rows if r['interface'])),
        'eligible_sites':sum(r['native_site_eligible'] for r in rows),
        'provisional_retained':sum(r['provisional_retained'] for r in rows),
        'old_train_retained':sum(r['old_train_mask'] for r in rows),'old_eval_retained':sum(r['old_eval_mask'] for r in rows),
        'newly_retained_vs_old_train':sum(r['provisional_retained'] and not r['old_train_mask'] for r in rows),
        'lost_vs_old_train':sum(r['old_train_mask'] and not r['provisional_retained'] for r in rows),
        'complexes_with_provisional_sites':sum(r['provisional_retained']>0 for r in per_complex),
        'gap_counts':dict(gap_counts),'per_complex':per_complex,'production_masks_changed':False,
        'note':'Structural eligibility only; no teacher coverage or split assignment. Near-gap tier takes precedence; overlapping uncalibrated influence is retained as a separate flag.'}
    atomic_json(out/'report.json',report)
    atomic_json(out/'manifest.json',{'source_campaign':str(campaign),'sources':sources,'code_sha256':digest(Path(__file__)),
        'output_sha256':digest(out/'site_tiers.parquet'),
        'policy':'Audit only. Clean: conservative whole-residue gap clearance >=20 A. Uncertain: fails clean but passes all calibrated anchors and >=20 A from uncalibrated envelopes. Near: distance < radius at any calibrated terminal gap. Uncalibrated: remaining influence of internal/long/unknown-anchor gap. Ineligible: unknown sequence coverage, incomplete target functional atoms or artificial terminus.',
        'radii':'Terminal N=1:10 A; N=2..5:15 A; N=6..10:20 A. Internal N=1..3:15 A to nearest flank. Untested intermediate lengths rounded up to next tested dose, including internal N=2; not independently validated. Terminal N>10 and internal N>3 uncalibrated.',
        'calibration':'radial-error-v2-nearcomplete/anchor-analysis-v2; paired delta-pKa, 23 complexes; exploratory local-band bounds.'})
    print(json.dumps({k:v for k,v in report.items() if k!='per_complex'},indent=2))
