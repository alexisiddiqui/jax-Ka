"""Verify antibody scope correction without weakening other native safeguards."""
import argparse
import json
from pathlib import Path

from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.sabdab_scope import POLICY


def read(path):return json.loads(Path(path).read_text())


def gap_key(d):
    return (d['kind'],d['chain'],d['start'],d['end'],d['length'],tuple(map(tuple,d['centres'])))


def run(args):
    require_compute(threads=1)
    before=read(args.before/'summary.json');after=read(args.after/'summary.json')
    old={r['unit']['unit_id']:r for r in before['records']};rows=[];pins={}
    for receipt in after['records']:
        unit=receipt['unit'];uid=unit['unit_id'];cid=uid.split(':',1)[1]
        if unit!=old[uid]['unit']:raise ValueError('Scope correction changed native input, source identity or split')
        a=args.before/'entries'/cid;b=args.after/'entries'/cid
        for name,expected in receipt['files_sha256'].items():
            if digest(b/name)!=expected:raise ValueError('Corrected curation output changed')
        for name in ['labels.json','AB.pdb','A.pdb','B.pdb','request.json','terminal-evidence.json']:
            if digest(a/name)!=digest(b/name):raise ValueError('Scope correction changed native teacher/termini: '+name)
        previous=read(a/'source-evidence.json');current=read(b/'source-evidence.json')
        if current['antibody_scope']['status']!='published_boundaries_verified':raise ValueError('Antibody crop not proven')
        for name in ['atoms','components','component_dictionary','components_without_heavy_atoms']:
            if current[name]!=previous[name]:raise ValueError('Non-crop defect/context evidence changed: '+name)
        old_gaps={gap_key(d) for d in previous['defects'] if d['kind'].endswith('_gap')}
        new_gaps={gap_key(d) for d in current['defects'] if d['kind'].endswith('_gap')}
        if not new_gaps<=old_gaps:raise ValueError('Scope correction introduced/changed retained gaps')
        ranges={}
        for domain in current['antibody_scope']['domains']:
            ranges.setdefault(domain['chain'],[]).append((domain['source_start'],domain['source_end']))
        removed=old_gaps-new_gaps
        for _,chain,start,end,_,_ in removed:
            if chain not in ranges or any(max(start,lo)<=min(end,hi) for lo,hi in ranges[chain]):
                raise ValueError('Removed gap overlaps a retained unit or belongs to the antigen')
        def sites(path):
            return {(s['chain'],s['resnum'],s['icode'],s['group']):s for s in read(path/'sites.json')}
        first=sites(a);second=sites(b)
        if first.keys()!=second.keys():raise ValueError('Scope correction changed candidate site inventory')
        for key,site in first.items():
            for mask in ['train_mask','eval_mask']:
                if site[mask] and not second[key][mask]:raise ValueError('Scope correction lost previously eligible supervision')
            if site['true_terminus']!=second[key]['true_terminus']:raise ValueError('Crop promoted a false natural terminus')
        mask='train_mask' if unit['split']=='train' else 'eval_mask'
        rows.append(dict(unit_id=uid,operational_role=unit['split'],before_sites=old[uid]['counts'][mask],
            after_sites=receipt['counts'][mask],before_interface_sites=old[uid]['counts'][mask+'_interface'],
            after_interface_sites=receipt['counts'][mask+'_interface'],removed_out_of_scope_gap_regions=len(removed),
            retained_gap_regions=len(new_gaps),teacher_bytes_and_true_termini_unchanged=True,
            component_and_missing_atom_evidence_unchanged=True))
        pins[str(b/'receipt.json')]=digest(b/'receipt.json')
    original=read(args.original_preparation/'prepared-units.json')
    prepared=read(args.preparation/'prepared-units.json')
    lookup={u['unit_id']:u for u in original['units']}
    if {u['unit_id'] for u in prepared['units']}!=set(lookup):raise ValueError('Fresh preparation membership changed')
    for unit in prepared['units']:
        prior=read(Path(lookup[unit['unit_id']]['entry'])/'structure.json')
        current=read(Path(unit['entry'])/'structure.json')
        for name in ['native_coordinate_sha256','residue_map','side_by_chain','n_atoms','n_res']:
            if current[name]!=prior[name]:raise ValueError('Crop metadata changed physical preparation')
        if current['antibody_scope']['status']!='published_boundaries_verified':raise ValueError('Fresh prepared antibody scope missing')
    loader=read(args.loader/'report.json')
    for path in [args.before/'summary.json',args.after/'summary.json',args.preparation/'summary.json',args.loader/'report.json']:
        pins[str(path)]=digest(path)
    report=dict(status='published_antibody_scope_fix_verified',policy=POLICY,examples=rows,
        qualified_paired_complexes_before=sum(r['before_interface_sites']>0 for r in rows),
        qualified_paired_complexes_after=sum(r['after_interface_sites']>0 for r in rows),
        fresh_native_preparations_with_identical_geometry=len(prepared['units']),
        fresh_preparation_failures=read(args.preparation/'summary.json')['failures'],
        packed_reader=loader,teacher_recalculation_required=False,masif_recalculation_required=False,
        production_release_activated=False,pins=pins)
    atomic_json(args.output,report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['before','after','original-preparation','preparation','loader','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
