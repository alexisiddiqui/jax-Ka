"""Diagnostic crop sensitivity only; never replaces native training masks."""
import argparse
import copy
import json
from pathlib import Path

from pkabench.runtime import atomic_json,require_compute
from pkabench.sabdab_units import load_prepared
from pkabench.sabdab_labels import make_sites,qualify_labels


def run(args):
    require_compute(threads=1)
    report=json.loads((args.labels/'summary.json').read_text());rows=[]
    for receipt in report['records']:
        if receipt['status']!='complete':continue
        unit=receipt['unit'];entry=args.labels/'entries'/unit['unit_id'].split(':',1)[1]
        topology,meta=load_prepared(unit)
        evidence=json.loads((entry/'source-evidence.json').read_text())
        terminal=json.loads((entry/'terminal-evidence.json').read_text())
        real={tuple(k) for k in terminal['real']};known={tuple(k) for k in terminal['known']}
        residues={(k.chain,k.number,k.insertion):topology.residue(i) for i,k in enumerate(topology.keys)}
        labels=json.loads((entry/'labels.json').read_text())
        positions={(m['chain'],m['source_label_position']):m for m in meta['residue_map']}
        results={}
        for scenario in ['relax_antibody_crop_boundaries','relax_all_declared_unit_boundaries']:
            relaxed=copy.deepcopy(evidence);removed=[];retained=[]
            for defect in evidence['defects']:
                drop=False
                if defect['kind']=='terminal_gap':
                    drop=scenario=='relax_all_declared_unit_boundaries' or meta['side_by_chain'][defect['chain']]=='A'
                elif defect['kind']=='internal_gap' and meta['side_by_chain'][defect['chain']]=='A':
                    left=positions.get((defect['chain'],defect['start']-1));right=positions.get((defect['chain'],defect['end']+1))
                    drop=bool(left and right and left['source_author_chain']!=right['source_author_chain'])
                (removed if drop else retained).append(defect)
            relaxed['defects']=retained
            sites=make_sites(topology,meta,relaxed,real,known,residues)
            counts=qualify_labels(sites,labels)
            results[scenario]=dict(counts=counts,removed_gap_regions=[dict(kind=d['kind'],chain=d['chain'],length=d['length']) for d in removed])
        rows.append(dict(unit_id=unit['unit_id'],operational_role=unit['split'],baseline=receipt['counts'],
            baseline_reasons=receipt['mask_reasons'],diagnostics=results))
    atomic_json(args.output,dict(status='diagnostic_only_no_mask_changes',records=rows,
        interpretation='Boundary relaxations test how full deposited-sequence gaps interact with the declared ML scope. They are unapproved sensitivity analyses, not validated replacements.',
        true_terminal_evidence_unchanged=True,native_labels_or_masks_modified=False,active_release_changed=False))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
