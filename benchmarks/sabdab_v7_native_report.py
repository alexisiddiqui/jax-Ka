"""Collect native-pilot evidence and verify real biological-side graph masks."""
import argparse
import json
from pathlib import Path

import numpy as np

from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.sabdab_units import load_prepared
from pkanet.graph import geometry


def read(path):return json.loads(Path(path).read_text())


def run(args):
    require_compute(threads=1);root=args.root
    label_root=args.labels or root/'labels-v1';loader_root=args.loader or root/'loader-v1'
    crop_path=args.crop_audit or root/'crop-mask-audit.json'
    labels=read(label_root/'summary.json');masif=read(root/'masif-v1/report.json')
    loader=read(loader_root/'report.json');surfaces=read(root/'masif-reader-report.json')
    crops=read(crop_path);crop_by_id={r['unit_id']:r for r in crops['records']}
    timing_by_id={}
    for row in masif['parents']:
        timing_by_id.setdefault(row['cid'],[]).append(row)
    examples=[];branch_checks=[]
    for row in labels['records']:
        unit=row['unit'];topology,meta=load_prepared(unit)
        graph,_=geometry(topology.backbone,topology.chain_index,radius=20.)
        side=np.asarray([meta['side_by_chain'][k.chain] for k in topology.keys])
        same=side[graph['neighbors']]==side[:,None]
        free=graph['edge_mask'] & same
        cross=topology.chain_index[graph['neighbors']]!=topology.chain_index[:,None]
        expected=int(np.sum(graph['edge_mask'] & same & cross));actual=int(np.sum(free & cross))
        if expected!=actual or (free & ~same).any():raise ValueError('Native biological-side free branch differs')
        if len([c for c,s in meta['side_by_chain'].items() if s=='A'])>1 and not actual:
            raise ValueError('Native joint antibody has no retained H/L edges')
        branch_checks.append(dict(unit_id=unit['unit_id'],retained_edges_between_same_side_physical_chains=actual,
            removed_edges_between_opposite_sides=int(np.sum(graph['edge_mask'] & ~same)),
            antibody_physical_chains=[c for c,s in meta['side_by_chain'].items() if s=='A']))
        cid=unit['unit_id'].split(':',1)[1];features=timing_by_id[cid]
        if {f['state'] for f in features}!={'AB','A','B'}:raise ValueError('Missing native feature state')
        crop=crop_by_id[unit['unit_id']]
        examples.append(dict(unit_id=unit['unit_id'],residues=unit['n'],role=unit['split'],
            antibody_type=meta['antibody_type'],construct=meta['construct'],
            pkai_total_wall_s=row['wall_s'],pkai_teacher_process_s=row['timings_s']['teacher_process'],
            pkai_api_s=sum(row['teacher_timings_s'].values()),
            qualified_paired_sites=row['counts']['train_mask' if unit['split']=='train' else 'eval_mask'],
            qualified_paired_interface_sites=row['counts']['train_mask_interface' if unit['split']=='train' else 'eval_mask_interface'],
            masif_surface_count=len(features),masif_annotation_s=sum(f['annotation_s'] for f in features),
            masif_stage_s={name:sum(f['timings'].get(name,0.) for f in features) for name in sorted({k for f in features for k in f['timings']})},
            relaxed_antibody_boundary_diagnostic=crop['diagnostics']['relax_antibody_crop_boundaries']['counts'],
            relaxed_all_boundary_diagnostic=crop['diagnostics']['relax_all_declared_unit_boundaries']['counts']))
    pins=[root/'selected-units.json',label_root/'manifest.json',label_root/'summary.json',
        root/'masif-v1/report.json',root/'masif-qualified-v1/manifest.json',root/'masif-reader-report.json',
        loader_root/'report.json',crop_path]
    report=dict(status='native_annotation_pilot_complete',complexes=len(examples),examples=examples,
        label_models=['pKAI','pKAI+'],label_states=['AB','A','B'],masif_states=['AB','A','B'],
        pkai_pilot_wall_s=labels['wall_s'],pkai_workers=5,
        masif_pilot_wall_s=masif['wall_s'],masif_workers=4,masif_allocated_cpus=20,masif_allocated_gpus=1,
        paired_loader=loader,surface_reader=surfaces,real_geometry_branch_checks=branch_checks,
        release_ready=False,active_release_changed=False,model_training_performed=False,
        remaining_decision='Qualify a declared-unit crop/masking policy consistently across sources before expanding SAbDab supervision.',
        limitations=['Selected integration panel; not a full-pool yield or unbiased runtime sample.',
            'pKAI timings include one cold interpreter/model environment per complex.',
            'MaSIF uses batch size one, including container, model and MATLAB startup; full-build amortisation is not measured.',
            'Boundary-relaxed counts are diagnostic only; stored supervision retains the original conservative masks.',
            'No cache ownership, hard-test freeze or production recipe activation is implied.'],
        input_pins={str(p):digest(p) for p in pins})
    atomic_json(args.output or root/'pilot-report.json',report)
    print(json.dumps(dict(status=report['status'],complexes=report['complexes'],pkai_pilot_wall_s=report['pkai_pilot_wall_s'],
        masif_pilot_wall_s=report['masif_pilot_wall_s'],native_graph_records=loader['native_records'],native_surface_records=surfaces['surface_records'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    for name in ['labels','loader','crop-audit','output']:parser.add_argument('--'+name,type=Path)
    run(parser.parse_args())
