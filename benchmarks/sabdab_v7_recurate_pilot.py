"""Recurate native pilot masks with pinned CCD metadata; reuse teacher bytes."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import time

from pkabench.runtime import atomic_json,digest,require_compute
from pkabench.sabdab_units import load_prepared
from pkabench.sabdab_labels import source_evidence,make_sites,qualify_labels
from pkabench.sabdab_scope import crop_record, POLICY


def run(args):
    require_compute(threads=1)
    if args.output.exists():raise FileExistsError(args.output)
    old=json.loads((args.labels/'summary.json').read_text());args.output.mkdir(parents=True)
    metadata={};metadata_pin=None
    if args.metadata:
        metadata_pin=dict(path=str(args.metadata),sha256=digest(args.metadata))
        wanted={r['unit']['unit_id'].split(':',1)[1] for r in old['records']}
        with args.metadata.open() as handle:
            for row in csv.DictReader(handle):
                if row['INSTANCE'] in wanted:
                    if row['INSTANCE'] in metadata:raise ValueError('Duplicate published metadata instance')
                    metadata[row['INSTANCE']]=row
        if set(metadata)!=wanted:raise ValueError('Missing selected published metadata')
    records=[];start=time.monotonic()
    for previous in old['records']:
        if previous['status']!='complete':raise ValueError('Cannot reuse failed native teachers')
        unit=previous['unit'];cid=unit['unit_id'].split(':',1)[1];source=args.labels/'entries'/cid
        for name,expected in previous['files_sha256'].items():
            if digest(source/name)!=expected:raise ValueError('Native teacher input changed')
        entry=args.output/'entries'/cid;entry.mkdir(parents=True)
        topology,meta=load_prepared(unit);crop=None
        if args.metadata:
            row=metadata[cid]
            if row['ab_ag_split']!=unit['native_source_split'] or row['ab_ag_cluster']!=unit['component']:
                raise ValueError('Published crop metadata split/component differs from selected unit')
            crop=crop_record(row,metadata_pin)
        evidence,real,known,residues=source_evidence(unit,topology,meta,antibody_crop=crop)
        if args.metadata and evidence['antibody_scope']['status']!='published_boundaries_verified':
            raise ValueError('Published antibody scope was not verified')
        terminal=json.loads((source/'terminal-evidence.json').read_text())
        if sorted(map(list,real))!=terminal['real'] or sorted(map(list,known))!=terminal['known']:
            raise ValueError('Curation-only update changed native terminal evidence')
        for name in ['AB.pdb','A.pdb','B.pdb','request.json','labels.json','terminal-evidence.json','teacher.log']:
            shutil.copyfile(source/name,entry/name)
        if crop:atomic_json(entry/'crop-metadata.json',crop)
        labels=json.loads((entry/'labels.json').read_text())
        sites=make_sites(topology,meta,evidence,real,known,residues);counts=qualify_labels(sites,labels)
        atomic_json(entry/'source-evidence.json',evidence);atomic_json(entry/'sites.json',sites)
        current=dict(previous,counts=counts,mask_reasons=dict(Counter(s['gap_rule'] for s in sites)),
            teacher_reused_byte_identical=True,teacher_source_receipt_sha256=digest(source/'receipt.json'),
            component_policy='mask-all-v1 using source or pinned CCD types; untyped fallback 30 A',
            supervision_scope_policy=evidence['antibody_scope']['policy'],
            intentional_omissions=evidence['intentional_omissions'],
            files_sha256={p.name:digest(p) for p in entry.iterdir() if p.is_file()})
        atomic_json(entry/'receipt.json',current);records.append(current)
    summary=dict(old,records=records,curation_update_wall_s=time.monotonic()-start,
        prior_labels_sha256=digest(args.labels/'summary.json'),teacher_reused_byte_identical=True,
        component_policy='Existing mask-all-v1 radii after pinned local CCD type lookup; only unknown types use 30 A',
        supervision_scope_policy=POLICY if args.metadata else 'from_prepared_metadata_or_conservative_fallback')
    atomic_json(args.output/'summary.json',summary)
    manifest=json.loads((args.labels/'manifest.json').read_text());manifest.update(prior_label_manifest_sha256=digest(args.labels/'manifest.json'),
        curation_adapter_sha256=digest(Path(__file__).parents[1]/'src/pkabench/sabdab_labels.py'),
        scope_adapter_sha256=digest(Path(__file__).parents[1]/'src/pkabench/sabdab_scope.py'),
        published_metadata_pin=metadata_pin,
        supervision_protocol='published IMGT antibody scope; unresolved retained residues and deposited antigen gaps keep long-gap-v1',
        component_dictionaries=[json.loads((args.output/'entries'/r['unit']['unit_id'].split(':',1)[1]/'source-evidence.json').read_text())['component_dictionary'] for r in records])
    atomic_json(args.output/'manifest.json',manifest)
    print(json.dumps(dict(status='native_teachers_reused_with_ccd_qualified_curation',records=len(records),wall_s=summary['curation_update_wall_s'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--metadata',type=Path,help='Pinned published SAbDab abag_split.csv for explicit antibody crop boundaries')
    run(p.parse_args())
