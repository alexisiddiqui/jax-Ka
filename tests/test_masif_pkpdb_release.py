import sys
from pathlib import Path
import copy

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from masif_pkpdb_release import validate_record
from masif_pkpdb_abag import UNIT_POLICY
from masif_surface_quality import SURFACE_POLICY


def test_transfer_validation_rejects_lost_chains_bad_mesh_and_unchecked_descriptors():
    unit=dict(role='Ab',chain_indices=[0,1],source_files_sha256={'structure':'native'},canonical_residue_ids=[0,1])
    arrays=dict(vertices=np.array([[0.,0,0],[1,0,0],[0,1,0]]),descriptors=np.ones((3,80)),normals=np.ones((3,3)),
                faces=np.array([[0,1,2]]),owner=np.array([0,0,1]),canonical_residue_ids=np.array([0,1]))
    receipt=dict(surface_policy=SURFACE_POLICY,status='complete',descriptor_max_reference_error=1e-6,
        reference_patches_checked=3,mesh_coverage=dict(source_atoms_outside_bbox_plus_2A=0,raw_surface_max_nearest_A=1.))
    record=dict(metadata=dict(unit,unit_policy=UNIT_POLICY,receipt=receipt),arrays=arrays)
    validate_record(unit,record)
    changes=[('metadata','chain_indices',[0]),('arrays','canonical_residue_ids',np.array([0])),
             ('arrays','faces',np.array([[0,0,1]])),('arrays','descriptors',np.full((3,80),np.nan))]
    for section,key,value in changes:
        bad=copy.deepcopy(record);bad[section][key]=value
        with pytest.raises(ValueError):validate_record(unit,bad)
    bad=copy.deepcopy(record);bad['metadata']['receipt']['reference_patches_checked']=0
    with pytest.raises(ValueError):validate_record(unit,bad)
    with pytest.raises(ValueError,match='model/runtime'):
        validate_record(unit,record,expected_profile={'checkpoint_sha256':{'data':'different'}})


def test_quarantine_holds_the_entire_pair_and_refuses_unreviewed_failures(tmp_path,monkeypatch):
    import json
    import masif_pkpdb_release as module
    from pkatrain.augmentation_store import ReleaseWriter,AugmentationStore
    pairs=[]
    for cid in ('good','bad'):
        for role in ('Ab','Ag'):
            pairs.append(dict(id='pkpdb-abag/'+cid+'/complex-000/'+role,cid=cid,role=role,
                split='val' if cid=='bad' else 'train',group=cid,source_chain_labels=['H','L'] if role=='Ab' else ['G'],
                chain_indices=[0,1] if role=='Ab' else [2],source_files_sha256={'structure':cid},
                canonical_residue_ids=[0,1] if role=='Ab' else [2,3],unit_policy=UNIT_POLICY))
    monkeypatch.setattr(module,'load_inventory',lambda *a:(pairs,dict(inventory_sha256='inventory')))
    build=tmp_path/'build';build.mkdir();failed=pairs[-1]['id']
    profile=dict(checkpoint_sha256={'data':'checkpoint'},paper_runtime_sha256='runtime',preprocessing_profile='paper')
    parents=[dict(id=u['id'],status='failed' if u['id']==failed else 'complete',stage='electrostatics',error='incomplete source') for u in pairs]
    (build/'report.json').write_text(json.dumps(dict(profile,status='partial',parents=parents,counts={'complete':3,'failed':1})))
    with ReleaseWriter(build/'fingerprints',dict(profile),zstd_level=1) as writer:
        for u in pairs[:-1]:
            receipt=dict(profile,surface_policy=SURFACE_POLICY,status='complete',descriptor_max_reference_error=1e-6,
                reference_patches_checked=3,mesh_coverage=dict(source_atoms_outside_bbox_plus_2A=0,raw_surface_max_nearest_A=1.))
            arrays=dict(vertices=np.array([[0.,0,0],[1,0,0],[0,1,0]]),descriptors=np.ones((3,80)),normals=np.ones((3,3)),
                faces=np.array([[0,1,2]]),owner=np.array([0,0,1]),canonical_residue_ids=np.array(u['canonical_residue_ids']))
            writer.add(u['id'],dict(u,receipt=receipt),arrays)
        writer.finish(expected_ids=[u['id'] for u in pairs[:-1]])
    with pytest.raises(ValueError,match='unresolved failures'):
        module.publish(tmp_path,tmp_path/'inventory',build,tmp_path/'unreviewed')
    assert not (tmp_path/'unreviewed').exists()
    plan=tmp_path/'quarantine.json';plan.write_text(json.dumps(dict(inventory_sha256='inventory',failures=[dict(id=failed,reason='reviewed incomplete source')])))
    output=tmp_path/'release';module.publish(tmp_path,tmp_path/'inventory',build,output,quarantine=plan)
    with AugmentationStore(output) as store:assert set(store.by_id)=={u['id'] for u in pairs[:2]}
    quality=json.loads((output/'quality.json').read_text())
    assert quality['status']=='complete_with_quarantine' and quality['validated_units']==2 and quality['quarantined_units']==2
