import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
import masif_partner_release as module
from masif_partner_units import UNIT_POLICY
from masif_surface_quality import SURFACE_POLICY
from pkatrain.augmentation_store import ReleaseWriter,AugmentationStore


def test_disjoint_builds_publish_all_chains_and_reject_model_mixing(tmp_path,monkeypatch):
    profile=dict(checkpoint_sha256={'data':'weights'},preprocessing_profile='paper',paper_runtime_sha256='runtime')
    units=[];builds=[]
    for role,state,canonical in [('Ab','A',[0,1]),('Ag','B',[2,3])]:
        u=dict(id='pinder-v5/complex/'+role,dataset='pinder',cid='complex',state=state,role=role,
            unit_policy=UNIT_POLICY,source_files_sha256={'AB.cif.gz':'native'},n=2,split='train',group='group',source_chain_labels=[state])
        units.append(u);build=tmp_path/role;build.mkdir();builds.append(build)
        receipt=dict(profile,status='complete',surface_policy=SURFACE_POLICY,reference_patches_checked=3,descriptor_max_reference_error=1e-6,
                     mesh_coverage=dict(source_atoms_outside_bbox_plus_2A=0,raw_surface_max_nearest_A=1.))
        arrays=dict(vertices=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]),faces=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]]),
            normals=np.ones((4,3)),owner=np.array([0,0,1,1]),raw_features=np.ones((4,4)),descriptors=np.ones((4,80)),canonical_residue_ids=np.array(canonical))
        with ReleaseWriter(build/'fingerprints',profile,zstd_level=1) as writer:
            writer.add(u['id'],dict(u,receipt=receipt,source=dict(canonical_residue_ids=canonical,actual_residues=2)),arrays)
            writer.finish(expected_ids=[u['id']])
        (build/'report.json').write_text(json.dumps(dict(profile,status='complete',parents=[dict(id=u['id'],status='complete')])))
    monkeypatch.setattr(module,'inventory_v5',lambda *args:(units,dict(inventory_sha256='source')))
    monkeypatch.setattr(module,'validate_native',lambda *args:dict(atoms_outside_convex_hull_plus_2A=0))
    module.publish(tmp_path,tmp_path,builds,tmp_path/'release')
    with AugmentationStore(tmp_path/'release') as store:assert set(store.by_id)=={u['id'] for u in units}
    assert json.loads((tmp_path/'release/quality.json').read_text())['pairs']==1
    failed=tmp_path/'failed';failed.mkdir()
    with ReleaseWriter(failed/'fingerprints',profile,zstd_level=1) as writer:writer.finish(expected_ids=[])
    (failed/'report.json').write_text(json.dumps(dict(profile,status='partial',parents=[dict(id=units[1]['id'],status='failed',stage='protonation') ])))
    module.publish(tmp_path,tmp_path,[builds[0],failed],tmp_path/'recovered',recovery_builds=[builds[1]])
    assert json.loads((tmp_path/'recovered/quality.json').read_text())['recovered_units']==1
    with pytest.raises(ValueError,match='duplicate successful'):
        module.publish(tmp_path,tmp_path,builds,tmp_path/'no-overwrite',recovery_builds=[builds[1]])
    report=json.loads((builds[1]/'report.json').read_text());report['checkpoint_sha256']={'data':'other'}
    (builds[1]/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='mixed scientific'):
        module.publish(tmp_path,tmp_path,builds,tmp_path/'must-not-publish')
    assert not (tmp_path/'must-not-publish').exists()
