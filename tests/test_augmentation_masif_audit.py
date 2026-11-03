"""Controls for sampling, exported correspondence, protonation, and weighting."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

BENCH=Path(__file__).resolve().parents[1]/'benchmarks'
sys.path.insert(0,str(BENCH))
import augmentation_masif_audit as audit
import augmentation_masif_report as report
import augmentation_masif_bridge as bridge


def test_sampling_covers_bins_and_separates_typical_from_tail():
    eligible=[]
    for d in audit.DATASETS:
        for b in range(5):
            for i,score in enumerate([1.,2.,9.]):
                eligible.append(dict(parent=f'{d}/{b}/{i}',dataset=d,size_bin=b,
                    mean_local_surface_distance_A=score,p95_local_surface_distance_A=score))
    chosen=audit.choose_sample(eligible)
    assert len(chosen)==30
    assert all(r['parent'].endswith('/1') for r in chosen if r['sample_kind']=='typical')
    assert all(r['parent'].endswith('/2') for r in chosen if r['sample_kind']=='tail')
    assert chosen==audit.choose_sample(list(reversed(eligible)))
    with pytest.raises(ValueError):audit.choose_sample(eligible[:-3])


def test_export_keeps_chains_atom_identity_and_rounding(tmp_path):
    names=['N','CA','C','O']*3
    meta=dict(atom_names=names,residue_names=['ALA','GLY','SER'],disulfides=[],residue_keys=[['A',1],['A',2],['B',1]])
    arrays=dict(segments=np.array([0,0,1]),atom_residue=np.repeat(np.arange(3),4))
    xyz=np.arange(36).reshape(12,3)/7
    path=tmp_path/'input.pdb';audit.write_pdb(path,xyz,meta,arrays)
    parsed=bridge.heavy_pdb(path)
    assert len(parsed)==12
    assert max(np.max(np.abs(parsed[int(r)+1,n]-x)) for r,n,x in zip(arrays['atom_residue'],names,xyz))<=.000501
    lines=path.read_text().splitlines();assert sum(l=='TER' for l in lines)==2
    assert {l[21] for l in lines if l.startswith('ATOM')}=={'A','B'}


def test_histidine_state_is_fixed_without_moving_heavy_atoms(tmp_path,monkeypatch):
    names=['N','CA','C','O','CG','ND1','CE1','NE2','CD2']
    xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[1,2,0],[0,1,0],[-1,1,0],[-1,2,0],[0,3,0],[1,2,0]],float)
    meta=dict(atom_names=names,residue_names=['HIS'],disulfides=[],residue_keys=[['A',1]])
    arrays=dict(segments=np.array([0]),atom_residue=np.zeros(len(names),int))
    path=tmp_path/'input.pdb';audit.write_pdb(path,xyz,meta,arrays)
    monkeypatch.setattr(bridge,'checked',lambda args:path.read_text())
    output,error=bridge.protonate(dict(input=str(path),his_states={'1':['HE2']}))
    assert error==0
    lines=[l for l in output.read_text().splitlines() if l.startswith('ATOM')]
    hydrogen=next(l for l in lines if l[12:16].strip()=='HE2')
    assert hydrogen[17:20]=='HIS' and int(hydrogen[22:26])==1
    h=np.array([float(hydrogen[k:k+8]) for k in [30,38,46]])
    assert np.isclose(np.linalg.norm(h-xyz[names.index('NE2')]),1.,atol=.001)
    assert all(l[12:16].strip()!='HD1' for l in lines)


def test_parent_view_patch_weights_do_not_favor_large_proteins():
    rows=[dict(parent='a',role='one',distance=1.)]*10+[dict(parent='b',role='one',distance=3.)]
    w=report.weights(rows)
    assert np.isclose(w[:10].sum(),.5) and np.isclose(w[10],.5)
    assert report.quantile(rows,'distance')['p95']==3.
    assert report.bootstrap_margin([dict(parent='a',margin=1.),dict(parent='b',margin=2.)],'margin')['proteins']==2


def test_hydrogens_remain_contiguous_with_each_residue(tmp_path,monkeypatch):
    names=['N','CA','C','O','CG','ND1','CE1','NE2','CD2']
    xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[1,2,0],[0,1,0],[-1,1,0],[-1,2,0],[0,3,0],[1,2,0]],float)
    meta=dict(atom_names=names*2,residue_names=['HIS','HIS'],disulfides=[],residue_keys=[['A',1],['A',2]])
    arrays=dict(segments=np.zeros(2,int),atom_residue=np.repeat(np.arange(2),len(names)))
    path=tmp_path/'input.pdb';audit.write_pdb(path,np.concatenate([xyz,xyz+5]),meta,arrays)
    monkeypatch.setattr(bridge,'checked',lambda args:path.read_text())
    output,error=bridge.protonate(dict(input=str(path),his_states={'1':['HE2'],'2':['HD1']}))
    ids=[int(l[22:26]) for l in output.read_text().splitlines() if l.startswith('ATOM')]
    assert ids==sorted(ids) and ids.count(1)==10 and ids.count(2)==10


def test_reduce_product_validation_does_not_mask_fatal_exit(monkeypatch):
    monkeypatch.setattr(bridge.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=b'ATOM  product',stderr=b'adjustments'))
    assert 'ATOM' in bridge.checked(['reduce','input.pdb'])
    monkeypatch.setattr(bridge.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=b'ATOM  incomplete',stderr=b'FATAL error'))
    with pytest.raises(RuntimeError):bridge.checked(['reduce','input.pdb'])


def test_revision_provenance_does_not_require_git_on_compute_node(tmp_path):
    git=tmp_path/'.git';git.mkdir();value='a'*40
    (git/'HEAD').write_text(value+'\n');assert audit.git_revision(tmp_path)==value
    (git/'HEAD').write_text('ref: refs/heads/master\n')
    (git/'packed-refs').write_text('# packed refs\n'+value+' refs/heads/master\n')
    assert audit.git_revision(tmp_path)==value
    (git/'refs/heads').mkdir(parents=True);(git/'refs/heads/master').write_text('b'*40+'\n')
    assert audit.git_revision(tmp_path)=='b'*40
