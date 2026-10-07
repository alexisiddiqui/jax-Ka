import sqlite3
import pytest
import numpy as np
from pkabench.pkpdb_pilot_refs import disallowed_hit
from pkabench.pkpdb_pilot_clean import gap_tier
from pkabench.pkpdb_pilot import labels
from pkabench.runtime import atomic_json,digest


def test_component_without_heavy_atoms_is_explicit_rejection(monkeypatch):
    from types import SimpleNamespace
    import biotite.structure as struc
    from pkabench import audit, glycan_buffer_policy
    from pkabench.prep import Rejection
    atoms = struc.AtomArray(2)
    atoms.chain_id = np.array(['A', 'B'])
    atoms.res_id = np.array([1, 2])
    atoms.res_name = np.array(['ALA', 'UNK'])
    atoms.element = np.array(['C', 'H'])
    atoms.coord = np.array([[0., 0., 0.], [4., 0., 0.]])
    component = dict(chain='B', resnum=2, name='UNK', code='ligand', start=1, end=2)
    monkeypatch.setattr(audit, 'component_inventory', lambda *args: [component])
    monkeypatch.setattr(glycan_buffer_policy, 'glycan_inventory', lambda *args: [])
    with pytest.raises(Rejection) as caught:
        glycan_buffer_policy.classify_components(atoms, SimpleNamespace(block={}), {'A': ['A'], 'B': []})
    assert caught.value.code == 'component_without_heavy_atoms'


def test_sequence_boundary_and_fragment_reservation():
    assert disallowed_hit(.9,.8,.2,['benchmark'])
    assert disallowed_hit(1.,1.,1.,['benchmark'])
    assert not disallowed_hit(.899,.99,.99,['benchmark'])
    assert not disallowed_hit(.99,.79,.79,['benchmark'])
    assert disallowed_hit(.3,.8,.8,['experimental'])
    assert not disallowed_hit(.9,.8,.2,['experimental'])
    assert disallowed_hit(.3,.9,.9,['experimental','benchmark'])


def test_calibrated_anchor_and_unknown_long_tail():
    d=dict(kind='terminal_gap',length=1,centres=[[0,0,0]],extent=8.)
    gap=(d,10.,np.array([[0.,0,0]]))
    assert gap_tier(np.array([[9.,0,0]]),True,[gap])=='near_gap'
    assert gap_tier(np.array([[11.,0,0]]),True,[gap])=='uncertain'
    assert gap_tier(np.array([[40.,0,0]]),True,[gap])=='clean'
    assert gap_tier(np.array([[40.,0,0]]),False,[gap])=='ineligible'
    long=dict(d,length=30)
    assert gap_tier(np.array([[40.,0,0]]),True,[(long,None,None)])=='uncalibrated'


def test_label_index_preserves_numbering_and_duplicates(tmp_path):
    root=tmp_path/'runtime';source=root/'pretraining/pkpdb-v1';source.mkdir(parents=True)
    path=source/'pkas.csv'
    path.write_text('idcode;chain;residue_name;residue_number;pk\n1abc;A;ASP;-2;3.2\n1abc;B;LYS;10A;9.4\n1abc;A;ASP;-2;3.3\n')
    atomic_json(source/'labels-receipt.json',{'sha256':digest(path)})
    out=tmp_path/'pilot';out.mkdir();labels(root,out)
    db=sqlite3.connect(out/'labels.sqlite');rows=db.execute('select chain,kind,number,pka from labels').fetchall();db.close()
    assert rows==[('A','ASP','-2',3.2),('B','LYS','10A',9.4),('A','ASP','-2',3.3)]
