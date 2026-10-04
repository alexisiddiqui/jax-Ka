import json
from pkabench import pool


def test_legacy_extended_ids_preserve_genuine_extended_ids():
    assert pool.entry_id('pdb_000010bt')=='10bt'
    assert pool.entry_id('PDB_12345678')=='pdb_12345678'


def test_frozen_sampling_is_reproducible_disjoint_and_bounded(tmp_path,monkeypatch):
    monkeypatch.setattr(pool,'require_compute',lambda:None)
    metadata=tmp_path/'metadata'; metadata.mkdir()
    (metadata/'assemblies.json').write_text(json.dumps({'assembly_ids':['1ABC-1','2ABC-1','3ABC-1','4ABC-1']}))
    (metadata/'sabdab-candidates.json').write_text(json.dumps({'candidates':[{'PDB':'pdb_00001abc'}]}))
    pool.initialise(metadata,tmp_path/'a',3,1); pool.initialise(metadata,tmp_path/'b',3,1)
    first=json.loads((tmp_path/'a/download-manifest.json').read_text())
    assert first==json.loads((tmp_path/'b/download-manifest.json').read_text())
    rows=first['assemblies']; assert len({r['assembly_id'] for r in rows})==3
    assert sum(r['stratum']=='antibody' for r in rows)==1
    assert rows[0]['pdb_id']=='1abc-assembly1'
