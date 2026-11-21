import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from masif_surface_quality import complete_terminal_oxygens,clean_mesh,validate_coverage,validate_reduce_warning
from masif_partner_units import antibody_side


def test_roles_use_domains_not_uniprot_and_reject_ambiguity():
    key=lambda s:'s'+hashlib.sha256(s.encode()).hexdigest()[:20]
    numbered={key('antibody'):[dict(chain_type='H',cdr3=[[105,'','A']])]}
    assert antibody_side(['antigen','antibody'],numbered)[0]==1
    for seqs in [('antigen','unknown'),('antibody','antibody')]:
        with pytest.raises(ValueError,match='exactly one'):antibody_side(seqs,numbered)


def test_terminal_completion_preserves_existing_atoms_and_every_fragment():
    atom=dict(CA=np.array([-1.,0.,0.]),C=np.zeros(3),O=np.array([.5,1.,0.]))
    atoms=[dict(atom),dict(atom),dict(atom,OXT=np.array([.5,-1.,0.]))]
    repaired,added=complete_terminal_oxygens(atoms,[0,1,1])
    assert added==[0] and 'OXT' not in atoms[0] and 'OXT' not in repaired[1]
    assert np.isclose(np.linalg.norm(repaired[0]['OXT']),1.25)
    for original,new in zip(atoms,repaired):
        for name,xyz in original.items():np.testing.assert_array_equal(xyz,new[name])


def test_final_cleanup_removes_repeated_faces_and_three_vertex_island():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[8,0,0],[8,1,0],[8,0,1]],float)
    f=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3],[0,0,1],[2,1,0],[4,5,6]])
    vv,ff,info=clean_mesh(v,f)
    assert vv.shape==(4,3) and ff.shape==(4,3) and info['removed_faces']==3
    assert validate_coverage(vv,vv,vv)['raw_surface_max_nearest_A']==0
    with pytest.raises(ValueError,match='coverage loss'):validate_coverage(v,vv,v)


def test_coverage_catches_finite_but_collapsed_mesh():
    with pytest.raises(ValueError,match='coverage loss'):
        validate_coverage(np.array([[0.,0,0],[20,0,0]]),np.array([[0.,0,0],[1,0,0]]),np.array([[10.,0,0]]))


def test_v5_partner_unit_keeps_multichain_and_gap_fragments_together(tmp_path,monkeypatch):
    import json
    from masif_partner_units import inventory_v5,sha,UNIT_POLICY
    import pkabench.training_pools as pools
    table=tmp_path/'pool/pool-v5';table.mkdir(parents=True)
    runtime=tmp_path/'runtime';prefilter=runtime/'audits/pinder-prefilter-v1';prefilter.mkdir(parents=True)
    (prefilter/'prefiltered_v1.parquet').write_bytes(b'fixture')
    (prefilter/'uniq.fasta').write_text('>a\nantibody\n>b\nantigen\n')
    (table/'pinder.tsv').write_text('id\tgroup\tstratum\ncomplex\tg\tAb/Ag\n')
    (table/'pinder-val.tsv').write_text('id\tgroup\tstratum\n')
    key='s'+hashlib.sha256(b'antibody').hexdigest()[:20]
    (tmp_path/'pool/anarci.json').write_text(json.dumps(dict(numbered={key:[dict(chain_type='H',cdr3=[[105,'','A']])]})))
    (table/'pool-v5.json').write_text(json.dumps(dict(
        abag_rule=dict(anarci_json_sha256=sha(tmp_path/'pool/anarci.json')),
        datasets=dict(pinder=dict(pool=dict(sha256=sha(table/'pinder.tsv')),
                                 validation=dict(sha256=sha(table/'pinder-val.tsv')))))))
    entry=runtime/'pretraining/pinder-pkai-v1/entries/complex';entry.mkdir(parents=True)
    (entry/'AB.cif.gz').write_bytes(b'fixture')
    # One antibody assembly has H + L, and H is split into two exported fragments.
    (entry/'map_A.json').write_text(json.dumps([['A',1,'H',1,''],['B',2,'H',5,''],['C',3,'L',1,'']]))
    (entry/'map_B.json').write_text(json.dumps([['A',1,'G',1,'']]))
    monkeypatch.setattr(pools,'pinder_chains',lambda _:({'complex':('a','b')},{'a':'antibody','b':'antigen'}))
    units,metadata=inventory_v5(tmp_path,runtime)
    assert len(units)==2 and [u['role'] for u in units]==['Ab','Ag']
    assert units[0]['source_chain_labels']==['H','L'] and units[0]['fragment_chain_labels']==['A','B','C']
    assert units[0]['n']==3 and metadata['unit_policy']==UNIT_POLICY
    (table/'pinder.tsv').write_text('tampered')
    with pytest.raises(ValueError,match='membership table'):inventory_v5(tmp_path,runtime)


def test_reduce_warning_requires_complete_identity_backbone_and_hydrogen_counts():
    def atom(serial,name,element,x):
        return f'ATOM  {serial:5d}  {name:<3s} ALA A   1    {x:8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{0.:6.2f}          {element:>2s}'
    source='\n'.join([atom(1,'N','N',0),atom(2,'CA','C',1.4),atom(3,'C','C',2.8),atom(4,'O','O',3.8)])+'\nEND\n'
    output='USER  MOD reduce.3.23.130521 H: found=0 add=1\n'+source.replace('END\n',atom(0,'H','H',-.8)+'\nEND\n')
    stderr='Found 0 hydrogens (0 hets)\nAdded 1 hydrogens (0 hets)\nRemoved 0 hydrogens (0 hets)\n'
    assert validate_reduce_warning(source,output,stderr)['added_hydrogens']==1
    for bad in [output.replace('END\n',''),output.replace(atom(4,'O','O',3.8)+'\n',''),
                output.replace(atom(2,'CA','C',1.4),atom(2,'CA','C',5.4)),output.replace('reduce.3.23.130521','reduce.unknown')]:
        with pytest.raises(ValueError):validate_reduce_warning(source,bad,stderr)
    with pytest.raises(ValueError):validate_reduce_warning(source,output,stderr.replace('Added 1','Added 2'))
