import gzip
import hashlib
import io

import numpy as np
import pytest
from biotite.structure.io import pdbx

from pkabench.sabdab_units import prepare
from pkabench.sabdab_labels import source_evidence,export_states,qualify_labels
from test_sabdab_units import cif,row


def fixture(tmp_path, trailing_crop=False, nonpolymer='ALA', published_crop=False, internal_gap=False, antigen_tail=False):
    file=cif()
    file.block['struct_asym']=pdbx.CIFCategory(dict(id=list('ABCDZ'),entity_id=['1','2','3','4','5']))
    if nonpolymer!='ALA':
        cat=file.block['atom_site'];mask=cat['label_asym_id'].as_array(str)=='Z'
        for name in ['label_comp_id','auth_comp_id']:
            if name in cat:
                values=cat[name].as_array(str);values=np.asarray(values,dtype='U8');values[mask]=nonpolymer
                cat[name]=pdbx.CIFColumn(values)
    if trailing_crop or internal_gap or antigen_tail:
        file.block['entity_poly']['pdbx_seq_one_letter_code_can']=pdbx.CIFColumn([
            'AAAA' if trailing_crop or internal_gap else 'AA','AA','AAAA' if antigen_tail else 'AA','AA'])
    if internal_gap:
        cat=file.block['atom_site'];mask=(cat['label_asym_id'].as_array(str)=='A') & (cat['label_seq_id'].as_array(int)==2)
        for field in ['auth_seq_id','label_seq_id']:
            values=cat[field].as_array(int);values[mask]=3;cat[field]=pdbx.CIFColumn(values)
    declared=row()
    if published_crop:
        for prefix in ['VH','VL']:
            last=3 if prefix=='VH' and internal_gap else 2
            declared.update({prefix+'_numbering_list':[['A',[1,' ']],['A',[last,' ']]],
                prefix+'_numerable_seq':'AA',prefix+'_imgt_keep_interval':[[1,' '],[last,' ']]})
    stream=io.StringIO();file.write(stream);raw=stream.getvalue().encode()
    atoms,topology,side,meta=prepare(file,declared)
    meta['source_cif_sha256']=hashlib.sha256(raw).hexdigest()
    with gzip.open(tmp_path/'source.cif.gz','wb') as f:f.write(raw)
    evidence,real,known,res=source_evidence(dict(entry=str(tmp_path)),topology,meta)
    return topology,meta,evidence,real,known


def test_full_sequence_crop_is_not_a_true_terminus(tmp_path):
    topology,meta,evidence,real,known=fixture(tmp_path,trailing_crop=True)
    assert ('A',2,'','CTERM') in known
    assert ('A',2,'','CTERM') not in real
    assert ('A',1,'','NTERM') in real
    assert any(d.get('chain')=='A' and d['kind']=='terminal_gap' and d['length']==2 for d in evidence['defects'])
    assert any(c['cls']=='untyped_nonprotein' for c in evidence['components'])


def test_joint_ab_export_contains_both_physical_chains_and_inverse_map(tmp_path):
    topology,meta,evidence,real,known=fixture(tmp_path)
    mappings=export_states(topology,meta,real,tmp_path)
    a={tuple(r[2:]) for r in mappings['A']['residues']}
    b={tuple(r[2:]) for r in mappings['B']['residues']}
    ab={tuple(r[2:]) for r in mappings['AB']['residues']}
    assert {r[0] for r in a}=={'A','B'}
    assert {r[0] for r in b}=={'C','D'}
    assert not a&b and a|b==ab
    assert len(ab)==topology.n_residues


def test_missing_paired_teacher_cannot_enable_supervision():
    site=dict(chain='A',resnum=1,icode='',group='ASP',partner='A',train_mask=True,eval_mask=True,interface=True)
    labels=dict(errors=[],pkai=dict(AB=[['A',1,'','ASP',4.]],A=[],B=[]))
    counts=qualify_labels([site],labels)
    assert not site['train_mask'] and not site['eval_mask'] and counts['paired_sites']==0


def test_missing_component_metadata_uses_pinned_ccd(tmp_path):
    _,_,evidence,_,_=fixture(tmp_path,nonpolymer='NAG')
    assert evidence['component_dictionary']['sha256']
    assert 'SACCHARIDE' in evidence['component_dictionary']['resolved_types']['NAG'].upper()
    assert any(c['name']=='NAG' and c['cls']=='glycan' for c in evidence['components'])


def test_published_crop_removes_constant_tail_without_creating_true_terminus(tmp_path):
    topology,meta,evidence,real,known=fixture(tmp_path,trailing_crop=True,published_crop=True)
    assert meta['antibody_scope']['status']=='published_boundaries_verified'
    assert not any(d.get('chain')=='A' for d in evidence['defects'] if d['kind'].endswith('_gap'))
    assert any(d['chain']=='A' and d['start']==3 and d['end']==4 for d in evidence['intentional_omissions'])
    assert ('A',2,'','CTERM') in known and ('A',2,'','CTERM') not in real
    mappings=export_states(topology,meta,real,tmp_path)
    inverse={(c,n):(oc,on,oi) for c,n,oc,on,oi in mappings['AB']['residues']}
    assert not any(kind=='CTR' and inverse[oc,on]==('A',2,'')
        for _,_,kind,oc,on in mappings['AB']['termini'])


def test_unresolved_residue_inside_published_domain_keeps_gap_mask(tmp_path):
    from pkabench.pkpdb_pilot_clean import gap_context
    from pkabench.long_gap_policy import site_usable
    topology,meta,evidence,real,known=fixture(tmp_path,internal_gap=True,published_crop=True)
    assert any(d.get('chain')=='A' and d['kind']=='internal_gap' and d['start']==2 for d in evidence['defects'])
    residues={(k.chain,k.number,k.insertion):topology.residue(i) for i,k in enumerate(topology.keys)}
    gaps,verified=gap_context(evidence,residues)
    usable,reason=site_usable(topology.backbone[:1,1],True,gaps)
    assert verified and not usable and reason=='near_gap'


def test_antigen_tail_is_not_exempted_by_antibody_crop(tmp_path):
    _,_,evidence,_,_=fixture(tmp_path,trailing_crop=True,published_crop=True,antigen_tail=True)
    assert any(d.get('chain')=='C' and d['kind']=='terminal_gap' and d['length']==2 for d in evidence['defects'])
    assert not any(d['chain']=='C' for d in evidence['intentional_omissions'])


def test_packed_source_payload_uses_the_same_verified_evidence(tmp_path):
    topology,meta,evidence,real,known=fixture(tmp_path,trailing_crop=True,published_crop=True)
    raw=gzip.decompress((tmp_path/'source.cif.gz').read_bytes())
    actual,a,b,_=source_evidence({},topology,meta,source_cif_bytes=raw)
    assert actual==evidence and a==real and b==known
    with pytest.raises(ValueError,match='changed'):
        source_evidence({},topology,meta,source_cif_bytes=raw+b'\n')
