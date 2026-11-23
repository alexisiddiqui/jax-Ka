import json

from sabdab_v7_screen import merge_numbering


def inputs(tmp_path,old,new):
    (tmp_path/'numbering-cached.json').write_text(json.dumps(dict(numbered={'seq':[old]},attempted=['seq'])))
    (tmp_path/'numbering-new.json').write_text(json.dumps(dict(numbered={'seq':[new]},version='test')))


def test_cdr3_only_evidence_can_be_enriched_without_claiming_absent_cdrs_changed(tmp_path):
    old=dict(chain_type='H',start=0,end=110,cdr3=[[105,'','A'],[106,'','R']])
    new=dict(chain_type='H',start=0,end=110,numbering=[[27,'','G'],[56,'','S'],[105,'','A'],[106,'','R']])
    inputs(tmp_path,old,new)
    numbered,_,mismatches=merge_numbering(tmp_path)
    assert mismatches==[]
    assert numbered['seq'][0]['numbering']==new['numbering']


def test_a_real_change_to_existing_cdr_evidence_is_not_silently_accepted(tmp_path):
    old=dict(chain_type='H',start=0,end=110,cdr3=[[105,'','A'],[106,'','R']])
    new=dict(chain_type='H',start=0,end=110,numbering=[[27,'','G'],[56,'','S'],[105,'','A'],[106,'','K']])
    inputs(tmp_path,old,new)
    numbered,_,mismatches=merge_numbering(tmp_path)
    assert mismatches==[dict(sequence_key='seq',reason='cached_cdr3_not_reproduced')]
    assert numbered['seq']==[old]
