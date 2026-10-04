import numpy as np
import pytest
from pkabench.frozen_score import measures,aggregate,pairs,complete_linkage,METRICS


def test_group_macro_does_not_weight_large_groups_or_complexes_more():
    def row(g,c,a,b):
        return {'component_id':g,'complex_id':c,'n':len(a),**measures(a,b)}
    data=[row('a','1',[1],[1]),row('a','2',[1]*100,[3]*100),row('b','3',[1],[1])]
    result,groups=aggregate(data)
    assert result['skill']==pytest.approx(0.)
    assert result['mae']==pytest.approx(.5)
    assert result['skill_ci95'] is None


def test_null_skill_and_undefined_metrics():
    result=measures([1,-1],[0,0])
    assert result['skill']==0 and result['sign_accuracy']==0
    assert measures([0,0],[1,1])['skill'] is None
    assert measures([.1,.2],[.1,.2])['sign_accuracy'] is None


def test_frozen_masks_override_legacy_and_split_exclusion():
    s={'complex_id':'c','chain':'A','resnum':1,'icode':'','group':'ASP','partner':'A','supervision_mask':False}
    k=('c','A',1,'','ASP'); rows=[{**s,'state':state,'method':'m','status':'ok','pka':value} for state,value in [('AB',5.),('A',4.)]]
    mask={k:{'training_eligible':True,'evaluation_eligible':False}}
    assert pairs(rows,[s],mask,'m')[k][0]==1
    mask[k]['training_eligible']=False
    assert pairs(rows,[s],mask,'m')=={}
    mask[k]['evaluation_eligible']=True; rows[0]['status']='failed'
    assert pairs(rows,[s],mask,'m')=={}


def test_linkage_refuses_partial_charge_and_uses_current_mask():
    s={'complex_id':'c','chain':'A','resnum':1,'icode':'','group':'ASP','partner':'A','supervision_mask':False}; k=('c','A',1,'','ASP')
    rows=[{**s,'state':state,'status':'ok','curve':[v]*73,'curve_source':'native'} for state,v in [('AB',.75),('A',.25)]]
    mask={k:{'training_eligible':True,'evaluation_eligible':False}}
    result=complete_linkage(rows,[s],mask)
    assert result['status']=='ok' and result['delta_q']==[.5]*73
    assert result['delta_g'][36]==0
    mask[k]['training_eligible']=False
    assert complete_linkage(rows,[s],mask)['delta_g'] is None


def test_bootstrap_resamples_groups_deterministically():
    rows=[{'component_id':str(i),'complex_id':str(i),'n':1,**measures([1],[i/5])} for i in range(6)]
    a,_=aggregate(rows,200); b,_=aggregate(rows,200)
    assert a==b and a['skill_ci95'] is not None
    assert a['skill_ci95'][0]<=a['skill']<=a['skill_ci95'][1]
