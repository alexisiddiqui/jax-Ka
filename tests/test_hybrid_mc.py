"""Energy convention and masking gate; run only within a compute allocation."""
import copy
import math
from pkabench.runtime import require_compute
from pkabench.hybrid_mc import changed_energies

def test_energy_contract():
    sites=[]
    for i,eligible in enumerate((True,True,False)):
        sites.append(dict(complex_id='x',chain='A',resnum=i+1,icode='',group=('ASP','LYS','GLU')[i],
            supervision_eligible=eligible,tautomers=['t0','reference']))
    raw=dict(all_sites=['A_ASP_1','A_LYS_2','A_GLU_3'],npossible_states=[2]*3,
        possible_states_g=[[-9.,0.,None],[12.,0.,None],[-10.,0.,None]],
        possible_states_occ=[[1,0,-500],[0,1,-500],[1,0,-500]],
        interactions=[[0,1],[1,0]],interactions_look=[[0,1],[2,3],[4,5]],states_ddG=[[1],[2],[3]])
    before=copy.deepcopy(raw)
    replacements=[dict(site=sites[0],values={'test':[4.]}),dict(site=sites[1],values={'test':[10.]})]
    got=changed_energies(raw,sites,replacements,'test')
    assert math.isclose(got['possible_states_g'][0][0],-4*math.log(10))
    assert math.isclose(got['possible_states_g'][1][0],10*math.log(10))
    assert got['possible_states_g'][2]==raw['possible_states_g'][2]
    assert raw==before
    assert changed_energies(raw,sites,replacements,'teacher')==raw
    for field in raw:
        if field!='possible_states_g': assert got[field]==raw[field]
    assert all(row[1:]==[0.,None] for row in got['possible_states_g'])

def test_group_weighting():
    import pandas as pd
    from pkabench.hybrid_score import group_errors
    rows=[]
    for cid,group,count,error in [('a','one',1,1.),('b','one',9,3.),('c','two',1,10.)]:
        for i in range(count):
            for state in ('AB','A'):
                rows.append(dict(complex_id=cid,component_id=group,chain='A',resnum=i,icode='',group='ASP',state=state,ae=error,se=error**2))
    got=group_errors(pd.DataFrame(rows))
    assert got.loc['one','mae']==2 and got.loc['two','mae']==10
    assert got.mae.mean()==6

if __name__=='__main__':
    require_compute(); test_energy_contract(); test_group_weighting(); print('hybrid MC contract tests passed')
