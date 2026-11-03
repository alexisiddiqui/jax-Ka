import numpy as np
from pkatrain.augmentation_selection import candidate_plan,select_quota
from pkatrain.augmentation_quality import rmsd


def test_replacements_stay_in_bin_and_stop_once_quota_is_available():
    metrics={c:dict(geometry_valid=c in (3,9,17,21)) for c in range(1,25)}
    initial=candidate_plan(metrics)
    assert initial['counts']==[0,1,1] and initial['attempted']==[1,2,9,10,17,18,19,20]
    refill=candidate_plan(metrics,replace=True)
    assert refill['counts']==[1,1,2] and refill['quota_met']
    assert refill['attempted']==[1,2,3,9,10,17,18,19,20,21]
    assert refill['replacements']==2


def test_partial_quotas_do_not_borrow_candidates_from_other_amplitudes():
    metrics={c:dict(geometry_valid=c<17) for c in range(1,25)}
    plan=candidate_plan(metrics,replace=True)
    assert plan['counts']==[1,1,0] and not plan['quota_met']
    assert plan['selected_count']==2 and plan['replacements']==4


def test_maximin_selects_distinct_deformations_and_preserves_quotas():
    rng=np.random.default_rng(4);zero=rng.normal(size=(15,3))
    change=rng.normal(size=zero.shape);other=rng.normal(size=zero.shape)
    coordinates={1:zero+.5*change,2:zero+.4*other,9:zero+change,10:zero+.8*other,
                 17:zero+2*change,18:zero+1.99*change,19:zero-2*change,20:zero+.3*other}
    plan=candidate_plan({c:dict(geometry_valid=True) for c in coordinates})
    selected=select_quota(plan,coordinates,zero)
    assert len(selected)==4 and selected[0] in (1,2) and selected[1] in (9,10)
    assert set(selected[2:])=={17,19}
    assert rmsd(coordinates[17],coordinates[19])>rmsd(coordinates[17],coordinates[18])
