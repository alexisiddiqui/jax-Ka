from pkabench.dataset_audit import SCENARIOS, diversity_summary


def candidate(cid, accepted=True):
    return {'complex_id':cid, 'scenarios':{m:{'status':'accepted' if accepted else 'rejected'} for m in SCENARIOS}}


def test_similarity_groups_examples_without_deleting_them():
    rows = [candidate('p1'), candidate('p2')]
    pairs = {'p1':['a','b'], 'p2':['c','d']}
    low = diversity_summary(rows,pairs,[('a','c',.4)],.3)['scenarios']['strict']
    high = diversity_summary(rows,pairs,[('a','c',.4)],.5)['scenarios']['strict']
    assert low['retained_pairs'] == high['retained_pairs'] == 2
    assert low['independent_components'] == 1
    assert high['independent_components'] == 2


def test_rejected_bridge_remains_in_presplit_graph():
    rows = [candidate('p1'),candidate('p2'),candidate('bridge',False)]
    pairs = {'p1':['a','b'], 'p2':['c','d'], 'bridge':['b','c']}
    result = diversity_summary(rows,pairs,[],.3)['scenarios']['strict']
    assert result['retained_pairs'] == 2
    assert result['independent_components'] == 1
    assert result['largest_component_pairs'] == 2


def test_homomers_do_not_duplicate_components():
    rows = [candidate('p1'),candidate('p2')]
    result = diversity_summary(rows,{'p1':['a','a'],'p2':['a','a']},[],.3)['scenarios']['strict']
    assert result['retained_pairs'] == 2
    assert result['unique_cluster_pairs'] == 1
    assert result['independent_components'] == 1
