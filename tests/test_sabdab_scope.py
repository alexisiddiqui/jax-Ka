import copy
import pytest

from pkabench.sabdab_scope import crop_record,retained_intervals,omitted_runs


def declared():
    r=dict(INSTANCE='test',Hchain='Q',Lchain='R')
    for prefix in ['VH','VL']:
        r.update({prefix+'_numbering_list':[['A',[1,' ']],['A',[2,' ']]],
            prefix+'_numerable_seq':'AA',prefix+'_imgt_keep_interval':[[1,' '],[2,' ']]})
    return r


def mapping():
    return [dict(chain='A',source_author_chain=alias,source_author_number=n,
        source_author_insertion='',source_label_position=pos)
        for alias,n,pos in [('Q',1,1),('Q',2,2),('R',1,5),('R',2,6)]]


def test_scfv_linker_is_explicitly_outside_two_verified_domain_intervals():
    residues=mapping();before=copy.deepcopy(residues)
    intervals,proof=retained_intervals(crop_record(declared()),residues,'AAAA',{'A':'A'},'test')
    assert intervals=={'A':[(1,2),(5,6)]}
    assert omitted_runs({1,2,5,6},6)==[[3,4]]
    assert proof['status']=='published_boundaries_verified' and residues==before


@pytest.mark.parametrize('defect',['unobserved-boundary','missing-retained-residue','wrong-sequence','outside-declared-scope'])
def test_unproved_crops_are_rejected_instead_of_using_observed_extrema(defect):
    row=declared();residues=mapping();letters='AAAA'
    if defect=='unobserved-boundary':row['VH_imgt_keep_interval']=[[0,' '],[2,' ']]
    elif defect=='missing-retained-residue':
        row['VH_numbering_list']=[['A',[1,' ']],['A',[3,' ']],['A',[2,' ']]];row['VH_numerable_seq']='AAA'
    elif defect=='wrong-sequence':letters='CAAA'
    elif defect=='outside-declared-scope':row['VH_imgt_keep_interval']=[[1,' '],[1,' ']]
    with pytest.raises(ValueError):
        retained_intervals(crop_record(row),residues,letters,{'A':'A'},'test')
