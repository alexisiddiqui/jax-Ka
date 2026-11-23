import copy
import pytest
from sabdab_v7_full_rescope import verify_sabdab_antigen


def fixture():
    row=dict(agchains='P/Q',agtypes='PROTEIN/PEPTIDE',agresolvedseqs='DE/K',
        ag_index_list='[["D",[12," "]],["E",[13,"A"]]]/__FULL_CHAIN__')
    meta=dict(chains=[dict(chain='C',observed_sequence='DE'),dict(chain='D',observed_sequence='K')],
        residue_map=[dict(chain='C',number=1,side='B',source_author_chain='P',source_author_number=12,source_author_insertion=''),
                     dict(chain='C',number=2,side='B',source_author_chain='P',source_author_number=13,source_author_insertion='A'),
                     dict(chain='D',number=1,side='B',source_author_chain='Q',source_author_number=1,source_author_insertion='')])
    return row,meta


def test_retained_antigen_proof_preserves_author_insertions_and_joint_selection():
    row,meta=fixture()
    assert verify_sabdab_antigen(row,meta)


@pytest.mark.parametrize('defect',['sequence','indices','missing-partner'])
def test_native_scope_mismatch_cannot_inherit_published_identity(defect):
    row,meta=fixture();meta=copy.deepcopy(meta)
    if defect=='sequence':meta['chains'][0]['observed_sequence']='EE'
    elif defect=='indices':meta['residue_map'][1]['source_author_insertion']=''
    else:meta['residue_map']=meta['residue_map'][:-1]
    with pytest.raises(ValueError):verify_sabdab_antigen(row,meta)
