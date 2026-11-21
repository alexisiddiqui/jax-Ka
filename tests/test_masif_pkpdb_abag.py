import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from masif_pkpdb_abag import partition_chains,contact_components,sequence_key
from masif_annotation_pilot import canonical_partner_indices


def test_heavy_and_light_are_one_partner_antigen_is_separate():
    sequences=[dict(chain='A',sequence='light',description='antibody light chain'),
        dict(chain='B',sequence='heavy',description='antibody heavy chain'),
        dict(chain='C',sequence='antigen',description='peptide antigen')]
    numbered={sequence_key('light'):[dict(chain_type='K',cdr3=[[105,'','Q']])],
              sequence_key('heavy'):[dict(chain_type='H',cdr3=[[105,'','R']])]}
    ab,ag,evidence=partition_chains(sequences,numbered)
    assert ab==[0,1] and ag==[2]
    assert [e['role'] for e in evidence]==['Ab','Ab','Ag']
    chain=np.array([0,0,1,1,2,2])
    np.testing.assert_array_equal(canonical_partner_indices(chain,ab,3),[0,1,2,3])
    np.testing.assert_array_equal(canonical_partner_indices(chain,ag,3),[4,5])


def test_unresolved_antibody_chain_cannot_be_mislabelled_as_antigen():
    with pytest.raises(ValueError,match='unresolved antibody'):
        partition_chains([dict(chain='L',sequence='constant',description='immunoglobulin light chain')],{})


def test_separate_copies_form_separate_contact_components():
    # Two complexes; each consists of adjacent H, L and antigen chains.
    atoms=[];chain=[]
    for i,x in enumerate([0,2,4,100,102,104]):
        atoms.append(dict(N=np.array([x,0.,0]),CA=np.array([x,1.,0]),C=np.array([x,2.,0])))
        chain.append(i)
    groups,edges=contact_components(atoms,np.array(chain),6)
    assert groups==[[0,1,2],[3,4,5]]
    assert all(set(e['chain_indices'])<=set(groups[0]) or set(e['chain_indices'])<=set(groups[1]) for e in edges)


def test_partner_mapping_preserves_chain_indices_and_rejects_empty_chain():
    chain=np.array([0,0,2,2,3])
    np.testing.assert_array_equal(canonical_partner_indices(chain,[0,2],4),[0,1,2,3])
    for invalid in ([1],[0,0],[]):
        with pytest.raises(ValueError):canonical_partner_indices(chain,invalid,4)
