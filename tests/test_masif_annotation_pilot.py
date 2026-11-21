import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from masif_annotation_pilot import choose_sample,cell,canonical_chain_indices


def test_chain_selection_preserves_indices_when_nonprotein_chains_are_absent():
    chain=np.array([0,0,2,2])
    np.testing.assert_array_equal(canonical_chain_indices(chain,2,4),[2,3])
    with pytest.raises(ValueError,match='no canonical protein residues'):
        canonical_chain_indices(chain,1,4)
    with pytest.raises(ValueError,match='inventory differs'):
        canonical_chain_indices(chain,4,4)
    with pytest.raises(ValueError,match='inventory differs'):
        canonical_chain_indices(np.array([0,4]),0,4)


def test_sample_is_deterministic_population_calibrated_and_stress_excluded():
    units=[]
    for family in ('pkpdb_assembly','pinder_partner','pinder_bound'):
        for n in (100,220,450,900):
            for i in range(7):units.append(dict(id=f'{family}/{n}/{i}',family=family,n=n+i))
    sample=choose_sample(units)
    assert sample==choose_sample(list(reversed(units)))
    assert len({r['id'] for r in sample})==len(sample)
    assert np.isclose(sum(r['weight'] for r in sample),len(units))
    for key in {cell(r) for r in units}:
        assert sum(r['weight'] for r in sample if r['stratum']==key)==sum(cell(r)==key for r in units)
    assert all(r['weight']==0 for r in sample if r['stress'])
