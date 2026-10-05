"""Extra tautomers must not increase a site's aggregation weight."""
import numpy as np
import pandas as pd
from pkabench.native_intrinsic import group_scores
from pkabench.runtime import require_compute


def test_site_then_complex_then_group():
    require_compute()
    rows = []
    for cid, group, residue, count, error in [
        ('a', 'g1', 1, 1, 1.), ('a', 'g1', 2, 5, 3.),
        ('b', 'g1', 1, 4, 6.), ('c', 'g2', 1, 1, 8.),
    ]:
        for tautomer in range(count):
            rows.append(dict(complex_id=cid, component_id=group, chain='A',
                             resnum=residue, icode='', group='ASP', state='AB',
                             tautomer=str(tautomer), prediction=error, target=0.))
    result = group_scores(pd.DataFrame(rows))
    assert result.loc['g1', 'mae'] == 4.
    assert result.loc['g2', 'mae'] == 8.
    assert result.mae.mean() == 6.
    assert np.isclose(result.loc['g1', 'rmse'], (np.sqrt(5.) + 6.) / 2)
