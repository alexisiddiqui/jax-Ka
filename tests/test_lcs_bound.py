import os
import random

import pytest

from pkabench.lcs_bound import LCSBound


def reference(a,b):
    row=[0]*(len(b)+1)
    for x in a:
        new=[0]
        for j,y in enumerate(b):new.append(row[j]+1 if x==y else max(row[j+1],new[j]))
        row=new
    return row[-1]


def test_exact_multiword_bound_and_no_false_rejection():
    library=os.environ.get('SABDAB_EXACT_LCS_LIBRARY')
    if not library:pytest.skip('Build the optional library in the compute allocation')
    rng=random.Random(419)
    targets=[''.join(rng.choices('ACDEFGHIKLMNPQRSTVWY',k=n)) for n in [1,3,31,63,64,65,127,128,129,255,256,257]]
    sequences={str(i):s for i,s in enumerate(targets)}
    specs=[dict(sequence_key=str(i),antigen_type='PROTEIN') for i in range(len(targets))]
    bound=LCSBound(library,sequences,specs)
    queries=targets+[''.join(rng.choices('ACDEFGHIKLMNPQRSTVWY',k=n)) for n in [3,65,128,257]]
    for a in queries:
        for b,value in zip(targets,bound.bounds(a)):
            if min(len(a),len(b))/.4<max(len(a),len(b)):
                assert value==-1
            else:assert value==reference(a,b)
    bound.close()
