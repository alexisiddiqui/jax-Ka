"""Inspect the released PROPKA interaction graph for cross-chain terms."""
import sys
from pkabench.runtime import require_compute, atomic_json

require_compute()
from propka.run import single
model=single(sys.argv[1],write_pka=False)
terms=[]
for group in model.conformations['AVR'].groups:
    for kind, determinants in group.determinants.items():
        for determinant in determinants:
            other=determinant.group
            if group.atom.chain_id!=other.atom.chain_id and abs(determinant.value)>1e-8:
                terms.append({'site':group.label,'partner_site':other.label,'kind':kind,'value':determinant.value})
atomic_json(sys.argv[2],{'cross_chain_term_count':len(terms),'terms':terms})
