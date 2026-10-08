"""First deposited residue-coherent altloc; alternatives are provenance only."""
from collections import defaultdict
import numpy as np
from biotite.structure.io import pdbx
from .prep import CANONICAL, Rejection
from jaxpropka.geometry import _template


def resolve(cif, selected_chains):
    cat=cif.block['atom_site']; n=cat.row_count
    def field(name,default):
        return cat[name].as_array(str) if name in cat else np.full(n,default)
    model=field('pdbx_PDB_model_num','1'); models=list(dict.fromkeys(model))
    if len(models)!=1:
        raise Rejection('multiple_models','Explicit per-model paired examples required; refusing silent model-1 selection','selection')
    chain=field('label_asym_id',''); seq=field('label_seq_id','?'); auth=field('auth_seq_id','?')
    ins=field('pdbx_PDB_ins_code','?'); comp=field('label_comp_id',''); atom=field('label_atom_id','')
    alt=field('label_alt_id','.'); element=field('type_symbol','').astype(str)
    try: occ=field('occupancy','?').astype(float)
    except ValueError as exc: raise Rejection('ambiguous_occupancy','Missing/non-numeric atom occupancy') from exc
    if not np.isfinite(occ).all() or np.any((occ<0)|(occ>1.001)):
        raise Rejection('ambiguous_occupancy','Nonfinite or out-of-range atom occupancy')
    coords=np.column_stack([field(f'Cartn_{axis}','?').astype(float) for axis in 'xyz'])
    if not np.isfinite(coords).all(): raise Rejection('invalid_coordinates','Nonfinite atom coordinates')
    groups=defaultdict(list)
    for i in range(n): groups[(chain[i],seq[i],auth[i],ins[i])].append(i)
    keep=np.zeros(n,bool); records=[]; defects=[]
    blank={'',' ','?','.'}
    for identity,indices in groups.items():
        ids=np.asarray(indices); positive=ids[occ[ids]>0]
        shared=positive[np.isin(alt[positive],list(blank))]
        alternatives=list(dict.fromkeys(a for a in alt[positive] if a not in blank))
        if len(set(comp[positive]))>1:
            raise Rejection('altloc_identity','Alternate residue identities require explicit chemistry handling')
        score={a:float(occ[positive[alt[positive]==a]].sum()) for a in alternatives}
        chosen=alternatives[0] if alternatives else None
        chosen_ids=np.concatenate([shared,positive[alt[positive]==chosen]]) if chosen else shared
        if len(set(atom[chosen_ids]))!=len(chosen_ids):
            raise Rejection('ambiguous_residue_key','Shared/alternate atom names overlap or duplicate')
        keep[chosen_ids]=True
        if identity[0] not in selected_chains or not len(positive): continue
        heavy=positive[~np.isin(np.char.upper(element[positive]),['H','D'])]
        if not alternatives and not np.any(occ[heavy]<.999): continue
        key=[str(identity[0]),int(identity[2]),'' if identity[3] in blank else str(identity[3])]
        variants=[]
        for a in alternatives or [None]:
            vv=np.concatenate([shared,positive[alt[positive]==a]]) if a else shared
            vv=vv[~np.isin(np.char.upper(element[vv]),['H','D'])]
            if len(set(atom[vv]))!=len(vv): raise Rejection('ambiguous_residue_key','Duplicate atoms in alternate conformer')
            name=str(comp[vv[0]]) if len(vv) else ''
            expected=set(_template(name)[0])-{'OXT'} if name in CANONICAL else set()
            variants.append({'alt_id':a,'occupancy_sum':score.get(a),'atoms':atom[vv].tolist(),
                'occupancies':occ[vv].tolist(),'coordinates':coords[vv].tolist(),
                'source_row_indices':vv.tolist(),'missing_heavy_atoms':sorted(expected-set(atom[vv]))})
        record={'key':key,'label_seq_id':str(identity[1]),'selected_alt_id':chosen,
            'variants':variants,'zero_occupancy_rows_removed':int(sum(occ[ids]==0)),
            'uncertain':len(alternatives)>1 or bool(np.any(occ[heavy]<.999))}
        records.append(record)
    new=pdbx.CIFCategory()
    for name in cat: new[name]=pdbx.CIFColumn(cat[name].as_array(str)[keep])
    new['label_alt_id']=pdbx.CIFColumn(np.full(int(keep.sum()),'.'))
    cif.block['atom_site']=new
    return cif,{'models':models,'selection':'first positive-occupancy alt ID in source order per residue; shared atoms retained; zero occupancy removed',
        'records':records,'defects':defects,'zero_occupancy_rows_removed':int(sum(occ==0)),
        'limits':'Labels are conditional on the selected conformation. Alternatives are provenance only and do not cause masking or extra teacher runs.'}
