"""Dense native MaSIF surfaces routed by complete training assembly."""
from pathlib import Path
import json

import numpy as np
from pkabench.runtime import digest
from .augmentation_store import AugmentationStore


class MaSIFStore:
    """Read explicitly versioned surfaces; incomplete assemblies fail closed."""
    def __init__(self,path):
        self.path=Path(path);self.store=AugmentationStore(path)
        try:
            expected=self.store.manifest['files']['coverage.json']
            if digest(self.path/'coverage.json')!=expected:raise ValueError('MaSIF coverage metadata changed')
            coverage=json.loads((self.path/'coverage.json').read_text())
            if coverage['format']!='masif-native-assembly-coverage-v1':raise ValueError('unknown MaSIF assembly coverage format')
            self.parents={}
            for row in coverage['parents']:
                key=(row['dataset'],row['cid'])
                if key in self.parents:raise ValueError('duplicate or overlapping assembly membership')
                if row['status']=='covered' and any(u['id'] not in self.store.by_id for u in row['required_units']):
                    raise ValueError('covered assembly has missing packed surfaces')
                self.parents[key]=row
        except BaseException:self.close();raise

    def assemblies(self,*,split,dataset=None):
        return [row for row in self.parents.values() if row['status']=='covered' and row['split']==split and (dataset is None or row['dataset']==dataset)]

    def load_assembly(self,dataset,cid,*,split=None):
        row=self.parents[dataset,cid]
        if split is not None and row['split']!=split:raise ValueError('requested MaSIF split differs from source membership')
        if row['status']!='covered':raise KeyError('MaSIF assembly is withheld: '+dataset+'/'+cid+'; '+str(row['reasons']))
        units=[];seen=set()
        for spec in row['required_units']:
            record=self.store.raw(spec['id']);a=record['arrays'];m=record['metadata']
            if m['dataset']!=dataset or m['cid']!=cid or m['split']!=row['split']:raise ValueError('MaSIF frame source membership differs')
            ids=a['canonical_residue_ids'];keys=set(ids.tolist())
            if len(keys)!=len(ids) or seen&keys:raise ValueError('overlapping or duplicate canonical MaSIF residue selection')
            seen.update(keys)
            owner=a['owner']
            if owner.shape!=(len(a['vertices']),) or owner.dtype.kind not in 'iu' or owner.min()<0 or owner.max()>=len(ids):
                raise ValueError('invalid MaSIF surface-to-residue map')
            units.append(dict(id=spec['id'],role=spec['role'],state=spec['state'],arrays=a,
                surface_canonical_residue_ids=ids[owner],surface_policy=m['receipt'].get('surface_policy','original-v1')))
        if seen!=set(range(row['canonical_residues'])):raise ValueError('MaSIF surfaces do not partition the full canonical training assembly')
        return dict(dataset=dataset,cid=cid,split=row['split'],group=row['group'],units=units,
                    canonical_residues=row['canonical_residues'])

    def close(self):self.store.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
