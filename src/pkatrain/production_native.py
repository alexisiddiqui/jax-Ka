"""Explicit immutable-base graph patches and role-safe augmentation receipts."""
from pathlib import Path

from pkabench.runtime import digest
from .production_graphs import ProductionStore,FIELDS,DIMS


def declared_store_kind(manifest,store):
    kind=store.metadata.get('data_kind')
    if kind is not None:return kind
    # Only registered legacy native catalogues can inherit their documented
    # tensor kind, with the actual metadata pinned by the new manifest.
    known={'pkpdb':'site_only','pinder':'paired'}
    dataset=manifest['dataset'];pin=manifest.get('legacy_store_metadata_sha256')
    if dataset in known and pin and digest(store.path/'metadata.json')==pin:
        if store.metadata.get('dataset')!=dataset or set(store.fields)!=set(FIELDS[dataset]):
            raise ValueError('Legacy native graph field/source contract differs')
        return known[dataset]
    return None


class PatchedStore:
    """Manifest-pinned replacement records; untouched base arrays stay in place."""
    def __init__(self,base,references,rows,dataset):
        self.base=base;self.patches=[];self.routes={};self.path=base.path
        self.metadata=base.metadata;self.fields=base.fields;self.ids=base.ids;self.by_id=base.by_id
        try:
            for reference in references:
                patch=ProductionStore(reference['store']);self.patches.append(patch)
                if digest(patch.path/'verification.json')!=reference['store_verification_sha256']:
                    raise ValueError('Native record patch verification changed')
                if patch.metadata.get('dataset')!=dataset or set(patch.fields)!=set(base.fields):
                    raise ValueError('Native patch changes source/tensor capability')
                if patch.metadata['dtypes']!=base.metadata['dtypes']:raise ValueError('Native patch changes tensor dtypes')
                for cid in patch.by_id:
                    if cid in self.routes or cid not in rows or cid not in base.by_id:raise ValueError('Duplicate or foreign native patch record')
                    dims=patch.dims[patch.by_id[cid]]
                    if any(int(rows[cid][key])!=int(size) for key,size in zip(DIMS,dims)):
                        raise ValueError('Native patch dimensions differ from candidate manifest')
                    self.routes[cid]=patch
        except BaseException:
            self.close();raise

    def raw(self,cid):return self.routes.get(cid,self.base).raw(cid)

    def close(self):
        for patch in self.patches:patch.close()
        self.base.close()


def augmentation_guard(manifest,row):
    """Opt-in v7 donor checks; legacy recipes do not acquire new behaviour."""
    policy=manifest.get('augmentation_role_policy')
    if policy is None:return None
    if policy.get('schema_version')!=1:raise ValueError('Unknown augmentation role policy')
    recipient=manifest['dataset']+':'+row['id'];role=row['operational_role']
    def check(label,cid,suffix,record):
        key=cid if suffix is None else cid+'::'+suffix
        route=policy.get('records',{}).get(label,{}).get(key)
        if route is None:return 'unassessed_augmentation_donors'
        if route.get('recipient_unit_id')!=recipient:return 'augmentation_recipient_mismatch'
        if route.get('recipient_context_sha256')!=row.get('native_coordinate_sha256') or not route.get('recipient_context_sha256'):
            return 'augmentation_native_context_mismatch'
        donors=route.get('donor_unit_ids',[])
        if not donors:return 'unassessed_augmentation_donors'
        donor_roles=policy.get('unit_roles',{})
        if any(donor not in donor_roles for donor in donors):return 'unassessed_augmentation_donors'
        if any(donor_roles[donor]!=role for donor in donors):return 'augmentation_donor_role_mismatch'
        return None
    return check
