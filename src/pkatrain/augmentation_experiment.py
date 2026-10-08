"""Register controlled, from-scratch cleaned-cohort augmentation arms."""
import json
import os
from pathlib import Path
import shutil
from pkabench.runtime import atomic_json,digest,require_compute


def read(path):return json.loads(Path(path).read_text())


def register(root, *, sidechains=False):
    name='augmentation-sidechains-v1' if sidechains else 'augmentation-v1'
    out=root/'pretraining'/name;out.mkdir(parents=True,exist_ok=True)
    context=out/'contexts';assert read(context/'verification.json')['passed']
    source=(out/'source') if sidechains else root/'pretraining/pkpdb-5k-comparison-v1/gqt-clean'
    parent=read(source/'manifest.json')
    if sidechains and parent['config'].pop('strict_backbone',None) is not None:
        # Preparation completed before this inherited backbone-only loader flag
        # was removed. This changes metadata only; graph hashes stay identical.
        atomic_json(source/'manifest.json',parent)
    protocol=dict(source=str(source),source_manifest_sha256=digest(source/'manifest.json'),
        context_verification_sha256=digest(context/'verification.json'),seed=17,
        dataset='same cleaned 5k cohort and unaugmented clean validation; no test evaluation',
        input_features='backbone plus native side-chain heavy-atom geometry' if sidechains else 'backbone only',
        gqt_arms=['baseline','dropout','mask','both'],pkai_arms=['baseline','mask'],
        shared_mask_probability=.05,mask_refresh='structure/epoch; repeat visits share mask',
        protected='all clean GQT supervised residue centres',
        dropout='GQT attention/FF residual outputs at 0.1; pKAI native dropout unchanged',
        gqt_selection='fixed epoch 20',pkai_selection='native recipe validation MSE early stopping',
        scope='seed-17 ablation; confirm promising settings with independent seeds later')
    atomic_json(out/'protocol.json',protocol)
    for arm in protocol['gqt_arms']:
        dest=out/f'gqt-{arm}';dest.mkdir(exist_ok=True)
        m=read(source/'manifest.json')
        assert 'resume_checkpoint' not in m
        if sidechains:m['config'].pop('strict_backbone',None)
        m['parent']=dict(path=str(source),manifest_sha256=digest(source/'manifest.json'))
        m['context_path']=str(context);m['context_plan_sha256']=digest(context/'plan.json')
        m['augmentation_protocol_sha256']=digest(out/'protocol.json')
        m['config'].update(batch_size=8,accumulation=1,matmul_precision='highest',
            dropout_rate=.1 if arm in ('dropout','both') else 0.,
            context_mask_probability=.05 if arm in ('mask','both') else 0.,augmentation_arm=arm)
        if (dest/'manifest.json').exists():
            if sidechains and not (dest/'seed-17/run.json').exists():atomic_json(dest/'manifest.json',m)
            else:assert read(dest/'manifest.json')==m
        else:
            (dest/'data').symlink_to(source/'data',target_is_directory=True)
            shutil.copy2(source/'preparation.json',dest/'preparation.json')
            atomic_json(dest/'manifest.json',m)


if __name__=='__main__':
    import sys
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True)
    register(Path(os.environ['PKABENCH_RUNTIME']),sidechains=len(sys.argv)>1 and sys.argv[1]=='sidechains')
