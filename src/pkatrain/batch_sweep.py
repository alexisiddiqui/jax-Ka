"""Register the controlled backbone-only GQT batch-size sweep."""
import json
import os
from pathlib import Path
import shutil
from pkabench.runtime import atomic_json,digest,require_compute


def read(path):return json.loads(Path(path).read_text())


def register(root):
    out=root/'pretraining/gqt-backbone-batch-sweep-v1';out.mkdir(parents=True,exist_ok=True)
    source=root/'pretraining/pkpdb-5k-comparison-v1/gqt-clean'
    protocol=dict(source=str(source),source_manifest_sha256=digest(source/'manifest.json'),
        batches=[4,8,16,32,64],seed=17,epochs=20,learning_rate=.001,
        controls='same cleaned 5k train set, validation set, model, shuffle seed, objective and final-epoch rule',
        interpretation='fixed epochs and learning rate; larger batches intentionally receive fewer optimizer updates',
        input_features='strict backbone-only; 20 A C-alpha graph',precision='full float32; highest matmul precision')
    atomic_json(out/'protocol.json',protocol)
    for batch in protocol['batches']:
        dest=out/f'batch-{batch}';dest.mkdir(exist_ok=True)
        manifest=read(source/'manifest.json');assert 'resume_checkpoint' not in manifest
        manifest['parent']=dict(path=str(source),manifest_sha256=digest(source/'manifest.json'))
        manifest['batch_sweep_protocol_sha256']=digest(out/'protocol.json')
        manifest['config'].update(batch_size=batch,accumulation=1,matmul_precision='highest',
            dropout_rate=0.,context_mask_probability=0.,batch_sweep=batch)
        if (dest/'manifest.json').exists():assert read(dest/'manifest.json')==manifest
        else:
            (dest/'data').symlink_to(source/'data',target_is_directory=True)
            shutil.copy2(source/'preparation.json',dest/'preparation.json')
            atomic_json(dest/'manifest.json',manifest)


if __name__=='__main__':
    require_compute(threads=int(os.environ['SLURM_CPUS_PER_TASK']),allow_comp1400=True)
    register(Path(os.environ['PKABENCH_RUNTIME']))
