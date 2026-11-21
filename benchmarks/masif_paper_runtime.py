"""Install and verify the already-tested original-paper MaSIF runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from masif_checkpoint import validate_arrays

PROFILE = 'masif-paper-2019-v1'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def verify(path,checkpoint=None):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if manifest['profile']!=PROFILE:raise ValueError('unknown paper preprocessing profile')
    for relative,expected in manifest['files'].items():
        target=(path/relative).resolve()
        if not target.is_relative_to(path.resolve()) or sha(target)!=expected:
            raise ValueError('paper runtime changed: '+relative)
    if checkpoint is not None:
        for name,suffix in [('model.data-00000-of-00001','.data-00000-of-00001'),('model.index','.index'),('model.meta','.meta')]:
            if sha(str(checkpoint)+suffix)!=manifest['checkpoint_sha256'][name]:
                raise ValueError('checkpoint does not match original-paper preprocessing profile')
    return manifest


def install(verified,output):
    verified=Path(verified);fix=verified.parent/'masif-checkpoint-fix-v1'
    proof=json.loads((verified/'verification.json').read_text())
    if not proof['full_pdb_to_fingerprints_smoke_passed'] or not proof['original_paper_feature_pipeline_verified']:
        raise ValueError('original source has not passed verification')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    try:
        shutil.copytree(verified/'paper-python3',output/'python3',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(verified/'dependencies/pyflann',output/'dependencies/pyflann',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(fix/'paper-source/source/matlab_libs',output/'matlab_libs')
        original=output/'original/source/masif_modules';original.mkdir(parents=True)
        for name in ['read_data_from_matfile.py','extract_features.py','compute_input_feat.py']:
            shutil.copy2(fix/'paper-source/source/masif_modules'/name,original/name)
        source=(output/'matlab_libs/coords_mds.m').read_text()
        if source.count('function success = coords_mds(paths, params)')!=1 or source.count('if toc(tStart) > 250')!=1:
            raise ValueError('unexpected original MATLAB driver')
        # The numerical code and all-center iteration remain unchanged. Only
        # prevent the upstream per-file exit from terminating a multi-unit job.
        (output/'matlab_libs/coords_mds_batch.m').write_text(source.replace(
            'function success = coords_mds(paths, params)','function success = coords_mds_batch(paths, params)').replace(
            'if toc(tStart) > 250','if false % lifetime controlled by the batch driver'))
        bundle=fix/'validated/reference-bundle.npz';metadata=json.loads(bundle.with_suffix('.json').read_text())
        if sha(bundle)!=metadata['bundle_sha256'] or metadata['checkpoint_sha256']!=proof['full_pdb_smoke']['checkpoint_sha256']:
            raise ValueError('validated weight export does not match the paper model')
        with np.load(bundle,allow_pickle=False) as data:
            weights={k[len('weight__'):]:data[k] for k in data.files if k.startswith('weight__')}
        validate_arrays(weights);np.savez(output/'weights.npz',**weights)
        manifest=dict(profile=PROFILE,source_repository='pablogainza/masif_paper',
            source_commit='05d15e6bdcee6dc8eff0047735c2db59edcf70fc',
            checkpoint_sha256=metadata['checkpoint_sha256'],verification=proof['jobs'],
            components='MSMS per expanded-sphere component; complete unit electrostatics',
            compatibility=['Python 2-to-3 syntax','subprocess text','unused debugger shim',
                           'MATLAB batch driver disables process exit; numerical code unchanged'],
            files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    except BaseException:
        shutil.rmtree(output);raise
    return verify(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('command',choices=['install','verify'])
    parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--verified',type=Path)
    args=parser.parse_args()
    result=install(args.verified,args.runtime) if args.command=='install' else verify(args.runtime)
    print(json.dumps(dict(profile=result['profile'],files=len(result['files']),manifest_sha256=sha(args.runtime/'manifest.json'))))
