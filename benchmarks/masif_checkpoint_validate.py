"""Validate a replacement checkpoint on saved real patches in the TF1 container."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

import augmentation_masif_bridge as base
from masif_checkpoint import validate_checkpoint


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.time()
    validation=validate_checkpoint(args.checkpoint)
    legacy='/masif/data/masif_ppi_search/nn_models/sc05/all_feat/model_data/model'
    try:
        validate_checkpoint(legacy)
    except ValueError as exc:
        legacy_error=str(exc)
    else:
        raise ValueError('negative control unexpectedly accepted the damaged checkpoint')
    source_meta=json.loads(args.reference.with_suffix('.json').read_text())
    if base.sha(args.reference)!=source_meta['bundle_sha256']:
        raise ValueError('source real-patch bundle hash mismatch')
    data=np.load(str(args.reference),allow_pickle=False);arrays={};cases=[]
    obj=base.model(checkpoint=args.checkpoint)
    with obj.graph.as_default():
        for variable in tf.trainable_variables():
            name=variable.name.split(':')[0]
            arrays['weight__'+name]=obj.session.run(variable).astype(np.float32)
    changes=[0.]*5;activation_counts=[0]*5;patches=0
    relus=[op.outputs[0] for op in obj.graph.get_operations()
           if op.type=='Relu' and op.outputs[0].shape.as_list()==[None,80]][:5]
    if len(relus)!=5:
        raise ValueError('unexpected feature-branch graph')
    for i,source_case in enumerate(source_meta['cases']):
        feat=data['features_'+str(i)];rho=data['rho_'+str(i)]
        theta=data['theta_'+str(i)];mask=data['mask_'+str(i)]
        t=time.time();reference=base.infer(obj,feat,rho,theta,mask);elapsed=time.time()-t
        patches+=len(feat)
        feed={obj.input_feat:feat,obj.rho_coords:rho[...,None],
              obj.theta_coords:theta[...,None],obj.mask:mask[...,None],obj.keep_prob:1.}
        values=obj.session.run(relus,feed_dict=feed)
        for f,value in enumerate(values):
            if not np.isfinite(value).all():raise ValueError('nonfinite feature activations')
            activation_counts[f]+=int(np.count_nonzero(value))
            changed=feat.copy();changed[:,:,f]*=-1
            delta=float(np.max(np.abs(base.infer(obj,changed,rho,theta,mask)-reference)))
            changes[f]=max(changes[f],delta)
        for key,value in [('features',feat),('rho',rho),('theta',theta),('mask',mask),('reference',reference)]:
            arrays[key+'_'+str(i)]=value
        cases.append(dict(source_case,cpu_inference_s=elapsed))
        print(json.dumps(dict(id=source_case['id'],patches=len(feat),changes=changes)),flush=True)
    if not all(x>1e-5 for x in changes) or not all(activation_counts):
        raise ValueError('a feature branch is inactive or descriptor-insensitive')
    bundle=args.output/'reference-bundle.npz';np.savez_compressed(str(bundle),**arrays)
    metadata=dict(cases=cases,checkpoint_sha256=obj.checkpoint_sha256,
        bundle_sha256=base.sha(bundle),model_restore_s=0.,
        inputs='Unchanged real native patches from the original pilot; new independently validated checkpoint',
        source_bundle_sha256=source_meta['bundle_sha256'],checkpoint_validation=validation,
        model_profile='paper-2019-all-features',wall_s=time.time()-started)
    base.savejson(args.output/'reference-bundle.json',metadata)
    report=dict(passed=True,checkpoint=args.checkpoint,checkpoint_sha256=obj.checkpoint_sha256,
        validation=validation,patches=patches,cases=len(cases),feature_nonzero_activations=activation_counts,
        feature_sign_flip_max_descriptor_change=changes,
        legacy_rejected=True,legacy_error=legacy_error,wall_s=time.time()-started,
        scope='Numerical checkpoint, branch activity and real-patch CPU inference; no training',
        limitation='Original paper weights were trained with the earlier MATLAB geodesic pipeline; current Python preprocessing task accuracy is not established by these numerical checks.')
    base.savejson(args.output/'validation.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
