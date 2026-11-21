"""Python-3.6 CPU check of the legacy checkpoint's nonfinite branch semantics."""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import augmentation_masif_bridge as base

data=np.load('/audit/reference-bundle.npz');obj=base.model(allow_nonfinite=True)
feat=data['features_0'][:2];rho=data['rho_0'][:2];theta=data['theta_0'][:2];mask=data['mask_0'][:2]
feed={obj.input_feat:feat,obj.rho_coords:rho[...,None],obj.theta_coords:theta[...,None],obj.mask:mask[...,None],obj.keep_prob:1.}
activations=[op.outputs[0] for op in obj.graph.get_operations() if op.type=='Relu' and op.outputs[0].shape.as_list()==[None,80]][:5]
values=obj.session.run(activations,feed_dict=feed)
with obj.graph.as_default():
    probe=tf.nn.relu(tf.constant([np.nan,-1.,0.,1.],tf.float32))
    maxprobe=tf.reduce_max(tf.constant([[np.nan,1.],[np.nan,2.]],tf.float32),axis=0)
relu=obj.session.run(probe);maximum=obj.session.run(maxprobe)
result=dict(relu_nan_probe=[float(v) if np.isfinite(v) else None for v in relu],
    relu_finite=np.isfinite(relu).tolist(),maximum_nan_probe=[float(v) if np.isfinite(v) else str(v) for v in maximum],
    feature_activations=[dict(feature=i,finite=bool(np.isfinite(v).all()),nonzero=int(np.count_nonzero(v)),minimum=float(v.min()),maximum=float(v.max())) for i,v in enumerate(values)])
original=base.infer(obj,feat,rho,theta,mask);changed=feat.copy();changed[:,:,0]=.75
result['shape_index_perturbation_max_descriptor_change']=float(np.max(np.abs(base.infer(obj,changed,rho,theta,mask)-original)))
base.savejson('/audit/checkpoint-probe.json',result);print(json.dumps(result),flush=True)
