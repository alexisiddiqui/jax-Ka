"""Python-3.6 author-container export of real patches, checkpoint and CPU references."""
import json
from pathlib import Path
import time

import numpy as np
import pymesh

import augmentation_masif_bridge as base
import masif_annotation_bridge as dense


def main():
    from masif_modules.read_data_from_surface import compute_ddc
    import tensorflow as tf
    spec=json.loads(Path('/audit/reference-jobs.json').read_text())
    began=time.time();obj=base.model();restore=time.time()-began
    arrays={};metadata=[]
    with obj.graph.as_default():
        for variable in tf.trainable_variables():
            name=variable.name.split(':')[0]
            arrays['weight__'+name]=obj.session.run(variable).astype(np.float32)
    for i,item in enumerate(spec):
        t=time.time()
        with np.load('/audit/'+item['file']) as z:
            vertices=z['vertices'].astype(np.float64);faces=z['faces'];normals=z['normals'].astype(np.float64);raw=z['raw_features']
        mesh=pymesh.form_mesh(vertices,faces)
        for axis,name in enumerate('xyz'):
            mesh.add_attribute('vertex_n'+name);mesh.set_attribute('vertex_n'+name,normals[:,axis])
        chosen=np.random.RandomState(20261115+i).choice(len(vertices),min(256,len(vertices)),replace=False)
        dense.prepare_distances(mesh)
        rho,theta,mask,indices=dense.sparse_polar(mesh,chosen,20261115)
        features=np.zeros((len(chosen),200,5),np.float32)
        for row,center in enumerate(chosen):
            members=indices[row,mask[row].astype(bool)];cp=int(np.flatnonzero(members==center)[0])
            features[row,:len(members),0]=raw[members,0]
            features[row,:len(members),1]=compute_ddc(vertices[members],normals[members],cp,rho[row,:len(members)])
            features[row,:len(members),2:]=raw[members,1:]
        rho=rho.astype(np.float32);theta=theta.astype(np.float32);mask=mask.astype(np.float32)
        prep=time.time()-t;t=time.time();reference=base.infer(obj,features,rho,theta,mask);elapsed=time.time()-t
        arrays['features_'+str(i)]=features;arrays['rho_'+str(i)]=rho;arrays['theta_'+str(i)]=theta
        arrays['mask_'+str(i)]=mask;arrays['reference_'+str(i)]=reference
        metadata.append(dict(id=item['id'],vertices=len(vertices),patches=len(chosen),prepare_s=prep,cpu_inference_s=elapsed))
        print(json.dumps(metadata[-1]),flush=True)
        dense.DISTANCES=None
    np.savez_compressed('/audit/reference-bundle.npz',**arrays)
    base.savejson('/audit/reference-bundle.json',dict(cases=metadata,model_restore_s=restore,
        checkpoint_sha256=obj.checkpoint_sha256,checkpoint_validation=obj.checkpoint_validation,
        bundle_sha256=base.sha('/audit/reference-bundle.npz'),wall_s=time.time()-began,
        inputs='Real native surface patches reconstructed from saved pilot vertices/normals/chemistry; identical float32 inputs for CPU and GPU'))


if __name__=='__main__':main()
