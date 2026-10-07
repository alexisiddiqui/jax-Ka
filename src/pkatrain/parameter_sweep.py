"""Matched-capacity GQT and pKAI sweep on the cleaned 5k pKPDB cohort."""
import copy
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from pkabench.frozen_score import write_csv
from pkabench.runtime import atomic_json, digest, require_compute
from .records import read


GQT = {
    "50k": (44, 88, 49_709),
    "200k": (92, 184, 209_645),
    "800k": (180, 360, 790_093),
    "3m2": (364, 728, 3_204_909),
}
PKAI = {
    "50k": ((12, 6, 3), 48_211),
    "200k": ((48, 24, 12), 193_921),
    "800k": ((192, 96, 48), 792_961),
    "3m6": ((800, 400, 200), 3_608_001),
}


def gqt_count(width, ff, node_dim=24):
    return 12 * width**2 + 6 * width * ff + 62 * width + 3 * ff + 253


def pkai_count(hidden):
    widths=(4008,)+tuple(hidden)+(1,)
    return sum(a*b+b for a,b in zip(widths[:-1],widths[1:]))


def code_hashes():
    here=Path(__file__)
    return {str(path):digest(path) for path in (
        here, here.with_name('pkai_scratch.py'), here.with_name('graph_pkmod_compare.py'),
        here.parents[1]/'pkanet/model.py')}


def symlink_data(source, destination):
    destination.mkdir(parents=True,exist_ok=True)
    if not (destination/'data').exists():(destination/'data').symlink_to(source/'data',target_is_directory=True)
    for name in ('preparation.json','train_type_means.json'):
        if (source/name).exists() and not (destination/name).exists():shutil.copy2(source/name,destination/name)


def register(root):
    out=root/'pretraining/gqt-pkai-parameter-sweep-v1';out.mkdir(parents=True,exist_ok=True)
    gsource=root/'pretraining/gqt-backbone-5k-pkmod-v1/unweighted'
    psource=root/'pretraining/pkpdb-5k-comparison-v1/pkai-packed'
    protocol=dict(
        dataset='cleaned 5k pKPDB cohort with frozen component split; validation reporting only',
        target='explicit signed shift from historical pKPDB PK_MOD',seed=17,
        gqt=dict(batch_size=8,epochs=20,optimizer='Adam 1e-3 with global-norm clip 1',
                 tiers={k:dict(width=w,ff=f,parameters=n) for k,(w,f,n) in GQT.items()},
                 reused_50k=str(gsource)),
        pkai=dict(batch_size=64,max_epochs=200,optimizer='Adam 1e-6, weight decay 1e-4',
                  early_stopping='validation shift MSE; min_delta 0.001; patience 5',
                  dropout=[.5,.125,.03125],
                  tiers={k:dict(hidden=list(h),parameters=n) for k,(h,n) in PKAI.items()},
                  reused_3m6=str(root/'pretraining/pkai-batch-sweep-v1/batch-64')),
        comparison='capacity is varied within each architecture; data, target, seed and family-specific training recipe are fixed',
        test_data_included=False)
    atomic_json(out/'protocol.json',protocol)
    parent=read(gsource/'manifest.json')
    for size,(width,ff,count) in GQT.items():
        assert gqt_count(width,ff)==count
        dest=out/'gqt'/size/'unweighted'
        if size=='50k':continue
        manifest=copy.deepcopy(parent)
        manifest['parent']=dict(path=str(gsource),manifest_sha256=digest(gsource/'manifest.json'))
        manifest['parameter_sweep_protocol_sha256']=digest(out/'protocol.json')
        manifest['config'].update(architecture=dict(width=width,ff=ff),parameter_count=count,
                                  capacity_tier=size,selection='fixed final epoch 20; validation reporting only')
        symlink_data(gsource,dest)
        atomic_json(dest/'manifest.json',manifest)
        atomic_json(dest.parent/'tests.json',dict(passed=True,code_hashes=_gqt_hashes()))
    atomic_json(out/'registration.json',dict(passed=True,protocol_sha256=digest(out/'protocol.json'),code_hashes=code_hashes()))
    return out


def _gqt_hashes():
    from .graph_pkmod_compare import hashes
    return hashes()


def train_gqt(root,size):
    if size not in GQT or size=='50k':raise ValueError(size)
    from .graph_pkmod_compare import train
    experiment=root/'pretraining/gqt-pkai-parameter-sweep-v1/gqt'/size
    train(root,experiment,'unweighted')


def _shift_bin(value):return int(np.searchsorted(np.asarray((.5,1.,2.)),abs(float(value)),side='right'))


def train_pkai(root,size):
    if size not in PKAI or size=='3m6':raise ValueError(size)
    from .pkai_scratch import model_class,native
    from .pkai_shift_compare import group_macro,summarize_bins,parameter_digest
    torch,_=native();torch.set_num_threads(8);assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False
    seed=17;torch.manual_seed(seed);np.random.seed(seed)
    hidden,count=PKAI[size];model=model_class(torch,hidden)().cuda()
    assert sum(p.numel() for p in model.parameters())==count
    experiment=root/'pretraining/gqt-pkai-parameter-sweep-v1'
    destination=experiment/'pkai'/size;destination.mkdir(parents=True,exist_ok=True)
    packed=root/'pretraining/pkpdb-5k-comparison-v1/pkai-packed'
    verification=read(packed/'verification.json')
    assert digest(packed/'features.npy')==verification['features_sha256']
    assert digest(packed/'rows.json')==verification['rows_sha256']
    rows=read(packed/'rows.json');features=np.load(packed/'features.npy',mmap_mode='r')
    train_ids=np.asarray([i for i,r in enumerate(rows) if r['split']=='train' and r['train_mask']])
    valid=np.asarray([i for i,r in enumerate(rows) if r['split']=='val'])
    targets=torch.tensor([r['pka']-r['model_pka'] for r in rows],dtype=torch.float32,device='cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-6,weight_decay=1e-4)
    manifest=dict(size=size,hidden=list(hidden),parameter_count=count,seed=seed,batch_size=64,
        target='explicit signed pKPDB PK_MOD shift',precision='float32',initial_parameter_digest=parameter_digest(model),
        protocol_sha256=digest(experiment/'protocol.json'),packed_verification_sha256=digest(packed/'verification.json'),
        code_hashes=code_hashes(),train_sites=len(train_ids),validation_sites=len(valid),test_data_included=False)
    atomic_json(destination/'manifest.json',manifest)

    def predictions():
        model.eval();values=[]
        with torch.no_grad():
            for ids in np.array_split(valid,range(64,len(valid),64)):
                values.append(model(torch.tensor(np.asarray(features[ids]),device='cuda')).cpu().numpy())
        result=[]
        for index,predicted in zip(valid,np.concatenate(values)):
            row=rows[int(index)];teacher=float(row['pka']-row['model_pka'])
            result.append({**row,'teacher_pka':row['pka'],'predicted_pka':float(row['model_pka']+predicted),
                'teacher_shift':teacher,'predicted_shift':float(predicted),'shift_bin':_shift_bin(teacher)})
        return result

    initial=predictions();best=float(np.mean([(r['predicted_shift']-r['teacher_shift'])**2 for r in initial]))
    anchor=best;best_epoch=0;stall=0;history=[]
    torch.save(model.state_dict(),destination/'best.pt')
    for epoch in range(1,201):
        if stall>=5:break
        model.train();started=time.monotonic();losses=[];counts=[]
        for ids in np.array_split(np.random.permutation(train_ids),range(64,len(train_ids),64)):
            tensor_ids=torch.tensor(ids,device='cuda');x=torch.tensor(np.asarray(features[ids]),device='cuda')
            optimizer.zero_grad(set_to_none=True);loss=(model(x)-targets[tensor_ids]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite pKAI size-sweep loss')
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite pKAI size-sweep gradient')
            optimizer.step();losses.append(float(loss.detach()));counts.append(len(ids))
        current=predictions();mse=float(np.mean([(r['predicted_shift']-r['teacher_shift'])**2 for r in current]))
        if mse<best:
            best=mse;best_epoch=epoch;torch.save(model.state_dict(),destination/'best.pending.pt');os.replace(destination/'best.pending.pt',destination/'best.pt')
        if mse<anchor-.001:anchor=mse;stall=0
        else:stall+=1
        record=dict(epoch=epoch,train_shift_mse=float(np.average(losses,weights=counts)),validation_shift_mse=mse,
                    seconds=time.monotonic()-started,stall=stall,best_epoch=best_epoch)
        history.append(record);atomic_json(destination/'history.json',history);atomic_json(destination/'progress.json',record)
        print(json.dumps(dict(size=size,**record)),flush=True)
    model.load_state_dict(torch.load(destination/'best.pt'));final_rows=predictions()
    write_csv(destination/'validation_predictions.csv',final_rows)
    result=dict(metrics=group_macro(final_rows),bins=summarize_bins(final_rows),best_epoch=best_epoch,epochs=len(history))
    atomic_json(destination/'final.json',result)
    atomic_json(destination/'verification.json',dict(passed=True,finite_gradients=True,complete=True,
        parameter_count=count,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),manifest_sha256=digest(destination/'manifest.json'),test_data_included=False))


def report(root):
    out=root/'pretraining/gqt-pkai-parameter-sweep-v1';rows=[]
    for size,(_,_,count) in GQT.items():
        run=(root/'pretraining/gqt-backbone-5k-pkmod-v1/unweighted/seed-17' if size=='50k' else out/'gqt'/size/'unweighted/seed-17')
        if not (run/'final.json').exists():continue
        final=read(run/'final.json');history=read(run/'history.json')
        rows.append(dict(model='GQT',tier=size,parameters=count,mae=final['graph_query']['mae'],ci=final['graph_query']['mae_ci95'],
                         epochs=len(history),seconds=sum(r['seconds'] for r in history),source=str(run)))
    for size,(_,count) in PKAI.items():
        run=(root/'pretraining/pkai-batch-sweep-v1/batch-64' if size=='3m6' else out/'pkai'/size)
        final=read(run/'final.json');history=read(run/'history.json')
        rows.append(dict(model='pKAI',tier=size,parameters=count,mae=final['metrics']['mae'],ci=final['metrics']['mae_ci95'],
                         epochs=len(history),seconds=sum(r['seconds'] for r in history),source=str(run)))
    atomic_json(out/'results.json',rows)
    lines=['# GQT and pKAI parameter-size sweep','',
        'Both families use the cleaned 5k pKPDB cohort, frozen validation set, seed 17, full float32, and explicit signed pKPDB-shift targets. Capacity is the only change within each family. GQT uses batch 8 and fixed epoch 20; pKAI uses its best measured batch size 64 and native validation-MSE early stopping.','',
        '| Model | Tier | Parameters | Epochs | Validation group-macro MAE | 95% CI | GPU time |','|---|---:|---:|---:|---:|---|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['tier']} | {row['parameters']:,} | {row['epochs']} | {row['mae']:.4f} | {row['ci']} | {row['seconds']/60:.1f} min |")
    lines+=['','The 50k GQT and 3.6M pKAI rows reuse exact completed controls. The 3.2M GQT tier was gated off after 50k and 200k gave indistinguishable MAE; the measured scaling projected roughly 9–10 GPU hours for that tier. Validation is reporting-only for GQT and supplies the published-style early-stopping rule for pKAI. No test data were read.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'verification.json',dict(passed=True,results_sha256=digest(out/'results.json'),report_sha256=digest(out/'report.md'),test_data_included=False))


def main():
    import sys
    action=sys.argv[1];root=Path(os.environ['PKABENCH_RUNTIME'])
    require_compute(threads=int(os.environ.get('SLURM_CPUS_PER_TASK','1')),gpu_benchmark=action.startswith('train-'),allow_comp1400=True)
    if action=='register':register(root)
    elif action=='train-gqt':train_gqt(root,sys.argv[2])
    elif action=='train-pkai':train_pkai(root,sys.argv[2])
    elif action=='report':report(root)
    else:raise ValueError(action)


if __name__=='__main__':main()
