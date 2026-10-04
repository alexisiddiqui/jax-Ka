"""Verify nonprotein inclusion in actual DelPhi arrays before any calibration."""
import json
import sys
from pathlib import Path
from .runtime import require_compute,atomic_json


def main():
    require_compute()
    import numpy as np
    from pypka import Titration
    from pypka.config import Config
    request=json.loads(Path(sys.argv[1]).read_text())
    params=dict(request['config']); params.update(structure=request['pdb'],keep_ions=True,ncpus=1,pH='-2,16',pHstep=.25,save_pdb='delphi-retained.pdb')
    Titration(params,run='preprocess')
    xyz=np.asarray(Config.delphi_params.p_atpos); charges=np.asarray(Config.delphi_params.p_chrgv4); radii=np.asarray(Config.delphi_params.p_rad3)
    ix=np.where(np.linalg.norm(xyz-np.asarray(request['ion_xyz']),axis=1)<.01)[0]
    passed=len(ix)==1 and abs(float(charges[ix[0]])-1)<1e-5 and abs(float(radii[ix[0]])-1.097)<1e-4
    result={'passed':bool(passed),'matches':len(ix),'charges':[float(charges[i]) for i in ix],'radii_A':[float(radii[i]) for i in ix],
        'scope':request['scope'],'pka_calculations_run':False}
    atomic_json('result.json',result)
    if not passed: raise ValueError('Sodium did not reach DelPhi with expected charge/radius')


if __name__=='__main__': main()
