"""Wyman linkage on complete, paired protonated-fraction curves."""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from .schema import PH, key


def integrate(delta_q):
    q=np.asarray(delta_q,dtype=float)
    if q.shape!=(73,) or not np.isfinite(q).all(): raise ValueError("complete shared-grid delta Q required")
    area=cumulative_trapezoid(q,PH,initial=0)
    return 1.364*(area-area[np.flatnonzero(PH==7)[0]])


def linkage(rows, sites):
    if any(s.get('supervision_mask') is False for s in sites):
        return {'status':'masked_uncertain_charge_coverage','delta_q':None,'delta_g':None}
    expected={key(s):s for s in sites if not s["is_break_terminus"]}
    indexed={(key(r),r["state"]):r for r in rows}
    delta=np.zeros(73)
    for k,s in expected.items():
        ab=indexed.get((k,"AB")); free=indexed.get((k,s["partner"]))
        if any(r is None or r["curve"] is None or r["status"] in {"failed","not_reported"} for r in (ab,free)):
            return {"status":"incomplete_charge_coverage","delta_q":None,"delta_g":None}
        # Fixed deprotonated charges cancel site-by-site between matched states.
        delta+=np.asarray(ab["curve"])-np.asarray(free["curve"])
    return {"status":"ok","delta_q":delta.tolist(),"delta_g":integrate(delta).tolist()}
