"""Long-gap supervision rule for pKAI-teacher pretraining datasets (PINDER pairs, pKPDB rebuild).

Approved 2026-10-08 (decision log, experiments/00_shared.md). Calibrated by deleting observed segments from 516 fully
modelled PINDER references and re-running pKAI/pKAI+ (audits/pinder-longgap-v1): the radius is the strictest over
pKAI/pKAI+ x absolute/paired-delta error, using the pKPDB anchor-tier criterion (5 A bands, whole-reference bootstrap,
upper 95% of p95 error < 0.1 pKa). Distances are from the site's functional atoms to the CA of the observed flank
residue(s), i.e. the gap defect `centres`.

Relative to the anchor tiers (pkpdb_pilot_clean.gap_tier), a site is usable when it is eligible and:
- every calibrated short gap passes its existing anchor radius, and terminal tails of 6-10 residues are also at least
  LONG_GAP_RADII_A[('terminal', 10)] from the flank CA (the 20 A short-tail rule was calibrated on PypKa delta pKa);
- every uncalibrated gap either has >= 20 A envelope clearance (unchanged) or is a long gap (terminal > 10, internal > 3)
  within the tested lengths whose flank-CA distance is at least the radius of the next larger tested length.
Uncalibrated short gaps (incomplete anchor) and longer untested gaps keep excluding nearby sites.
"""
import numpy as np
from .supervision import clearance

LONG_GAP_RADII_A = {('terminal', 10): 30., ('terminal', 20): 35., ('terminal', 30): 40., ('terminal', 50): 45.,
                    ('internal', 5): 20., ('internal', 10): 20., ('internal', 20): 25.}
POLICY = {'version': 'long-gap-v1', 'radii_A': {f'{k}_{n}': v for (k, n), v in LONG_GAP_RADII_A.items()},
          'distance': 'site functional atoms to flank CA', 'envelope_clearance_A': 20.,
          'source': 'audits/pinder-longgap-v1 (516 references, 5,676 deletion variants)'}
TESTED = {'terminal': (10, 20, 30, 50), 'internal': (5, 10, 20)}


def long_gap_radius(kind, length):
    """Radius for a gap, using the next larger tested length; None if the gap is longer than any tested length."""
    larger = [n for n in TESTED[kind] if n >= length]
    return LONG_GAP_RADII_A[(kind, larger[0])] if larger else None


def flank_distance(points, defect):
    centres = np.asarray(defect.get('centres', []), float)
    if not len(points) or not len(centres): return None
    return float(np.linalg.norm(np.asarray(points, float)[:, None] - centres[None], axis=-1).min())


def site_usable(points, eligible, gaps):
    """gaps: pkpdb_pilot_clean.gap_context output [(defect, calibrated radius or None, anchor coords)]. Returns (usable, reason)."""
    if not eligible: return False, 'ineligible'
    for d, radius, anchor in gaps:
        kind = 'internal' if d['kind'] == 'internal_gap' else 'terminal'; length = int(d['length'])
        if radius is not None:
            if float(np.linalg.norm(np.asarray(points)[:, None] - anchor[None, :], axis=-1).min()) < radius: return False, 'near_gap'
            if kind == 'terminal' and 6 <= length <= 10:
                ca = flank_distance(points, d)
                if ca is None or ca < LONG_GAP_RADII_A[('terminal', 10)]: return False, 'near_gap_short_tail_pkai'
            continue
        if clearance(points, d) >= 20: continue
        long_gap = (kind == 'terminal' and length > 10) or (kind == 'internal' and length > 3)
        if not long_gap: return False, 'uncalibrated_short_gap'
        limit = long_gap_radius(kind, length); ca = flank_distance(points, d)
        if limit is None: return False, 'untested_long_gap'
        if ca is None or ca < limit: return False, 'near_long_gap'
    return True, 'usable'
