"""Structure-defined site loss weights (agreed 2026-10-08; experiments/4_backbone/05_pinder_dataset.md).

Weights depend only on the prepared structure, never on the label, so they redistribute emphasis without biasing the
regression target. Shapes come from PypKa 2.10 labels on the frozen benchmark (audits/shift-by-environment-v1):
- absolute pKa loss: mean |pKa - reference| falls steadily with free-state relative SASA (2.70 at RSA < 0.05,
  0.33 at RSA > 0.6; r = 0.58 with 1 - RSA), so weight = floor + (1 - floor) * (1 - clip(RSA, 0, 1));
- Siamese (bound - free) loss: |dpKa| is 1.13 within 4 A of the partner and roughly halves every 2-3 A beyond it
  (~0 past 15 A), and barely depends on burial near the interface, so weight = 1 up to 4 A, then
  exp(-(d - 4) / 3), with a small floor so distant "no change" pairs still contribute.
Distances are residue-level minimum heavy-atom distances to the partner, as pkabench.annotate.min_partner_distance.
Normalise to mean 1 over the training set (or per structure) when used.
"""
import numpy as np

BURIAL_FLOOR = 0.4
INTERFACE_CONTACT_A = 4.0; INTERFACE_DECAY_A = 3.0; INTERFACE_FLOOR = 0.05


def burial_weight(rsa, floor=BURIAL_FLOOR):
    """Absolute-pKa loss weight from free-state relative SASA: 1 when fully buried, `floor` when fully exposed."""
    return floor + (1. - floor) * (1. - np.clip(np.asarray(rsa, float), 0., 1.))


def interface_weight(distance, contact=INTERFACE_CONTACT_A, decay=INTERFACE_DECAY_A, floor=INTERFACE_FLOOR):
    """Siamese loss weight from residue-level distance to the partner: 1 in contact, exponential decay, bounded below."""
    d = np.asarray(distance, float)
    return np.maximum(floor, np.where(d <= contact, 1., np.exp(-(d - contact) / decay)))
