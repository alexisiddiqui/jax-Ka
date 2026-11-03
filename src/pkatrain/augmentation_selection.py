"""Five-mode ANM candidate quotas with deterministic, bounded replacement.

Displacement targets label bins; aligned post-repair CA RMSD selects within
each bin. Geometry rejection never changes the amplitude of a replacement.
"""
import numpy as np
from .augmentation_quality import rmsd

AMPLITUDES = (.5, 1., 2.)
QUOTAS = (1, 1, 2)
POOL_SIZES = (2, 2, 4)
RECIPE = dict(modes=5, amplitudes_A=list(AMPLITUDES), quotas=list(QUOTAS),
              initial_candidates_per_bin=list(POOL_SIZES), max_candidates_per_bin=8,
              reference="zero_coordinates", selection="maximin aligned CA RMSD within each bin, seeded by repaired zero",
              replacements="next saved direction in the same bin, only until quota can be met; maximum eight directions per bin",
              insufficient="retain valid partial quota explicitly; no cross-bin backfill")


def candidate_ids(bin_index, count=8):
    return list(range(1+8*bin_index,1+8*bin_index+count))


def candidate_plan(metrics, *, replace=False):
    """Return generation order, valid bin pools and quota yield, before ranking.

    metrics maps historical candidate IDs to their already computed quality.
    Production can make the same decisions incrementally after each new repair.
    """
    attempted=[];pools=[];counts=[]
    for b,(budget,quota) in enumerate(zip(POOL_SIZES,QUOTAS)):
        initial=candidate_ids(b,budget);attempted.extend(initial)
        valid=[c for c in initial if metrics.get(c,{}).get('geometry_valid',False)]
        if replace:
            for c in candidate_ids(b)[budget:]:
                if len(valid)>=quota:break
                attempted.append(c)
                if metrics.get(c,{}).get('geometry_valid',False):valid.append(c)
        pools.append(valid);counts.append(min(len(valid),quota))
    return dict(attempted=attempted,pools=pools,counts=counts,
                quota_met=tuple(counts)==QUOTAS,selected_count=sum(counts),
                replacements=max(0,len(attempted)-sum(POOL_SIZES)))


def select_quota(plan, coordinates, zero):
    """coordinates contains corresponding CA arrays, not full heavy atoms."""
    selected=[]
    for pool,quota in zip(plan['pools'],QUOTAS):
        chosen=[];distance={c:rmsd(coordinates[c],zero) for c in pool}
        while len(chosen)<min(quota,len(pool)):
            candidate=min((c for c in pool if c not in chosen),key=lambda c:(-distance[c],c))
            chosen.append(candidate)
            for c in pool:
                if c not in chosen:distance[c]=min(distance[c],rmsd(coordinates[c],coordinates[candidate]))
        selected.extend(chosen)
    return selected
