# PINDER as a backbone-only / pKAI-labelled dataset (2026-10-06/07)

Working notes for using PINDER 2024-02 as a large protein–protein source for the
backbone-only stage ([04](04_backbone_only.md)). Audits live under
`$PKABENCH_RUNTIME/audits/pinder-index-v1`, `pinder-prep-v1`, `pinder-prep-v2`
and `pinder-prefilter-v1`. No labels have been generated yet.

## Source

- PINDER index/metadata 2024-02 (`index.parquet` sha256 `ced1936c…`,
  `metadata.parquet` `3e3cfdc2…`), downloaded by the user because the cluster's
  TLS path to `pinderdata.org` is intercepted by a Cloudflare gateway CA.
- 2,319,564 dimers from 119,768 PDB entries; ~49k interface clusters.
- PINDER's `invalid` split marks systems that leak into **PINDER's own** test/val
  sets. That does not apply to our held-out sets, so `invalid` is included and
  filtered by our own leakage rules.

## Component policy

Components never reject; they are stripped and mask nearby supervision
(`mask-all-v1`, `src/pkabench/component_mask_policy.py`; decision log
2026-10-06 in `00_shared.md`). Train/eval radii: ligand (incl. covalent) 15/25,
buffer 15/25, glycan 20/25, exposed ion 25/25, bound/buried metal or metal
complex 30/30 Å. Pair context: chains A/B plus non-protein components within
25 Å, except those attached only to discarded chains. Buried area is recorded,
not gated (small interfaces kept; whether the Siamese loss uses them is open).

## Gap and rebuilt-residue rules (adopted 2026-10-07)

- Training: pKPDB anchor tiers (clean + uncertain); sites on PDB2PQR-rebuilt
  residues excluded; no mask around rebuilt neighbours anywhere (rebuilt side
  chains are acceptable as rough packing).
- Evaluation: as training, plus sites within 15 Å of rebuilt atoms on other
  residues are excluded (distance measured from the rebuilt atom coordinates).
- Component masks as above (training/evaluation radii).
- Long gaps (terminal > 10, internal > 3) remain uncalibrated and excluded by
  the anchor-tier rule; unmodelled termini are the dominant PINDER site loss.

Random-sample scores (4,330 accepted dimers, 587,956 sites, 121,252 interface):

| Rule | Usable sites | Interface | Dimers with ≥1 usable interface site |
|---|---:|---:|---:|
| Training, previous (worst-case gaps + rebuilt-neighbour mask) | 117,534 | 18,475 | 1,225 |
| **Training, adopted** | **165,201** | **29,080** | **1,687** |
| Evaluation, previous | 89,877 | 13,647 | 893 |
| **Evaluation, adopted** | **132,339** | **22,502** | **1,363** |
| Evaluation, CA + 8 Å envelope variant (not adopted) | 128,914 | 21,676 | 1,337 |

Under the adopted training rule, 46% of sites are uncalibrated-gap sites (188k
near termini > 10 residues), 13% are component-masked, 8% near short gaps.

## Prefiltering before preparation

Preparing all 2.32M dimers was stopped: under `mask-all-v1` ~87% of dimers are
accepted, so preparation no longer filters and most cost is redundancy.
`pinder-prefilter-v1/prefilter.py`: pair ≤ 1,500 residues; up to 10 ranked
candidates per interface cluster (X-ray first, resolution, distinct PDB entries
first); RCSB sequences; leakage vs test + validation + reserved + set-2 chains at
70% id / 80% cov (antibody–antigen: antigen chain + CDR checks); prepare the top
5 survivors per cluster, keep the rest as fallbacks.

## Deferred: redundant copies of each interface

Prefiltering drops the remaining copies of each interface — other PDB entries,
conformations and assembly copies of the same interaction. They could be added
later, for clusters of interest or as conformational augmentation. Not decided.

## Prefiltered preparation result (2026-10-07/08)

`pinder-prep-v2/prep_v7.py` on the 109,866 first-pass dimers (`shards_pf`) plus a
fallback round (`build_fb_shards.py`: next 5 survivors for clusters with no
accepted first-pass dimer; 459 of 7,499 such clusters had any, 1,589 dimers, 5
newly accepted). Scored by `score_final.py` (output `score_final.out`).

- Prepared 111,455: accepted 93,293 (62,284 hetero, 28,341 homo, 2,668 Ab/Ag),
  rejected 17,125, chain not found 1,034, crashed 3 (1m4x, 5l35 exceed memory).
  34,909 of 42,378 clusters have an accepted dimer.
- Sites 14.86M, interface 3.74M.

| Rule | Usable sites | Interface | Dimers with ≥1 usable interface site (clusters) | hetero / homo / Ab/Ag dimers |
|---|---:|---:|---:|---|
| Training | 1,802,726 (12.1%) | 445,341 | 25,502 (12,614) | 14,358 / 10,198 / 946 |
| Evaluation | 1,298,980 (8.7%) | 317,599 | 19,250 (10,059) | 11,131 / 7,375 / 744 |

Training-rule losses: uncalibrated gaps 59.2% (mostly unmodelled termini > 10),
near short gaps 12.1%, component masks 10.3%, ineligible 6.2%.

The usable fraction is well below the random sample (28%): one-per-cluster
selection is gappier than the redundancy-weighted sample (EM clean+uncertain
16% vs 38%, X-ray 29% vs 46%; `tier_by_method.py`). Cause not yet verified.
Unmodelled-termini calibration is the largest remaining lever.

## Long-gap calibration for pKAI labels (2026-10-08)

`audits/pinder-longgap-v1/` (`select_refs.py`, `calib.py`, `analyze.py`, `rescore.py`; job 743812). 516 fully
modelled X-ray/EM PINDER references (one per cluster, stratified hetero/homo/Ab-Ag × method), prepared exactly as
`prep_v7.py`, intact and with observed coordinates deleted before preparation: N/C-terminal 10/20/30/50 and internal
5/10/20 residues (5,676 variants, all succeeded). pKAI and pKAI+ on AB/A/B; error = variant − intact. Intact repeats
are bit-identical; free-state error of the untouched partner is exactly 0. Distance = site functional atoms to flank
CA (the stored gap anchor). Criterion as the pKPDB anchor tiers: 5 Å bands, 1,000 whole-reference bootstraps,
upper 95% of p95 error < 0.1 pKa, ≥ 10 references and 20 observations per band.

| Gap | pKAI abs | pKAI Δ | pKAI+ abs | pKAI+ Δ | Rule used (max) |
|---|---:|---:|---:|---:|---:|
| terminal 10 | 30 | 20 | 25 | 15 | 30 |
| terminal 20 | 35 | 25 | 30 | 20 | 35 |
| terminal 30 | 40 | 30 | 35 | 25 | 40 |
| terminal 50 | 45 | 35 | 35 | 30 | 45 |
| internal 5 | 20 | 15 | 15 | 10 | 20 |
| internal 10 | 20 | 20 | 20 | 15 | 20 |
| internal 20 | 25 | 20 | 25 | 15 | 25 |

Rescore of the stored records (lengths round up to the next tested length; longer gaps stay uncalibrated; sites with
no long gap in their record or a short gap < 20 Å stay excluded — approximate until exact tiers are recomputed):

| Rule | Sites | Interface | Dimers with usable interface (clusters) |
|---|---:|---:|---:|
| Training, before | 1,802,726 | 445,341 | 25,502 (12,614) |
| Training, long-gap radii | 2,786,907 | 611,965 | 41,353 (18,563) |
| Evaluation, before | 1,298,980 | 317,599 | 19,250 (10,059) |
| Evaluation, long-gap radii | 1,996,988 | 429,634 | 30,940 (14,844) |

Not adopted yet. Open: the terminal-10 result (30 Å for absolute pKAI) is stricter than the current short-tail rule
(20 Å for lengths 6–10, calibrated on PypKa ΔpKa); ordered segments were deleted, so conservativeness for
disordered tails is untested.
