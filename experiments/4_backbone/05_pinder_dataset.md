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

**Adopted 2026-10-08 (user): the strictest column ("Rule used"), also replacing the 20 Å rule for 6–10 residue tails
with 30 Å.** Chosen because the teacher (pKAI vs pKAI+) is undecided; a single-teacher column can replace it later.
Approximate effect (`rescore_options.py`, `complexes_options.py`):

| Option | Training sites | Interface | Complexes with usable interface site (clusters) |
|---|---:|---:|---:|
| Previous rule | 1,802,726 | 445,341 | 25,502 (12,614) |
| **Adopted: strictest, 6–10 tails at 30 Å** | **2,655,811** | **575,262** | **39,558 (17,835)** |
| pKAI+ absolute | 2,944,625 | 642,213 | 43,471 (19,231) |
| pKAI ΔpKa only | 3,188,362 | 701,572 | 46,097 (20,182) |
| pKAI+ ΔpKa only | 3,476,077 | 769,022 | 48,401 (20,974) |

Ordered segments were deleted, so conservativeness for disordered tails is untested. Exact counts come from the
labelling re-run, which recomputes tiers with these radii.

### Same rule on pKPDB (2026-10-08)

`audits/pkpdb-longgap-test-v1/` (copy of the mask-all-v1 threshold test with per-site gap records; job 744149,
`results_clean/`, `score.py`, `score.out`). Exact here (records keep calibrated flag, flank-CA distance and envelope
clearance). 19,277 entries, 2,121,580 mapped sites; the current-rule count reproduces the threshold test exactly
(290,778). pKPDB labels are PypKa, so this measures retention only.

| Rule | Entries with ≥ 1 site | Training sites |
|---|---:|---:|
| Current (mask-all-v1 + anchor tiers) | 9,312 | 290,778 (13.7%) |
| **Adopted strictest, 6–10 tails at 30 Å** | **13,753** | **382,352 (18.0%)** |
| Strictest, 6–10 tails left at 20 Å | 14,100 | 412,282 (19.4%) |
| pKAI+ absolute (tightened) | 14,568 | 427,925 (20.2%) |
| pKAI+ ΔpKa | 15,318 | 518,429 (24.4%) |

An earlier attempt (`results/`) is corrupt: a second, unrelated submission of the same array (744027) appended to the
same files concurrently. Ignore it.

## Sequence overlap: PINDER, pKPDB, held-out (2026-10-08)

`audits/seq-overlap-v1/` (`extract_seqs.py` full canonical `entity_poly` sequences from the 121,294 downloaded pKPDB
mmCIFs, verified on 2,000 files to include unmodelled residues; `overlap.py`; jobs 744215/744216). MMseqs2 ≥ 70%
identity, ≥ 80% coverage of both. Held-out = `pinder-prefilter-v1/ref_heldout_all.fasta` (5,775 chains).

- PINDER (93,293 accepted dimers, 46,136 unique chain sequences) vs pKPDB (121,283 entries, 80,276 unique
  sequences): 57.1% of PINDER chain sequences match pKPDB (21,798 identical); 47.6% of dimers have ≥ 1 matching
  chain, 34.7% both; 56.2% of PINDER clusters. 51.1% of pKPDB entries have a chain matching PINDER.
- pKPDB vs held-out: 20,302 of all entries (16.7%) match at 70%; the threshold-test slice only 211 (1.1%) because the
  5k pilot screened at ≥ 90% identity / 80% of the shorter sequence. PINDER was screened at 70%, so the two
  pretraining sources currently use different leakage thresholds.

**Adopted 2026-10-08 (user): pKPDB uses the same 70% / 80%-both held-out rule as PINDER** (decision log in
`00_shared.md`). `seq-overlap-v1/exclusions.py` writes `pkpdb_heldout_exclusions_70.tsv`: 20,302 entries (identity
70–80%: 854; 80–90%: 1,512; 90–95%: 1,504; ≥ 95%: 16,432). Slice recount (`pkpdb-longgap-test-v1/score.py` with
`EXCLUDE_70=1`, `score_excl70.out`): 19,066 entries; current rule 9,205 entries / 286,913 sites; adopted long-gap rule
13,583 entries / 377,067 sites. Not yet wired into `pkpdb_mask_all.py` (regeneration still pending).

## Labelled PINDER dataset and loss weights (2026-10-08)

`pretraining/pinder-pkai-v1/` (`audits/pinder-label-v1`: `label_prep.py` → `label_pkai.py` → `label_env.py`).
All 93,293 accepted dimers re-prepared (content hashes match the prep run), structures kept (AB/A/B teacher input,
student_AB), pKAI and pKAI+ on AB/A/B. `summary.json`: training-mask sites 2,692,723, of which 2,221,320 have bound and
own-free labels (interface 481,978); complexes with ≥ 1 labelled usable interface site 37,330 (22,305 hetero, 13,654
homo, 1,371 Ab/Ag) over 16,997 clusters; evaluation mask 1,587,473 labelled sites (interface 338,694).

Unlabelled usable sites (18%) are arginines and termini: pKAI's model covers ASP/GLU/HIS/LYS/CYS/TYR/NTR/CTR, and
its reader only recognises termini named NTR/CTR, which standard PDB input never has. pKPDB has no arginine labels
and PypKa does not titrate arginine (`TITRABLETAUTOMERS`); only PROPKA does (`audits/shift-by-environment-v1/
arg_propka.csv`). Arginine is ~18% of benchmark interface sites, interface termini ~2%.

Loss weights (decision log 2026-10-08, `src/pkabench/site_weights.py`) are stored per site in `sites.json`:
`rsa_free`, `rsa_bound`, `partner_distance_A`, `w_burial` (absolute loss), `w_interface` (Siamese loss). The
pkpdb-5k-v3 equivalents (`rsa`, `w_burial`) are in each entry's `environment.json`.

### Terminus labels for pKAI (2026-10-08/09)

The released pKAI reader never creates terminus sites (`protein.py`: `self.termini = {}  # TODO ... ignored for now`),
although the model has NTR/CTR classes. `audits/pinder-label-v1/pkai_termini.py` converts real chain termini (NTERM/CTERM
sites not in `artificial_terminal_keys`) in pKAI's PDB input: NTR = residue 1's N, CA, C, O (PypKa's NTR site,
G54A7 `NTRtau*.st`), CTR = last residue's O and OXT, each as its own residue entry. Validated on 400 benchmark complexes
(1,156 real termini with PypKa 2.10 labels; `audits/pkai-termini-v1`):

| Free-state MAE vs PypKa (median signed) | NTR = N only | NTR = N, CA, C, O (adopted) |
|---|---:|---:|
| pKAI NTERM | 0.90 (+0.81) | 0.50 (+0.40) |
| pKAI+ NTERM | 0.55 (+0.45) | 0.40 (+0.24) |
| pKAI CTERM | 0.25 (−0.06) | 0.25 (−0.07) |
| pKAI+ CTERM | 0.34 (−0.10) | 0.34 (−0.10) |

Other sites: the adopted conversion moves 5.4% (pKAI) / 3.2% (pKAI+) of non-terminus labels by > 0.1, and those moved
sites agree better with PypKa than without termini (pKAI 0.53 → 0.50, pKAI+ 0.89 → 0.80 MAE). Residual N-terminal
offset remains (+0.40 pKAI, +0.24 pKAI+). PINDER relabelled with termini (job 745763); previous labels kept as
`labels_noterm.json`.
