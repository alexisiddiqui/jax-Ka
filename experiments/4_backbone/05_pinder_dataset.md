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

Final counts with termini (`summary.json`, job 746014; all 93,293 `labels.json` have `"termini": true`):

| | Without termini | With termini |
|---|---:|---:|
| Training-mask sites with bound + own-free labels | 2,221,320 | **2,272,062** |
| …interface | 481,978 | **497,872** |
| Complexes with ≥ 1 labelled usable interface site | 37,330 | **37,603** (22,467 hetero / 13,757 homo / 1,379 Ab/Ag) |
| Clusters | 16,997 | **17,102** |
| Evaluation-mask labelled sites (interface) | 1,587,473 (338,694) | 1,626,125 (350,641) |

Remaining unlabelled usable sites are arginines (no teacher labels them) and termini adjacent to chain gaps.

### Held-out exclusions harmonised with pKPDB (2026-10-09)

PINDER's prefilter applied only the 70% / 80%-both rule against held-out chains (plus antibody CDR/antigen checks);
pKPDB additionally applies the experimental 30% rule, the 90% fragment rule and exact reserved PDB IDs. Those three
rules applied to PINDER (`audits/pinder-exp-leak-v1/exclusions.py`, references `pkpdb-full-v1/references.json`,
sha256 `48225053…`, identical to pkpdb-5k-v2/v3) give `pinder_heldout_exclusions_v1.tsv` (sha256 `91069204…`, copied
into `pretraining/pinder-pkai-v1/` with `exclusions_v1.json`): 5,605 of 93,293 complexes, 2,735 clusters touched
(experimental 30%: 3,785; fragment 90%: 2,746; reserved PDB ID: 50). Labels are unchanged; the list is a filter.

Counts after exclusion (`summary_clean_v1.json`; `summary.json` left unchanged because
`training/ogqt-pinder-factorial-v1` hashes it):

| | Before | After exclusions | Removed |
|---|---:|---:|---:|
| Complexes | 93,293 | 87,688 | 5,605 |
| Training-mask sites with bound + own-free labels | 2,272,062 | **2,143,026** | 129,036 |
| …interface | 497,872 | **471,396** | 26,476 |
| Complexes with ≥ 1 labelled usable interface site | 37,603 | **35,305** | 2,298 |
| …hetero / homo / Ab/Ag | 22,467 / 13,757 / 1,379 | 21,444 / 13,479 / **382** | 1,023 / 278 / 997 |
| Clusters | 17,102 | **15,874** | 1,228 |
| Evaluation-mask labelled sites (interface) | 1,626,125 (350,641) | 1,526,337 (330,798) | 99,788 (19,843) |

Antibody complexes lose 72%: 1,830 of the 1,952 flagged Ab/Ag complexes hit the experimental 30% rule, almost all via
two Fab references (1igc: 1,508 hits on heavy and light chains; 1axt: 308), whose conserved framework and constant
domains exceed 30% / 80% for essentially any Fab. Factorial cohort (`training/ogqt-pinder-factorial-v1/cohort.json`):
438 / 5,000 train and 12 / 400 val complexes are on the list (`audits/pinder-exp-leak-v1/factorial_cohort_flagged.tsv`).

**Antibody path (v2, adopted 2026-10-09).** v1 applied the extra rules whole-chain to antibody chains, bypassing the
prefilter's Ab/Ag path (antigen chain whole-chain, antibody chain by CDRs). The two Fab references (1igc, 1axt) are
experimental-only, so their CDRs were never in the prefilter's held-out CDR set. `audits/pinder-exp-leak-v1/ab_rule.py`
transfers CDRs onto the experimental antibody chains (1igc and 1axt H/L) by alignment to SAbDab-annotated V domains, as
in the prefilter, and re-checks flagged Ab/Ag dimers: antigen chain experimental 30% / fragment 90% whole-chain;
antibody chain concatenated CDRs ≥ 70% (same chain type) vs the experimental CDRs; reserved PDB IDs unchanged. Of 1,952
flagged Ab/Ag dimers, 1,588 are released; 364 stay excluded (334 antigen fragment 90%, 27 antigen experimental 30%,
3 CDR-only), with the antibody-path reason in column `ab_path_rules`. Other dimers, including antibody heavy–light
pairs (a framework interface), keep the whole-chain rules as in the prefilter.

`pinder_heldout_exclusions_v2.tsv` (sha256 `332eb8fc…`, 4,017 complexes, 1,882 clusters touched; `exclusions_v2.json`)
supersedes v1. Counts (`summary_clean_v2.json`, `EXCL_VERSION=v2 summarize_clean.py`):

| | Before | v1 | **v2 (adopted)** |
|---|---:|---:|---:|
| Complexes | 93,293 | 87,688 | **89,276** |
| Training-mask sites with bound + own-free labels | 2,272,062 | 2,143,026 | **2,192,979** |
| …interface | 497,872 | 471,396 | **478,590** |
| Complexes with ≥ 1 labelled usable interface site | 37,603 | 35,305 | **36,134** |
| …hetero / homo / Ab/Ag | 22,467 / 13,757 / 1,379 | 21,444 / 13,479 / 382 | **21,444 / 13,479 / 1,211** |
| Clusters | 17,102 | 15,874 | **16,368** |
| Evaluation-mask labelled sites (interface) | 1,626,125 (350,641) | 1,526,337 (330,798) | **1,565,937 (336,177)** |

Factorial cohort on the v2 list: 250 / 5,000 train and 12 / 400 val (`factorial_cohort_flagged_v2.tsv`).

**pKPDB under the same antibody path (measured 2026-10-09, not applied).** pKPDB rejects a whole entry when any chain
hits, with no antibody path. `audits/pkpdb-ab-path-v1/measure.py` collects each `sequence_overlap` entry's offending
chains from the full build's own three sources (`excluded.json`, the 70% rule re-read from `hits.tsv`,
`pkpdb_heldout_exclusions_70.tsv`) and releases an entry only if every offending chain is an antibody chain whose CDRs
are < 70% identical to both the held-out CDR sets (prefilter) and the experimental antibody CDRs:

| pkpdb-full-v1 `sequence_overlap` entries | 27,716 |
|---|---:|
| Offending chain is not an antibody (stay excluded) | 25,383 |
| Antibody CDRs ≥ 70% to a held-out or experimental antibody (stay excluded) | 1,400 |
| CDRs not transferable (stay excluded) | 73 |
| **Released, entry also has a non-antibody chain** | **450** |
| **Released, antibody-only entry** | **410** |

The two 70% hits whose sequence is absent from the entity table (ubiquitin variants) belong to entries that stay
excluded. At the build's acceptance rate after screening (~70%), the 860 released entries would add roughly 600
structures to the 63,240 accepted. **Applied (user decision 2026-10-09):** `pkpdb_mask_all` subtracts the released entries (`ANTIBODY_PATH`,
`antibody_released`; sha256 and rule in `protocol.json`) and gains `--revise`, which moves the previous protocol and
outputs of a verified build to `revisions/<n>/` and reuses cached entry receipts, so only newly passing entries are
cleaned. pkpdb-full-v1 is revised in place (job 746097, after the first build 746072 and its environment pass 746073),
followed by the environment pass for the new entries (746098).

Result (jobs 746072 → 746073 → 746097 → 746098, all completed): pkpdb-full-v1 now holds **64,000** structures
(previously 63,240), **1,802,108** clean training sites (previously 1,764,340) and 7,367,205 raw mapped sites; 1 pipeline
error (unchanged). Of the 860 released entries, 760 were accepted (388 with a non-antibody chain, 372 antibody-only),
72 had no clean sites, 27 an incomplete backbone and 1 ambiguous occupancy; they add 37,768 clean sites. All 63,240
earlier records are unchanged (graph and sites hashes identical). `sequence_overlap` rejections fall from 27,716 to
26,856. Every record has `environment.json` (rsa, w_burial) aligned with `sites.json`. The previous build is in
`revisions/00/`; `protocol.json` records `revises_protocol_sha256` and `antibody_path_sha256`.

Factorial cohort (decision log 2026-10-09): left as registered, with the caveat above; subsequent training subsets are
drawn from the full pool after the v2 exclusions (`pinder_heldout_exclusions_v2.tsv`).

### Antibody CDR gap fixed (v3, 2026-10-09)

The antibody path compared CDRs only with held-out complexes carrying SAbDab CDR annotations; 116 heavy and 122 light
held-out antibody chains (mostly Fab-internal and unresolved-antibody test complexes) had none, so identical antibodies
passed (e.g. 1u91, 1mcp, 4hbc). `audits/ab-cdr-gap-v1/check.py` transfers CDRs onto every held-out chain that aligns to
a SAbDab-annotated V domain (complete sets: 419 H / 342 L, previously 303 / 220; `heldout_cdr_{H,L}.fasta`).

- PINDER `pinder_heldout_exclusions_v3.tsv` (sha256 `6f8ae174…`) = v2 + 281 Ab/Ag dimers whose antibody CDRs are ≥ 70% to
  the complete sets (rule `cdr_heldout_70`): 4,298 complexes. `summary_clean_v3.json`: 35,967 complexes with a labelled
  usable interface site (21,444 hetero / 13,479 homo / 1,044 Ab/Ag), 16,264 clusters, 2,185,008 training sites with bound
  and free labels (477,379 interface), evaluation 1,559,766 (335,272).
- pKPDB `pkpdb_ab_path_v2.tsv` (sha256 `8cc54013…`) uses the complete sets: 660 entries released (v1: 860). Four entries
  released only by the v2 run (alignment-borderline CDRs) are `withheld_unstable`, so v2 releases only what both runs
  release. `pkpdb_mask_all --revise` (job 746446; previous builds in `revisions/00`, `revisions/01`): **63,814
  structures, 1,791,317 clean sites**, 7,346,962 raw; 186 entries withdrawn, all others byte-identical.

### pKPDB chain distance (2026-10-09)

User decision: no Siamese task on pKPDB; add the closest-chain distance instead. `environment.json` is now
`resolved-v2` for all 63,814 entries (`audits/pinder-label-v1/pkpdb_env.py`, copy in `experiments/4_backbone/scripts/`):
`rsa`, `w_burial` (unchanged, verified identical) plus `chain_distance_A` (residue-level minimum heavy-atom distance to
any other selected protein chain, homomer copies included; as PINDER `partner_distance_A`) and `w_interface`
(`site_weights.interface_weight`). Both are null for one-chain entries; 33,686 entries (4,858,146 sites) have distances.
Tasks 48–99 ran on comp1400 (user-authorised one-off, `_HPC/submission/jax-Ka/pkabench/comp1400-oneoff-env.sh`) while the
general partitions were full.

### Training pools and nested subsets (`pkabench.training_pools`, pool-v3, 2026-10-09)

Pools are training-only. Validation sets: PINDER 400 (`training/ogqt-pinder-factorial-v1/cohort.json` val; pKAI, current
PypKa audit) and the benchmark validation split (142 complexes in use; PypKa, pKAI, pKAI+ in AB/A/B in
`campaigns/production-full-v2`; pKAI there has no terminus sites). Pool rules:

- PINDER: accepted, not in exclusions v3, outside the 400 validation clusters (899 removed), ≥ 1 labelled usable
  interface site with both weights; all PINDER splits and all prepared complexes per cluster. Group = PINDER cluster;
  strata hetero / homo / Ab/Ag. `chain_70_to_validation` flags (does not remove) 258 complexes with a chain ≥ 70% / 80%
  both to a validation chain in another cluster.
- pKPDB: every `pilot.json` record minus entries with a chain ≥ 70% / 80% both to a PINDER validation chain. Group =
  sorted set of the entry's MMseqs2 30% / 80% chain clusters; strata monomer / homomer / heteromer.
- Both: structures with a chain exactly identical to a held-out benchmark chain or a pKPDB reference sequence (or, for
  pKPDB, a PINDER validation chain) are removed; this catches peptides too short for MMseqs2 (12 PINDER, 6 pKPDB beyond
  pool-v2).
- Subsets: seeded hash rank per group within its stratum, `min_fraction = (position + 1) / groups in stratum`; fraction f
  keeps `min_fraction ≤ f` (nested, stratified, groups never split).

| Fraction | PINDER complexes / clusters | PINDER sites (interface) | pKPDB structures / groups | pKPDB sites |
|---|---:|---:|---:|---:|
| 10% | 3,422 / 1,585 | 176,871 (42,819) | 5,878 / 1,595 | 157,674 |
| 50% | 17,444 / 7,931 | 905,085 (229,293) | 31,456 / 7,982 | 891,094 |
| 75% | 26,242 / 11,896 | 1,368,451 (345,733) | 46,860 / 11,972 | 1,334,029 |
| 100% | 35,056 / 15,864 | 1,834,630 (462,913) | 62,874 / 15,965 | 1,760,590 |

Files: `pretraining/pinder-pkai-v1/pool-v3.{tsv,json}`, `pretraining/pkpdb-full-v1/pool-v3.{tsv,json}`. Loader notes:
pKPDB is redundant (largest group 556 structures; the top 1% of groups hold 30% of sites; PINDER: max 5, 9%), so sample
by `group` rather than by structure or site; 5,694 PINDER and 7,917 pKPDB pool structures exceed the 768-residue GQT
bucket cap (`n_res` column).
