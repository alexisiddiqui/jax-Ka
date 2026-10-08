# Experimental expansion audit — 2026-10-05

There is substantially more experimental information than the original 28-site,
two-family pilot. It falls into two different datasets and must not be pooled as
if the labels meant the same thing.

## Absolute pKa archive

The full PKAD-R archive contains 1,024 records on 194 PDB/author-chain tasks. The
existing preparation and uncertainty policies retain 317 evaluation records,
including 286 numeric point labels. After removing local frozen train/validation
sequence overlap, 283 point-label candidates remain across 46 sequence families.
These counts are structural candidates: primary-source, construct, mutation,
state and condition checks still gate admission to training or independent
evaluation.

The frozen PROPKA, pKAI, pKAI+, JAX-Ka and PypKa run uses all 317 retained
records. Its main analysis gives each sequence family equal weight, reports a
separate subset with `|experimental pKa - residue-type null| >= 0.5`, and
bootstraps whole families. Pooled site metrics are supporting diagnostics only.

The corrected run completed on 2026-10-05. Every method reported 253 common
point labels across 38 families; the broader four-method intersection without
JAX-Ka contains 259 points across 41 families. The family-balanced common-set
result is:

| Method | Family-macro MAE | Difference from null, family bootstrap (95% CI) |
|---|---:|---:|
| PROPKA | 0.850 | -0.121 [-0.383, +0.135] |
| pKAI | 0.971 | -0.000 [-0.156, +0.165] |
| pKAI+ | 0.928 | -0.044 [-0.135, +0.050] |
| JAX-Ka | 0.856 | -0.115 [-0.370, +0.143] |
| PypKa | 1.001 | +0.029 [-0.148, +0.259] |
| Fixed residue-type null | 0.971 | reference |

No interval excludes zero, so this archive does not resolve a general method
ranking. The broader four-method sensitivity set gives family-macro MAEs of
1.152 (PROPKA), 1.180 (pKAI), 1.106 (pKAI+), 1.147 (PypKa) and 1.181 (null),
showing that dataset coverage materially changes the apparent ranking. pKAI+
remains diagnostic because its regularization was selected using experimental
performance. JAX-Ka reports 267/283 eligible points (94.3%), comparable to the
other methods' 95.1–96.8% coverage.

The original experimental adapter had inherited the 64-step constructor default
and reported only 218/283 points (77.0%). Re-running JAX-Ka alone with the
accepted frozen-production 1,024-step configuration recovered 49 experimental
records without losing any. All 68 pH grids then passed the unchanged 2e-5
residual threshold. The remaining missing outputs are explicit chemistry or
site-level midpoint-validity outcomes, including disulfide cysteines; none are
imputed. The 64-step release remains preserved for provenance.

## Direct free-to-bound pKa leads

| System | Site | Free pKa | Bound pKa | Evidence | Current status |
|---|---|---:|---:|---|---|
| cNTnC–sTnI peptide | sTnI His130 | 6.13 ± 0.02 | 6.36 ± 0.04 | NMR; bound fraction 0.51–0.78 | Useful quantitative label; exact construct structure and calcium policy unresolved |
| Ca4-calmodulin–PFK peptide | peptide His19 | 6.35 | 7.25 | NMR plus fluorescence binding | Useful quantitative label; matching deposited structure and calcium policy unresolved |
| OMTKY3–porcine pancreatic elastase | PPE His57 | 6.7 ± 0.1 | 5.2 ± 0.1 | Global fit of 143 ITC observations in 19 conditions | Strong label; no exact deposited PPE–OMTKY3 complex verified |

These add three systems and three direct paired sites, not dozens of sites. That
small yield is consistent with the literature: direct residue-resolved pKa
measurements in both free and bound protein states are rare. The troponin and
calmodulin systems contain essential coordinated calcium and therefore remain
outside the current protein-only metal policy. OMTKY3 has the cleanest and
largest shift, but substituting a complex with a different serine protease would
break the measured-structure correspondence.

## Quantitative binding-versus-pH leads

| System | Evidence | Appropriate use |
|---|---|---|
| Human prolactin–prolactin receptor | SPR pH series and histidine mutants; free histidine pKas measured by NMR; deposited antagonist/receptor complexes | Set 2b linkage and mutation ranking. Bound histidine pKas in the paper are thermodynamic model fits to SPR and are not independent direct site labels. |
| PCSK9–LDLR EGF-A/EGF-AB | Roughly 3–4-fold affinity increase from pH 7 to 6 and a 2.4 Å complex | Set 2b linkage after recovering the supporting EC50 table. Preserve EGF-A calcium and separate delta53 from full-length PCSK9. |
| Calmodulin–PFK peptide | Approximately 1,000-fold affinity increase from pH 9.0 to 4.8 | Joint site/linkage case after the calcium-aware branch exists. |
| OMTKY3–PPE | Multi-condition ITC proton linkage and direct fitted His57 shift | Joint site/linkage case if an exact measured complex structure can be identified. |

Binding pH dependence is not converted into a residue pKa. It tests the integrated
charge/linkage output and can coexist with a direct site label only when the
paper independently supports both quantities.

## Admission order

1. Verify primary tables and constructs family by family, prioritizing families
   that carry large null-relative shifts; preserve repeated conditions as
   separate observations.
2. Add prolactin–receptor to Set 2b because it has the best combination of
   structures, affinity data and mutation dissection without requiring a
   residue-specific inferred label.
3. Create a metal-aware validation branch before admitting troponin,
   calmodulin or PCSK9. This is a separate gate, not silent metal stripping.
4. Keep OMTKY3–PPE on a structure hold unless an exact experimental complex is
   found. Do not remodel or substitute a homologous protease for the initial
   benchmark.

The machine-readable evidence and blockers are in
`curation/experimental_expansion_leads.json`. No new label has been admitted to
training by this audit.

## First primary-label checks

Primary-source checking begins with the largest null-relative shifts rather than
random archive rows. The first pass found that label magnitude alone is not an
admission criterion:

| Archive case | Primary-source result | Dataset action |
|---|---|---|
| Human thioredoxin Asp26, 1ERT/1ERU | The primary NMR study reports pKa 9.9 and 8.1 for the reduced and oxidized C62A/C69A/C73A construct. The deposited NMR construct also carries M74T. The archive instead maps these values to wild-type crystals; reduced 1ERT also has a Cys73 intermolecular disulfide. | Hold the original mappings and prepare the matching monomeric NMR structures 1TRW (reduced) and 1TRS (oxidized), preserving all four sequence differences. |
| T4 lysozyme M102K Lys102, 1L54 | The primary structure paper reports pKa 6.5 from both differential titration and 13C NMR on the buried-charge mutant. | Admit candidate after checking the deposited background substitutions against the measured construct. |
| DsbA Cys30, 1DSB | The label is the reduced-thiol pKa, while 1DSB is explicitly oxidized and contains the active-site disulfide. | Keep 1DSB held. Admit sequence-identical reduced 1A2L as the state-correct candidate after the remaining primary-construct and split gates. |
| DsbA H32Y/H32L, 1FVJ/1AC1 | The deposited mutant structures are oxidized; the low Cys30 pKas describe the reduced thiols. | Hold until exact reduced mutant structures are found. |
| LacY Glu325, 2V8N | 2V8N is wild type, but the direct 10.5 SEIRAS site measurement used the G46W/G262W construct. Its agreement with the apparent wild-type binding pK does not remove the sequence mismatch. | Hold from exact structure-conditioned training; preserve as mechanistic evidence. |
| Human glutaredoxin-1 Cys22/23, 1JHB | 1JHB is reduced and unmutated. The archive flags the 3.6 label as approximately wild type because the measured construct replaced three non-active-site cysteines with serines. | Hold from exact structure-conditioned training; preserve as a construct-surrogate measurement. |
| Pig glutaredoxin Cys22, 1KTE | 1KTE contains the Cys22-Cys25 disulfide, while 3.8 is a reduced-thiol pKa. | Hold until a sequence-matched reduced structure is identified. |
| E. coli glutaredoxin-3 Cys11/Cys14, 1FOV | 1FOV is fully oxidized with a Cys11-Cys14 disulfide and contains C65Y; its archive labels describe reduced thiols and only approximate wild type. | Hold for both redox-state and construct mismatch. |
| Human PDI a-domain Cys36, 1MEK | 1MEK contains the Cys36-Cys39 disulfide, while 4.5 is a reduced-thiol pKa. | Hold until a sequence-matched reduced structure is identified. |
| ATP-synthase subunit-c Asp61, 1A91 | The 7.1 NMR value and structure describe the same unmutated monomer in chloroform/methanol/water. | Keep as an exact condition-matched candidate, with a solvent-domain flag; it is not an intact membrane c-ring measurement. |
| Streptomyces subtilisin inhibitor His43, 3SSI | The cited 3.25 value is the midpoint of cooperative acid denaturation, and the paper says His43 cannot protonate in the native conformation. | Reject as a residue-pKa label. |
| Human thioredoxin Cys32, 1ERT | The 6.3 value is for reduced recombinant thioredoxin. 1ERT has a reduced active site but a crystal Cys73-Cys73 intermolecular disulfide. | Hold 1ERT and prepare the reduced monomeric solution structure 4TRX. |
| E. coli thioredoxin P34H Cys32, 2FD3 | The pKa describes a reduced thiol, but 2FD3 contains the Cys32-Cys35 active-site disulfide; the archive also cites only a later review. | Hold pending the original numeric source and a reduced matching structure. |
| S. aureus thioredoxin P31T/C32S Cys29, 2O89 | The primary structure/function paper and deposited double mutant agree, and C32S leaves Cys29 as the measured thiol. | Admit as a candidate after extracting exact assay conditions and uncertainty from the full paper. |

These checks also explain some apparent method failures. JAX-Ka correctly marks
the oxidized DsbA disulfide cysteines as non-titrating. Scoring those predictions
against reduced-thiol labels would measure a dataset state mismatch rather than
model accuracy. The same issue occurs in several of the largest cysteine shifts:
1KTE, 1FOV and 1MEK all contain the active-site disulfide named above. Structural
completeness alone therefore does not certify a label; covalent state is an
explicit admission gate. The structured decisions and evidence URLs are in
`curation/primary_label_checks_v1.json`.

The alternate-structure search does not justify automatic substitution. The
literature states that a reduced pig-glutaredoxin structure was unavailable and
uses a molecular-dynamics model instead. The reduced E. coli glutaredoxin-3
entry 1ILB is a theoretical model removed from the experimental PDB archive;
3GRX is a C14S glutathione mixed-disulfide complex. Neither is an exact reduced
wild-type replacement for 1FOV. DsbA is different: 1A2L is an experimental,
unmutated reduced structure with the same deposited sequence as 1DSB, so it is
being evaluated as a separate state-corrected task.

### Reduced-DsbA recovery result

The 1A2L recovery task passed preparation: exact deposited-sequence agreement
with 1DSB, no detected disulfide, a complete Cys30 thiol and an anchor-aware
`uncertain` gap tier that remains eligible under both structural masks. Cys30 is
25.76 and 27.93 A from the visible anchors beside the two short missing tails,
well beyond their calibrated 15 and 10 A radii. Its conservative envelope
clearances are only 10.16 and 16.76 A, which illustrates why the envelope is a
diagnostic rather than the exclusion rule.

The state correction does not make the case easy for current predictors:

| Method | Reduced 1A2L prediction | Absolute error versus 3.5 |
|---|---:|---:|
| PROPKA | 8.850 | 5.350 |
| pKAI | 9.150 | 5.650 |
| pKAI+ | 8.860 | 5.360 |
| JAX-Ka | no valid midpoint | not scored |
| PypKa | 8.997 | 5.497 |

JAX-Ka converged numerically at 1,024 steps but failed its midpoint-validity
criterion; its intrinsic value was 10.011. These results make Cys30 a valuable
hard experimental example after the remaining primary-construct and leakage
checks. They do not justify a method ranking because this is one site.

### Human-thioredoxin structure recoveries

The archive's human-thioredoxin mappings were corrected rather than accepted
by name. The Asp26 measurements used the C62A/C69A/C73A/M74T NMR construct;
prepared 1TRW is its reduced monomer with no disulfide, and prepared 1TRS is its
oxidized monomer with exactly the expected Cys32-Cys35 disulfide. The Cys32 6.3
measurement is aligned to prepared 4TRX, the reduced M74T monomer with no
disulfide, instead of the Cys73-linked 1ERT crystal dimer. All three recovery
structures are complete. The Cys32 record retains the primary paper's warning
that the NMR samples had heterogeneous processing of the N-terminal methionine.

## Remaining large-shift primary checks

The six remaining candidate records with an absolute residue-type-null shift of
at least 2 pKa units were checked before any experimental fit. Four are direct
site measurements worth retaining as candidates, with their conditions kept as
part of the target. Two archive mappings do not describe the deposited native
structure and remain held.

| Record | System and site | Primary result | Decision |
|---:|---|---|---|
| 110 | Bovine chymotrypsinogen Asp102, 1EX3 | The proton-NMR paper reports pK-prime about 1.4 (archive value 1.36 +/- 0.03) for Asp102 in chymotrypsinogen A at 31 degrees C in D2O. | Candidate after correcting 298 K to 304 K and preserving D2O and pK-prime notation. |
| 595 | E. coli RNase HI Asp10, 2RN2 | Direct carboxyl-carbon NMR gives 6.1. Asp10 and Asp70 have coupled two-step titrations, and Mg2+ changes Asp10. | Candidate after extracting the exact Mg2+ condition and construct from the primary methods/table. |
| 102 | Rat CD2d1 Glu41, 1CDC | Direct NMR gives 6.73 at 1.2 mM protein and 6.36 at 0.1 mM; Glu41 and Glu29 titrate reciprocally and biphasically. | Candidate with 1.2 mM concentration and the coupled/self-association context attached to the 6.73 row. |
| 978 | Yeast Ubc13 Cys87, 1JBB | Direct spectrophotometric measurement on free E2 gives 11.1. 1JBB is unmutated native-length yeast Ubc13 and has an author-assigned monomeric biological assembly. | Candidate after confirming the primary expression construct and affinity-tag cleavage. |
| 753 | Human serum albumin Cys34, 1AO6 | The cited paper reports 8.7 for native Cys34 and 6.9 for the unfolded state. | Hold: the archive maps the unfolded-state value to a folded native structure. |
| 362 | E. coli MutT Lys39, 1MUT | The 8.4 value is an apparent pKa from the `kcat`-pH profile of the active MutT-Mg2+-dGTP-Mg2+ complex, assigned by K39Q mutagenesis. | Hold: it is neither a direct NMR site titration nor an apo microscopic pKa for ligand-free 1MUT. |

These decisions are machine readable in
`curation/primary_label_checks_v1.json`. The admission ledger carries a JSON
`primary_metadata_corrections` field and a `label_interpretation` field so a
later fit cannot discard concentration, isotope solvent, coupling, metal or
molecular-state information. Candidate status still does not enable model-fit
or headline-evaluation eligibility.

## First experimental fit gate

The exact and recovered-exact candidates were reviewed as a separate admission
gate. This gate asks whether the current point-label objective can represent
the measurement, after the structure, construct, redox state and assay context
have been matched. Twelve records pass. None is also enabled for headline
evaluation, because labels used for fitting are not independent test evidence.

| Decision | Records | Reason |
|---|---|---|
| Fit eligible | 142, 311, 317, 584, 585, 589, 590, 916, 917, 1021, 1022, 1023 | Exact or state-corrected structure, direct point label, adequate conditions and both structural masks |
| Condition/context hold | 102, 110, 169 | Concentration-dependent association, D2O pK-prime, or nonaqueous membrane-mimetic solvent is outside the first aqueous structure-only objective |
| Approximate/corrected-label hold | 318, 1020 | Asp70 was reported only as approximately 0.5; archive record 1020's 7.2 was corrected to 6.7 and is not an independent observation |
| Evidence pending | 595, 944, 978 | Exact Mg2+ condition, assay details, or expression/tag-cleavage evidence remains unresolved |
| Construct-mixture hold | 918 | The reduced M74T structure matches, but the measured sample had heterogeneous N-terminal methionine processing |

For ResA, the curated values retain the primary paper's precision: CPHC Cys74
6.33 +/- 0.07 and Cys77 5.71 +/- 0.05; CEHC Cys74 7.4 +/- 0.1 and Cys77
7.5 +/- 0.2, all at 298 K. The archive columns remain untouched and the
curated values are stored separately. The decisions and their explicit reasons
are machine readable in `curation/experimental_fit_gates_v1.json`.
