# First experimental set-2 screen — 2026-10-04

Three leads were screened against primary measurements and deposited structures.
Machine-readable evidence and unresolved fields are in
[curation/set2_leads.json](curation/set2_leads.json). No quantitative experimental
labels have been admitted and no residue-level free/bound pKa pair was verified.
These are set-2b leads, not a completed set-2a benchmark.

## Decisions

**Barnase–barstar: prioritize and retain reservation seed.** The
[primary binding study](https://pubmed.ncbi.nlm.nih.gov/8494892/) supports pH-sensitive
binding, but the full pH table and matched conditions still need verification.
[1BRS](https://www.rcsb.org/structure/1BRS) A/D are sequence proxies. Its barstar
C40A/C82A construct must be reconciled with the measurements. No full pH series
was transcribed from secondary databases or inferred from a fitted pKa.

**Wild-type protein G–Fc: prioritize, conditional on construct matching.**
[Figure 5b](https://pmc.ncbi.nlm.nih.gov/articles/PMC2673305/) provides the affinity
series. Raw values/uncertainties and assay temperature remain to extract. The
[1FCC entry](https://www.rcsb.org/structure/1FCC) contains C2 protein G and MO61 Fc;
these require sequence comparison with the measured B1/trastuzumab constructs.
Mutant complexes described by the study were modeled, so they are not admitted
to the current no-reprediction round. Mutant and wild-type curves do not count
as independent protein families.

**FcRn–IgG: hold for the first scoring round.** The
[1995 study](https://pubmed.ncbi.nlm.nih.gov/7578107/) establishes pH-dependent
binding, but apparent off-rates alone do not determine equilibrium binding free
energy. A [2024 primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11164218/)
explicitly separates affinity from avidity; those observables cannot be pooled.
The rat [1FRT](https://www.rcsb.org/structure/1FRT) is a 4.5 Å glycosylated complex.
Human [4N0U](https://www.rcsb.org/structure/4N0U) contains Fc-YTE and albumin in a
ternary complex. Neither is automatically a matched binary protein-only input.
No glycans were stripped and no binary structure was fabricated for this screen.

## Sequence reservation check

Job 730545 resolved ten author-chain sequences from four deposited structures:
1BRS A/D; 1FCC A/C; 1FRT A/B/C; and 4N0U A/B/E. Human albumin (4N0U author D)
was not included in the FcRn–IgG reservation proxy. Matching used the existing
30% identity / 80% bidirectional coverage rule on all candidate chains.

| Lead | Candidate matches in train | Matches in test | Non-test groups | Usable interface pairs affected in train/val |
| --- | ---: | ---: | ---: | ---: |
| Barnase–barstar | 0 | 0 | 0 | 0 / 0 |
| Protein G–Fc | 29 | 2 | 1 | 0 / 0 |
| FcRn–IgG | 29 | 121 | 1 | 0 / 0 |

No match means none at this cutoff, not proof of absence of remote or fragment
homologues. The non-test group must be reconciled with prospective experimental
reservations before any member is admitted to later training; zero usable
interface pairs does not mean it is safe to ignore that group forever. This
screen did not change assignments, and 664/150/500 usable-interface counts remain.
The accepted 1AXT H experimental exception does not automatically extend to
these new systems. FcRn overlap results are diagnostic while that lead is on hold.

Artifacts: `audits/set2-screen-v1/report.json`, `matched-candidate-chains.json`,
source-structure receipts and reference sequences/hits. The proposal hash was
checked unchanged. All downloads and sequence work ran on a compute node with
2 CPUs / 4 GB, excluding comp1400, through the 400-core-capped wrapper.

## What remains

For the two priority leads: verify exact sequences/constructs, extract primary
pH/affinity values with conditions and uncertainty, and check the actual structures
through preparation before scoring. Continue searching for additional suitable
systems and genuine residue-specific pKa shifts. The 10–25-system target has not
been met. The available three leads cannot honestly be called a complete
experimental benchmark or used to certify all future set-2 reservations.


## Subsequent structural reservation

[Structural freeze v1](16_structural_split_freeze.md) reserves both protein G–Fc matching sequence groups in test. The previously training-assigned group contains 41 candidates (29 direct reference matches); all moved intact. Experimental measurements and construct matching remain unfinished, and this reservation does not admit quantitative Set 2 labels.
