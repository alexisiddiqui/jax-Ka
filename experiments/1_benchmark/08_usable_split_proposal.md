# Usable-size split feasibility — 2026-10-04

User approved allocation by usable pairs rather than raw candidate count,
antigen-family separation with independent antibody novelty reporting, and
keeping large groups intact rather than enforcing the previous 5% size ceiling.
This is a feasibility proposal, not a frozen production split.

## Current result

| Split | Candidate pairs retained | Pairs with usable sites | Pairs with usable interface sites | Sequence groups |
| --- | ---: | ---: | ---: | ---: |
| Train | 5,980 | 932 | 664 | 2,117 |
| Validation | 284 | 171 | 150 | 83 |
| Test | 2,419 | 643 | 500 | 228 |

Counts use training masks for train and evaluation masks for validation/test.
Antibody–antigen interface pairs number 85/22/117 respectively. All 8,683
candidate records remain; rejected, fully masked and role-ineligible records do
not contribute to usable quotas. Largest connected group remains 541 candidates.

The allocator first reserves groups matched to experimental reference sequences
for test, then selects whole groups to reach 500 test and 150 validation pairs
with usable interface supervision. Among equally close sizes it minimizes lost
training-interface pairs. This establishes feasibility; it is not random sampling
or a guarantee of representative family composition. No teacher outcomes were
used. All detected protein similarity edges stay within one split.

## Antibody review

The full SAbDab table uses extended `pdb_0000...` identifiers; these are normalized
before mapping author chains onto the downloaded assembly's label chains.
The review identifies 133 antibody-internal pairs and leaves 160 unresolved.
The unresolved cases include ambiguous names: heavy/light chains are not unique
to antibodies. These two categories remain in the conservative sequence graph
but are not counted toward benchmark targets. Two additional internal cases were
found beyond the original name-flagged set.

Final roles: 1,460 antibody–antigen, 6,930 general, 133 antibody-internal,
160 unresolved. This does not prove that every name-negative general entry is
antibody-free. No antibody partner was reconstructed or silently regrouped.

## Experimental reservations

Local KaML tables supply 378 reference identifiers/chains; two barnase–barstar
chains are added from [1BRS](https://www.rcsb.org/structure/1BRS), A and D.
The earlier inventory's count of 372 "PDB entries" was imprecise: its identifier
column also contains mutant/model identifiers that are not PDB accessions.

305 reference chains resolve to deposited protein sequences. The remaining 75
mutant/model identifiers map to three UniProt parents: P00644, P09850 and Q6J4G7.
Their downloaded parent sequences are included for family reservation, explicitly
as proxies rather than exact experimental constructs. Source URLs and hashes are
saved. Searching these sequences at 30% identity and 80% bidirectional coverage
reserves 45 whole candidate groups for test. Adding parent proxies does not alter
the allocation counts above.

This is sequence-based reservation, not just PDB-ID matching. However, fragment
coverage, exact mutant construct mappings, completeness against PKAD-3 and the
remaining experimental set-2 systems still need review before a production freeze.

## Artifacts and validation

Runtime: `universe/combined-split-v1/usable-proposal-v2/`.
`roles.json`, `reference-inventory.json`, reference download receipts/sequences,
`reference-hits.tsv`, `proposal.parquet`, and `report.json` retain the evidence.
Earlier proposals remain available, including the allocation before parent
proxies. CDR novelty is recalculated for this assignment at 30/50/70/90% identity;
those are sensitivity analyses, not a selected biological family definition.

Jobs 730508/730511 reviewed roles; 730509/730513 resolved and matched reference
sequences; 730512/730514 calculated feasibility. Jobs 730515/730516 added source
hash checks, serialized-table equality checks and current antibody novelty.
All used compute nodes excluding comp1400, 2 CPUs / 4 GB per job, and the locked
400-core submission cap. No new all-against-all clustering or pKa runs were needed.

Before freezing: finish unresolved antibody/experimental coverage review, assess
family and interface-type representation, and adopt a documented CDR novelty
criterion. Production labels and their coverage remain subsequent checks.

Final verification jobs completed successfully. Of the 117 test antibody–antigen
pairs with usable interface sites, both-unseen counts are 24/31/114/116 at CDR
identity thresholds 30/50/70/90%. Antigen separation is held fixed; the difference
reflects the chosen antibody novelty definition. These counts remain provisional
until antibody coverage and the CDR criterion are finalized.

The completed [representation review](09_split_representation_review.md) finds
107/500 test pairs in one MHC/beta-2-microglobulin-containing sequence component,
and differing class/mask-burden distributions. See its reporting recommendations
before treating these feasible counts as a balanced benchmark. Dataset unchanged.
