# Experimental inventory completion — 2026-10-04

The accepted antigen-held-out split and 1AXT H independent-evaluation exception
remain unchanged. This work audits reference coverage before any production freeze.

## PKAD-3 source reconciliation

The locally installed KaML-CBtree train/test tables contain 1,707 unique
structure/chain/residue/value/accession records and 168 unique UniProt identifier
strings. Deduplicating by the released UniProt-residue annotation and value gives
1,073 records. These are different counting units from a database's experimental
measurements or mutant constructs; they cannot establish full PKAD-3 coverage.

Job 730521 downloaded the official PKAD-3 page on a compute node. The page
identifies itself as PKAD-3 Version 2 and explicitly says that only a preview is
initially displayed. Its public Download All control is therefore needed for a
complete inventory. The source is a NiceGUI application, not a Dash application;
404 responses from attempted Dash metadata endpoints do not mean the database
itself is unavailable. Public source download receipts are preserved under
`audits/experimental-inventory-v1/`.

## Experimental set-2 curation queue

These are source-supported candidates, not accepted quantitative benchmark rows.
Do not infer per-site pKa shifts from binding pH dependence alone.

| System | Primary evidence | Remaining work before inclusion |
| --- | --- | --- |
| Barnase–barstar | [Schreiber and Fersht binding experiments](https://pubmed.ncbi.nlm.nih.gov/8494892/) | Extract pH-dependent affinity observations, conditions and constructs. 1BRS A/D already serve as sequence reservation seeds; verify construct correspondence. |
| Protein G B1–IgG Fc | [Watanabe et al., pH-sensitive binding mutants](https://pubmed.ncbi.nlm.nih.gov/19269963/) | Extract SPR/ITC conditions and pH series for wild type and variants; map exact constructs and structure chains before reserving homologues. |
| FcRn–IgG | [Raghavan et al., receptor/antibody variants](https://pubmed.ncbi.nlm.nih.gov/7578107/) | Separate affinity from avidity, map receptor/Fc constructs and complete partner stoichiometry, extract quantitative pH series and reserve matching protein groups. |

Additional systems may be added after curation, but no system that overlaps
training can be presented as an independent experimental test without resolving
its reservation. These seeds do not constitute the planned 10–25 systems. No
experimental labels, pH values or affinities have been fabricated or imputed.

## Official PKAD-3 release check completed

The public Download All control yielded 1,804 records (download SHA256
`34da309ea8453cfddff44bfd168ebc307d96a4c9d4abb1a3ba2858d6776305f0`).
The repeated CSV header row was explicitly excluded and record IDs were verified
unique. This release is versioned by its downloaded content; older published
counts are not used as a completeness test.

- 1,608 records map to already-checked deposited chains.
- 178 mutant/model records are covered by the recorded parent-sequence family
  proxies; this does not verify exact mutant constructs.
- 12 records map to five additional deposited chains: 2MI7 A, 3SOA A, 1LKJ A,
  2QDB A and 2RDF A. Job 730527 resolved all five and searched their sequences
  against the candidate universe. All 19 matching candidate pairs are already
  assigned test; no new train/validation conflicts were found.
- Six entries are alanine-flanked model pentapeptides (AADAA, AAEAA, AAHAA,
  AACAA, AAKAA, AAYAA), outside the protein-complex evaluation scope.

No official record remains unaccounted for under these stated rules. The accepted
1AXT H exception remains in force. The split is unchanged. The official release
reservation check passes when combined with the prior scoped all-chain audit.
This closes PKAD-3 reference-inventory coverage for this snapshot, not the
separate literature curation of experimental set 2.

`official-delta-report.json` and `official-delta-matches.json` retain the evidence.
The versioned `usable-proposal-v2/independent-experimental-scope-v2.json` adds the
five reference IDs to scoring eligibility and links the official release and
prior scope hashes. Earlier scope/audit files remain intact. Production consumers
must use scope v2; unknown future references still require review.

The [first set-2 screen](11_set2_screen.md) is complete. Barnase–barstar and
wild-type protein G–Fc are priority leads pending construct/measurement curation;
FcRn–IgG remains on hold. Ten structure-proxy sequences were checked. One
training-assigned group matches the Fc-related leads but contributes no usable
interface pairs; assignments and 664/150/500 counts are unchanged.
