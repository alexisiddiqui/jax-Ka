# Matched context masking and hidden dropout

Authorized 2026-10-06. Use the cleaned 5,000-structure pilot, its unchanged
training labels/eligibility masks, and the same unaugmented clean validation set.
All arms start from random seed 17. Rerun controls: the previous GQT experiment
included one serial epoch, so reuse would confound this augmentation comparison.

| Model | Arms | Training / selection |
|---|---|---|
| GQT, 49,709 parameters | Baseline; 10% hidden dropout; 5% context masking; both | True batches of eight; strict float32; existing Adam .001; fixed epoch 20 |
| pKAI, 3,608,001 parameters | Native baseline; 5% context masking | Native dropout .5/.125/.03125; Adam 1e-6, weight decay 1e-4, batch 256; existing validation-MSE early stopping |

For each structure and epoch, draw Bernoulli(.05) masks over context residues.
Protect the union of all clean GQT supervised residue centres, including sites
unsupported by pKAI. The 5% probability applies to unprotected context residues,
not 5% of every input feature. Seed the mask by stable hash of seed, epoch and PDB
ID. Use identical masks across both models and masked GQT arms; record hashes each
epoch. Repeated GQT draws of a structure within an epoch share its mask. This is
a deliberate matching rule despite the models' different native sampling units.

GQT removes every spatial edge incident to a masked residue, zeroes its radial
and directional channels and attention switch, and clears its existing explicit
frame-valid indicator. Sequence identity and node indexing remain stored; with
spatial edges removed, the current spatial encoder cannot pass that isolated
residue's identity to other sites. No geometry is inferred from its hidden edges.
Targets, masks and protected query geometry remain unchanged. Hidden dropout
acts on attention and feed-forward outputs before residual addition, including
the query block. RNG keys are independent across layers, structures and updates,
and reproducible from seed/epoch/update. Evaluation disables it.

pKAI excludes all atoms belonging to the same masked residues before selecting
its first 250 sorted environment atoms. Cache *all* atoms within the native 15 Å
cutoff, with native classes/inverse-square features and owner-residue indices;
filter then take the first 250 each training batch. This allows atoms beyond the
original 250 to refill the input. Preserve native residue one-hot features. Do
not add an unavailable-atom indicator to pKAI: its native architecture cannot
distinguish missing context from absence. Verify all clean unmasked rows against
the existing native feature archive and verify GQT/pKAI residue indexing.

Indexing audit found native pKAI can contain additional amino-acid species outside
GQT's selected protein graph (for example, LYS A505 in 1bbu). Recover graph-node
identifiers from the exact original resolved CIF view and match by original
chain/residue/insertion/type, never by row number. Native-only context remains
identical in both pKAI arms and is never randomly masked; list it in receipts and
the report. Thus matching refers to the shared protein residues, not identical
native model inputs. This preserves the current pilot's inputs for the ablation.

This is supervised prediction from incomplete observations of a complete labeled
structure, not a claim that physical residue deletion preserves pKa. Augmentation
never restores quality-excluded labels. No rotations, coordinate noise, sequence
mutations, or contiguous-gap masks are introduced in this first ablation.

Validation is identical and unaugmented in all arms. Compare identical supported
sites, report group-macro MAE and paired 2,000-replicate group-bootstrap changes
versus each model's control. Audit matching mask hashes. Results are seed-17
development results; confirm promising settings across independent seeds later.
pKAI and GQT retain different native inputs, capacities and selection rules.

Runtime: `pretraining/augmentation-v1`. CPU preparation: 32 cores / 64 GB away
from comp1400. Each GPU job: one A40, 8 cores / 16 GB; maximum four concurrent
GPU jobs, user-wide queued/running CPU requests below 400. Original artifacts
remain separate. Tests and preparation gate the training jobs.

Preparation job 738397 completed: 5,000 structures, 202,152 clean native pKAI
sites, and 22,646,974 candidate atom contributions. Every eligible unmasked
input reproduced the existing native feature archive. GPU gate 738398 passed
15 tests. A one-shot scheduler dispatcher (`augmentation-submit.sh`) waits for
the account-wide queued/running requests to fall below the 400-core cap before
submitting each job. Unrelated work was above that cap at launch time; it is
not cancelled or modified. Job IDs and dispatcher state are recorded under
`augmentation-v1/jobs.tsv` and `submission-status.txt`.
