# Explicit titratable-site token pilot

## Question

Does explicit communication between titratable sites improve the backbone-only GQT at the fixed 20 A graph cutoff? The pilot isolates three effects: adding site-to-site communication, adding relative backbone orientation, and retaining unsupervised ARG sites as context.

## Frozen comparison

All arms use the existing cleaned 5k pKPDB split, the 50k backbone residue encoder, explicit signed pKPDB `PK_MOD` shifts, batch size 8, float32, seed 17, and the 20 A residue and site cutoffs. The current 20 A backbone GQT result is reused as the control. No test records are read.

| Arm | Site graph |
|---|---|
| Current GQT | Existing residue encoder and residue-query attention |
| Site-distance GQT | Candidate site tokens, local residue-context attention, and one site-to-site block using distance, directions, chain/residue flags, sequence separation, and directed site-type pairs |
| Site-orientation GQT | Site-distance arm plus the nine entries of `F_i^T F_j` |
| No-ARG ablation | Site-orientation arm with ARG site tokens and incident site edges masked |

Candidate tokens are created for ASP, GLU, HIS, CYS, TYR, LYS, ARG, N-termini, and C-termini. A terminal titratable residue has separate side-chain and terminal tokens. Every token starts from its encoded residue representation plus a learned site-type embedding. ARG participates in context but is never included in the pKPDB loss.

For residue `i`, the frame is constructed from N, CA, and C:

`x = normalize(C - CA)`, `y = normalize((N - CA) - ((N - CA) dot x)x)`, and `z = x cross y`.

A directed site edge contains 16 distance RBF values, the unit direction in each endpoint frame, the nine relative-orientation entries, same-chain and same-residue flags, signed sequence separation, and a learned directed site-type-pair bias. Site pairs are connected when their residue CA atoms are within 20 A.

## Training and selection

The three new arms add one site-attention/feed-forward block to the same residue encoder. They use AdamW with weight decay `1e-4`; learning rate `1e-3` through epoch 10 followed by cosine decay to `1e-5`; maximum 20 epochs; and validation patience 8. The selected checkpoint minimizes frozen-validation group-macro MAE.

The report must show overall validation MAE and MAE in absolute teacher-shift bins `<0.5`, `0.5-1`, `1-2`, and `>=2` pKa. The `>=2` bin is the primary diagnostic for range compression. It also records parameter count, selected epoch, runtime, and peak VRAM.

## Gates

Before production training:

1. Build one structure and verify labelled-site mapping, absence of ARG supervision, and multi-token terminal residues where present.
2. Apply a proper random rotation plus translation and require invariant graph geometry.
3. On the GPU, require predictions and site-attention weights to remain invariant after that rigid transform.
4. Require one finite forward/backward/update for every new arm.

The full site graph is prepared on CPU workers. The rotation gate and the three training arms run on comp1400 A40 GPUs. This is a one-seed screen; only a promising arm will be repeated.

## Implementation

- `src/pkatrain/site_graph_data.py`: candidate construction, invariant geometry, sidecar data, mmap loader, and ARG ablation.
- `src/pkanet/site_model.py`: site initialization, local residue-context attention, indexed site-to-site attention, and shift prediction.
- `src/pkatrain/gqt_site_tokens.py`: gates, training, validation, shift-bin analysis, and report generation.

Runtime artifacts are written under `_runtime/jax-Ka/pkabench/pretraining/gqt-site-tokens-v1`.

## Submitted run

The one-structure geometry gate passed in job 745337. The gated production chain is:

| Stage | Slurm job |
|---|---:|
| Full site-graph preparation | 745338 |
| Read-only mmap construction | 745339 |
| Experiment registration | 745340 |
| GPU prediction/attention rotation gate | 745341 |
| Three-arm finite-update smoke array | 745342 |
| Smoke verification | 745343 |
| Three-arm one-seed training array | 745344 |
| Frozen-validation report | 745345 |

Preparation covered 5,142 structures and produced 599,284 candidate sites, including 94,637 ARG context tokens and 2,729 residues carrying multiple tokens. The site graph contains 17,275,556 directed edges. Downstream stages use `afterok` dependencies so a failed correctness gate prevents training.

## Result

All gates and runs passed. The rigid-transform probe changed site-attention weights by at most `7.91e-6` and changed predictions by exactly zero at float32 precision. Each site-token arm used 67,725 parameters and about 1.42 GiB peak VRAM.

| Arm | Overall MAE | `<0.5` shift | `0.5-1` | `1-2` | `>=2` | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|
| Current GQT | 0.5845 | 0.3526 | 0.4571 | 0.7616 | **1.4838** | 14 |
| Site-distance | 0.5733 | 0.3163 | 0.4285 | 0.7532 | **1.5619** | 14 |
| Site-orientation | **0.5675** | **0.3160** | 0.4424 | **0.7439** | **1.5014** | 9 |
| No-ARG | 0.5876 | 0.3559 | 0.4454 | 0.7481 | **1.5051** | 14 |

The orientation arm improved overall MAE by 0.0170 relative to the current GQT and by 0.0058 relative to site-distance. Removing ARG worsened overall MAE by 0.0201 relative to orientation, which supports retaining ARG as an unsupervised context token in this seed.

The overall gain came from the low- and medium-shift bins. None of the site-token arms improved the primary `>=2` shift bin; orientation was 0.0176 worse than the current GQT there, and site-distance was 0.0781 worse. This does not resolve shift-range compression. The orientation result is promising for average accuracy but requires repeated seeds, and a subsequent change should target the large-shift objective rather than merely expanding site communication.

The generated machine-checked report is `_runtime/jax-Ka/pkabench/pretraining/gqt-site-tokens-v1/report.md`.
