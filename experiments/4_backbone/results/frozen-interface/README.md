# Frozen GQT interface probes

Approved: compare free-only per-site and partner-specific free-token pair heads on frozen 50% relational GQT. User chose existing pKa-labelled sites for this first comparison.

Backbones: selected checkpoints `control-v2-50-relations-s17`, `s29`, `s43`. Both heads run per checkpoint, with independently trained head weights only; three jobs may run in parallel, one GPU and one node per job. Features are cached from the free branch. No GQT gradients or optimizer updates occur; selected checkpoint digest checked before and after extraction. Free-branch partner perturbation tested.

Scope: pKa-labelled sites with finite paired targets, a site token, and free RSA >=0.1. Every unordered cross-chain pair among these surface sites is a candidate. Positive label: minimum observed heavy-atom distance <=10 Å in the bound structure. Distances and bound tokens are never head inputs. Surface/contact labels use existing fixed-coordinate structures; free means partner removed, not an independent unbound experimental conformation.

Site head: normalized own free site token -> width-64 MLP -> contact logit. Target: any contact to an eligible labelled surface site on another chain. Pair head: shared normalization, symmetric [token product, absolute difference] -> width-64 MLP -> contact logit. Pair scores aggregate by maximum for site-level comparison on identical support. Neither head is a transformer. This first test omits contacts to non-titratable residues and buried partner sites; it does not reproduce the original all-partner-residue contact-score label.

Training: existing 50% PINDER split only; unchanged validation split; no test data. 20 epochs, AdamW 1e-3, weight decay 1e-4. Per-complex 256 examples sampled per epoch, 50/50 positives/negatives when both are present, otherwise sample the available class. Uniform complex order ensures equal complex contribution. Site and pair heads have different parameter counts and supervision; report as practical probes, not a parameter-matched ablation. Seed-matched comparisons.

Selection: validation macro site average precision, over complexes containing positives. Pooled AP, AUROC and Brier include no-positive complexes. Pair AP also reported. Maximum aggregation and balanced sampling prevent treating output scores as calibrated probabilities. No change to pKa accuracy is expected because GQT and its pKa head stay frozen.

Labels and feature caches are packed archives, not loose files. Data preparation and inference run on allocated compute nodes. New outputs on Isambard: `_runtime/jax-Ka/pkabench/audits/frozen-interface-v1`. New scripts: `_submission/frozen-interface-v1`; frozen source: `_submission/invsqrt-v2/src`. Existing source snapshots/checkpoints remain untouched.

Smoke 7375934 completed successfully in 30 seconds: packed label construction, free-token partner independence, unchanged checkpoint digest, finite training for both heads, and pair-order symmetry. Initial smoke 7375909 failed only because sklearn was absent; replaced metrics with exact tie-aware NumPy AP/AUROC. Full label preparation 7375949; parallel per-checkpoint probe jobs 7375951/7375952/7375953; comparison report 7375954. Smoke validation is a tiny training subset for execution checks only; full runs use the existing held-out validation split.
