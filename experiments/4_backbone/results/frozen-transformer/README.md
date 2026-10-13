# Frozen free site transformer comparison

User approved a shallow transformer on the free-only site head, with a matched-capacity per-site MLP control. Same three frozen 50% relational GQT checkpoints, seeds 17/29/43. Reuse completed `audits/frozen-interface-v1/features-s{seed}.zip` and `labels.zip`; no backbone extraction or training needed.

Transformer: one pre-norm block, width 64, four heads, feed-forward width 128, final LayerNorm and scalar site classifier. 33,665 trainable parameters. Attention is all-to-all among cached eligible pKa-labelled surface site tokens on the same chain, with padding masked. No new geometry, position features, bound tokens or partner tokens. The cached free tokens already encode local structure. No cross-chain access.

Control: normalized own free token, width-510 hidden MLP and scalar output: 33,661 parameters. Both new heads use dropout 0.1, AdamW lr .001 and weight decay .0001. Compare also with original width-64, no-dropout site MLP, already completed. New matched MLP controls capacity and dropout when testing contextual attention.

Training: 20 epochs, same structure order and 256 class-balanced sampled output sites per complex per epoch as prior probe. Transformer retains all own-chain context sites while the loss uses sampled targets. Same train/validation splits and surface gate (free RSA >=0.1). Target: any <=10 Å heavy-atom contact to an eligible labelled surface site on the opposite partner. This excludes contacts to non-titratable residues; free conformations remain partner-removed bound structures.

Select by validation macro site average precision among positive-containing complexes. Report pooled AP and AUROC as well, retaining no-positive complexes there. Compare identical per-complex validation support with earlier probes. No test data used. Scores are not calibrated probabilities owing to balanced sampling. Frozen GQT pKa predictions do not change.

Smoke gate: exact absence of cross-chain input gradients, partner-token perturbation invariance, padding invariance, chain relabelling invariance, permutation equivariance, finite dropout gradients, and short actual training of both heads. Source and results isolated under `_submission/frozen-transformer-v1` and `_runtime/jax-Ka/pkabench/audits/frozen-transformer-v1`. One GPU and one node per job; three checkpoint jobs eligible in parallel. Each job trains transformer then matched MLP; final report follows all three.

Smoke 7377219 completed successfully in 29 seconds. Three comparison jobs submitted: 7377230 (17), 7377232 (29), 7377233 (43). Report 7377234 depends on all three. Each job trains the transformer and matched MLP, with no new GQT feature extraction.
