# Frozen Siamese interface head comparison

User approved small MLP, matched MLP and shallow transformer with shared bound/free interface heads on frozen 50% relational GQT. Checkpoint seeds 17,29,43; nine fits across three one-GPU jobs. Same existing pKa-labelled surface sites and contact labels as frozen-interface-v1. No GQT weight updates.

Loss: BCE(free logit, contact label) + BCE(bound logit, same label) + mean((sigmoid(free) - stop_gradient(sigmoid(bound))) squared). Weights 1/1/1 fixed in advance. Shared head parameters. The detached bound target receives no consistency gradient; bound BCE still updates the shared head. Identical dropout masks across branches prevent consistency penalizing dropout noise. There is no fixed independently trained teacher.

Architectures and initialization match prior free-only controls: small MLP 4,225 parameters, hidden width 64, no dropout; matched MLP 33,661 parameters, hidden width 510, dropout .1; transformer 33,665 parameters, one width-64 block, four heads, FF128, dropout .1. Transformer head attention remains same-chain in both branches; its bound GQT tokens already contain partner context. No new coordinates or positional features enter the heads.

Data: use existing free token caches, extract bound tokens from exactly the same selected frozen checkpoints, verify checkpoint hash and cached free-token alignment. Labels and RSA gate unchanged: eligible labelled surface sites with free RSA >=.1; contacts <=10 Å between eligible opposite-chain sites; per-site target is any such contact. This omits contacts to non-titratable partner residues. Free is a partner-removed bound conformation, not an independently determined unbound structure.

Training: 20 epochs, AdamW lr .001, weight decay .0001; same 256 class-balanced output-site samples per complex and existing validation split. Same sample indices and context for bound/free branches. Selection and primary evaluation are free-only macro site AP; bound AP is diagnostic. Test data untouched. Frozen GQT pKa predictions remain unchanged.

Smoke tests: inherited chain isolation, zero cross-chain gradients, padding/permutation/chain-label invariance, finite head gradients; additionally verify bound consistency-target gradients are zero and identical-token consistency is zero; cache alignment, checkpoint unchanged, and short training of all heads. All computation on allocated nodes. Single packed bound-token archive per checkpoint; no loose structure expansion.

Remote source: `_submission/frozen-siamese-v1`. Outputs: `_runtime/jax-Ka/pkabench/audits/frozen-siamese-v1`. Compare each head against its matching completed free-only baseline. Any gain measures the combined bound-supervision and consistency objective, not consistency alone; that attribution would require a bound-supervision-only control.

Smoke job 7379396 passed numerical checks and two-epoch training of all three heads. Submitted comparison jobs 7379402/7379404/7379406 for seeds 17/29/43, dependent on successful smoke completion. Report 7379407 compares all nine fits against their completed free-only references.
