# Fixed bound teacher and free interface student

Approved small-MLP distillation experiment on frozen 50% relational GQT, checkpoint seeds 17/29/43. Each one-GPU job trains an independent bound teacher, freezes its selected weights, then trains two separate free-only students at lambda .1 and 1. Same 4,225-parameter small MLP as the existing labels-only baseline; no shared teacher/student parameters and no GQT updates.

Teacher objective: BCE(bound contact logits, labels), selected by bound validation macro AP. Student objective: BCE(free logits, labels) + lambda*MSE(sigmoid(free logits), detached frozen teacher probabilities). Teacher targets generated on training complexes only. Student selection and evaluation use free tokens exclusively. Teacher predictions require bound input and are diagnostic only.

Identical cached free/bound tokens, checkpoint hashes, labelled-site surface support and train/validation splits as prior frozen probes. Surface RSA >=.1, positive if any <=10 Å contact to an eligible pKa-labelled partner surface site. Contacts to non-titratable residues omitted. Free coordinates are partner-removed bound structures. No independent test data used.

20 epochs each; AdamW .001, weight decay .0001, no dropout. Same initialization and 256 class-balanced sites per complex per epoch, same random order as prior small-MLP baseline. Existing labels-only runs reused. Each teacher selected before training either student. Verify teacher parameter hash unchanged after each student; zero teacher gradients; lambda-zero objective gradients equal labels-only loss; identical predicted probabilities yield zero consistency penalty. GPU smoke trains all stages on a small subset first.

Remote source `_submission/frozen-distill-v1`; results `_runtime/jax-Ka/pkabench/audits/frozen-distill-v1`. One node and one GPU per job; three seed jobs eligible in parallel, final comparison report dependent on all three.

Smoke 7381972 completed short teacher and both student training. Full jobs 7381985/7381986/7381987 submitted for seeds 17/29/43 after the successful smoke gate; comparison report 7381988.
