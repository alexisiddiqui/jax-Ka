# GQT data and target alignment audit

This audit runs before further GQT pretraining because the scratch models remain behind pKAI. It answers two concrete questions without fitting or selecting a model.

1. **Do graph nodes and edges represent the relevant titratable interactions?** Every training and validation query must map exactly to its structure residue, graph node, and titratable group. Across the full 5k pilot, functional-atom distances between titratable sites are compared with Cα distances and graph hops. On the validation intersection with the native PypKa export, microstate interaction matrices are reduced only for this diagnostic using the maximum absolute reference-state difference-in-differences. Coverage is reported at 0.25, 0.5, 1, and 2 kBT. This diagnostic does not turn native tautomer interactions into training labels.

2. **Are the models predicting absolute pKa or shifts?** GQT outputs `MODEL_PKA[group] + 8*tanh(residual)` and its loss is against absolute pKa. Therefore absolute-pKa error is exactly equal to residual-shift error relative to the same fixed model-compound value. This is a single-structure pKa task; it is not the bound-minus-free binding-shift task. The report compares calibration and prediction spread for side-chain GQT, scratch pKAI, and frozen pKAI on common validation support.

The graph gate requires all query mappings to agree, at least 99.9% of functional-atom pairs within 10 Å to have a direct graph edge, at least 99% node coverage and 95% direct-edge coverage for native interactions of at least 1 kBT, and no strong interactions collapsed onto one residue node. Failure sends the next implementation toward explicit titratable-site nodes/edges or functional-atom geometry before any longer pretraining run.

The job uses 32 CPUs and 64 GiB on responsive CPU partitions, excludes `comp1400`, and reads no test labels. Results are written atomically to `_runtime/jax-Ka/pkabench/audits/gqt-data-alignment-v1`.

## Result

Job `739864` completed the audit and all 12 preflight tests. All 215,050 supervised queries across 5,142 structures matched the intended residue key, graph-node index, residue/group identity, and terminal flag. This rules out a general label-to-node alignment bug.

The 20 Å Cα graph also has excellent geometric coverage. It directly connects 603,663 of 604,053 titratable pairs whose functional atoms are within 10 Å (99.935%). On all 142 validation complexes shared with the native PypKa export, it directly connects all 6,194 pairs with a diagnostic interaction strength of at least 1 kBT. Thus the current cutoff is not too short for the observed strong interactions.

The graph gate nevertheless fails because 30 of those strong native interaction pairs are distinct titratable sites on the same residue, principally terminal plus side-chain sites, and therefore share one residue node. Group embeddings distinguish the query outputs, but the context graph cannot represent those two sites as separate interacting entities. This is a real representation defect, though only 30/6,194 strong pairs, so it cannot by itself explain the full GQT-to-pKAI gap.

The target is confirmed to be absolute pKa. GQT emits the fixed group model-compound pKa plus a learned residual and minimizes error against the absolute teacher pKa. Re-expressing both prediction and target as shifts from that same group constant leaves every error unchanged. No bound-minus-free binding shift is used in this pretraining experiment.

On 7,764 common validation sites, the group-macro MAEs are 0.5552 for side-chain GQT with dropout, 0.3867 for scratch pKAI, and 0.3221 for frozen pKAI. GQT's residual-shift calibration slope is only 0.593 and its predicted shift standard deviation is 0.760 times the teacher standard deviation, compared with 0.877 and 0.985 for scratch pKAI. Its predictions are therefore compressed toward model-compound values. GQT MAE rises from 0.223 for sites below 0.25 kBT maximum native coupling to 0.864 for sites at or above 2 kBT. The evidence points to under-representation or underfitting of coupled electrostatic environments, rather than missing graph edges or an absolute-versus-shift bookkeeping error.

Before longer pretraining, the next controlled model experiment should create one token per titratable site, place it at its functional atoms, retain residue/backbone context, and give terminal and side-chain sites separate tokens. Compare it with the current residue-node GQT on the same split and budget. Capacity/depth remains a separate lever because the rare same-residue collisions do not explain the broad calibration compression.
