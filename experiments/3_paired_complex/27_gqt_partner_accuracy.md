# GQT partner-use accuracy audit

The fixed seed-17, epoch-20, explicit-pKPDB-shift backbone GQT is evaluated on the frozen validation split only. No parameters are trained or selected. Exact prepared AB, A and B structures define matched bound-minus-free predictions; current PypKa supplies the teacher and existing frozen pKAI plus zero shift are controls.

The primary causal score is the absolute change in bound GQT prediction after removing all opposite-partner edges. Weak use is below 0.05 pKa and strong use is at least 0.05 pKa. Continuous effects are retained. Binding-shift error is `(GQT_AB-GQT_free) - (PypKa_AB-PypKa_free)`.

Functional-atom ≤10 Å clusters and native PypKa ≥1 kBT clusters are reported separately and may overlap. For antibody records, SAbDab H/L/antigen chain assignments define separate antibody–antigen and heavy–light interventions. The heavy–light contribution is `(AB_full-AB_noHL) - (free_full-free_noHL)` and is not added to the antibody–antigen counterfactual.

Scores first average sites per complex, then complexes per frozen sequence component, with 2,000 component bootstrap replicates. Failed or absent state pairs are excluded explicitly. Strata below five components receive no confidence interval. A new AB inference must match the previous attention audit within 1e-5 pKa, and strict-backbone graphs keep the SG-derived disulfide channel zero.
