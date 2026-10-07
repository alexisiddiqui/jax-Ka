# GQT approach attention and causal partner-use audit

## Question

Does the backbone graph-query transformer use residue information from the other biological partner as two partners approach, especially for coupled clusters of titratable sites?

This is an interpretability diagnostic for the fixed epoch-20 unweighted explicit-pKPDB-shift model. It does not train or select a model and does not inspect the test set. The rigid separation path is a controlled model probe, not a physical binding pathway.

## Frozen scope

- All 142 frozen validation complexes, retaining component IDs and antibody/antigen roles.
- The current backbone-only GQT checkpoint: seed 17, epoch 20, 49,709 parameters.
- Partner B is translated outward from the bound structure by 0–40 Å in 2 Å increments. The direction is the partner-centroid axis, with its sign checked for monotonically increasing minimum interpartner Cα distance.
- Biological partners come from the prepared A and B structures. Chain and partner are distinct: antibody heavy/light communication is `same_partner_other_chain`; antibody-to-antigen communication is `opposite_partner`.

## Registered clusters

Two definitions are reported without selecting between them after seeing results:

1. Functional-atom graph: edges at ≤10 Å and connected components containing a cross-partner edge.
2. Native PypKa graph: edges at ≥1 kBT and connected components containing a cross-partner edge.

Sites may belong to both definitions. Native PypKa interactions define an analysis stratum only; no PypKa labels exist along the artificial separation path.

## Evidence

Attention mass and opportunity-normalised enrichment are recorded per layer, head, edge class, distance and clustered query. Attention is descriptive. Causal evidence comes from four interventions:

- remove all opposite-partner graph edges;
- retain geometry but zero opposite-partner amino-acid identity;
- shuffle opposite-partner amino-acid identities within that partner 20 times, using the same deterministic shuffles at every separation;
- separately remove edges to, and mask the identities of, other chains in the same biological partner.

The primary practical-use threshold is an absolute prediction change of 0.05 pKa. Fractions above 0.01 and 0.10 pKa are sensitivity analyses. Continuous effects remain the primary result. Aggregation first averages sites within complexes and complexes within frozen components; confidence intervals use 2,000 component bootstrap replicates.

## Outputs

`site_trajectory.parquet` contains prediction and intervention effects. `attention_summary.parquet` contains clustered-site attention summaries. `clusters.parquet` and `complexes.parquet` expose denominators. `summary.parquet`, the plots, and `report.md` are generated only after the full inference artifact verifies successfully.

