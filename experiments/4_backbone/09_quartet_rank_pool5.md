# 09 — Pairing (quartet) losses, pool-v5 and the Ab/Ag unit check

Status (2026-11-21): experimental. All arms are single seed (17); nothing here changes the exported default.
Code: `src/pkatrain/production_residue.py` (quartet / chain quartet terms), `production_train.py` (flags),
`production_extra.py` (extra record stores). Jobs, scorers and tests: `scripts/quartet-rank/`, `scripts/pool-v5/`.

## Losses

| Flag | What it compares | Snapshot |
|---|---|---|
| `--quartet-weight λ --quartet-count 128` | inside one complex: native contacts i–j, k–l vs crossed i–l, k–j (absent from the 10 Å table); softplus(1 − (s_ij + s_kl − s_il − s_kj)); half charge-matched | quartet-rank-v1 |
| `--chain-quartet-weight λc` (`--chain-quartet-count/-tau/-k/-abag-min`) | across two same-class complexes in a batch: S(A1,B1) + S(A2,B2) > S(A1,B2) + S(A2,B1), S = τ·logsumexp(s/τ) − τ·log k over surface pairs; same group / shared UniProt skipped; Ab/Ag regrouped within size buckets | quartet-rank-v2 |

Both score the free-branch edge head (`aux_redge`) on unmasked free tokens; both are bit-identical to the base at weight 0.
Per-residue (quartet) or per-chain (chain quartet) additive propensity cancels by construction.

## pool-v5

pool-v4 screened antibody chains on whole-chain identity to val antibodies, removing 83 % of Ab/Ag candidates on
conserved frameworks. pool-v5 screens antibody chains on CDRs instead (ANARCI IMGT; leak if CDR3 ≥ 0.90 or all CDRs
≥ 0.80 identical to a val Ig domain of the same type), antigens unchanged: 553 Ab/Ag structures (321 groups) vs 134.
Val set identical to pool-v4. Train with `--pinder-manifest training/pool-v5/pinder-manifest.json --pinder-extra
<pinder>/abag-v5 --mask-store-extra <pinder>/store-hotspot-v5abag` (builder: `scripts/pool-v5/`).

## Results (s17, masked2 unmasked, PINDER val 800; noise ~0.02 gates, 0.012 pKa)

| Run | Ab/Ag swap | swap all | wrong-partner | Ab/Ag wrong antibody | within AUROC | edge AP | pKa clean |
|---|---:|---:|---:|---:|---:|---:|---:|
| pool-v5 control | 0.682 | 0.803 | 0.585 | 0.513 | 0.657 | 0.046 | 0.612 |
| pool-v5 q03 (quartet 0.3) | 0.695 | 0.821 | 0.650 | 0.590 | 0.676 | 0.063 | 0.612 |
| qc (quartet 0.3 + chain 0.15) | 0.639 | 0.854 | 0.793 | 0.617 | 0.670 | 0.052 | 0.616 |
| conly (chain 0.15) | 0.654 | 0.845 | 0.783 | 0.603 | 0.651 | 0.050 | 0.608 |

The chain quartet lifts cross-complex gates strongly but lowers Ab/Ag swap 0.03–0.04; q03 is the only arm that raises
every gate including Ab/Ag. Seeds 29/43 not yet run.

## Ab/Ag unit check (`scripts/pool-v5/fv-check/`)

PINDER records are two-chain dimers, so a Fab enters training as one antibody chain + antigen (pool-v5: ~210 heavy, 88
light, ~232 nanobodies); POISK embeds the whole Fv. 72 val Fab cases rebuilt as whole Fv (partner chain from the
partner-antigen PINDER entry or RCSB, superposed into the PINDER frame): the missing chain is the only contact for a
median 29 % of epitope residues. Embedding the Fv (VH–VL edges kept) instead of the single chain lowers contact AUROC
by 0.024–0.045 in every model (3–7 SE) and the Ab/Ag swap analogue by ~0.03; model-vs-control differences are the same
in both views, so the single-chain gates still rank models fairly.

## Running the tests

```bash
cd experiments/4_backbone/scripts/quartet-rank/v1 && JAX_PLATFORMS=cpu GQT_PARAMETER_SCALING=1 python3 test_quartet.py <repo>/src <base-src>
cd experiments/4_backbone/scripts/quartet-rank/v2 && JAX_PLATFORMS=cpu GQT_PARAMETER_SCALING=1 python3 test_quartet2.py <repo>/src <base-src>
```

`<base-src>` is the tree the weight-0 bit-identity checks compare against (the drsa-within-v1 / quartet-rank-v1b
snapshots, `$S/_submission/...` on Isambard). The pairing-rank reference is read from `$PAIRING_RANK_CORE`.
`v1b/test_extra.py` `test_tree` / `test_cli_flags` pin the v1b snapshot's exact file tree and fail on later trees by design.
