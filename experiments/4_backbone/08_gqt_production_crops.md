# Inner-radius site crops on the production GQT recipe (2026-10-11)

User request: retest cropping on the current production recipe, at the 10% and 50% pools. Experiment 35 found no
benefit from query-centred crops, but on the old single-task 5k GQT, which did not overfit, and with only the centre
query supervised on a cropped draw. The production recipe does overfit at 50% without dropout (experiment 07), so
the question is whether crops add anything on top of dropout 0.1 and the burial auxiliary (`drop10-bur1`, the
recipe of the 100% width sweep).

## Method

`pkatrain.production_crop`, enabled by `production_train train ... --crop-radius R --crop-inner R_in
[--crop-probability 0.25]`:

- On 25% of training draws per structure and epoch, a uniformly drawn supervised site is the centre. The draw is
  deterministic in (seed, epoch, dataset, id), so resumed runs repeat it.
- Residues whose Cα lies within R of the centre are kept. Distances are decoded from the centre row's 16 RBF edge
  channels: the production graphs are exact 20 Å Cα radius graphs with no stored coordinates, so R ≤ 20.
- Every residue edge and site-graph edge with an end outside the crop is masked. Attention masks before the softmax,
  so no path crosses the boundary. Shapes and indices are unchanged.
- Only the originally supervised sites within R_in of the centre are supervised. By the triangle inequality each one
  keeps its full neighbourhood out to R − R_in.
- PINDER complexes get one crop, chosen from the AB branch and applied to both branches; the free branch keeps its
  own partner-removed masks. pKPDB structures are cropped on their own draws.
- Losses are unchanged (per-structure means), so a cropped structure carries the same weight as a whole one.
  Validation is never cropped.

Checks:

- **Unit tests** (`tests/test_production_crop.py`, built with `pkanet.graph.geometry`): the kept set equals true
  Cα distance ≤ R; supervision equals distance ≤ R_in; every supervised site keeps its full R − R_in neighbourhood;
  no edge crosses the boundary; the AB and free branches share one crop; draws are deterministic.
- **Real-store audit** (`production_crop audit`, 200 PINDER and 200 pKPDB training structures): decoded distances
  are exactly symmetric (maximum asymmetry 0.0 Å), never exceed 20 Å, and no edge or site edge crosses a crop.

Supervision is the limiting factor. A 20 Å crop holds about 75 residues, but few titratable sites lie near its
centre:

| R / R_in | Context kept per supervised site | Supervised sites per crop, PINDER / pKPDB | Labels kept on a cropped draw | Residues kept |
|---|---:|---:|---:|---:|
| 20 / 5 | ≥ 15 Å | 1.5 / 1.6 | 3% / 5% | 16% / 20% |
| 20 / 8 | ≥ 12 Å | 2.7 / 2.9 | 5% / 10% | 16% / 20% |
| 20 / 10 | ≥ 10 Å | 3.6 / 3.9 | 7% / 13% | 16% / 20% |

So the inner radius only partly fixes experiment 35's label loss: a cropped draw still loses about 90–97% of its
labels. Larger crops would need coordinates, since the stores only hold the 20 Å graphs. Two arms bracket the
trade-off: 20 / 5 (exp 38's 15 Å context, where 15 Å was level with 20 Å) and 20 / 10 (more labels, 10 Å context).

Runs, all with `--dropout 0.1 --aux-weight 1 --aux-heads burial`, batch 16, width 44, seeds 17 and 29:

- `pilot-{10,50}pct-drop10-bur1-crop20i{5,10}[-s29]`, Isambard jobs 7247491–7247498;
- smoke run `smoke-crop20i5` (job 7247489);
- code on branch `worktree-production-crop`, run from the Isambard worktree `$S/jax-Ka-crop` via
  `_submission/jaxka_gqt_crop.sbatch`.

Baseline: `pilot-{10,50}pct-drop10-bur1[-s29]`. Contrasts come from `production_crop report`: seed means, with 95%
paired bootstrap intervals over the PINDER validation complexes and the pKPDB validation structures. The report
reproduces experiment 07's burial − dropout-only contrasts: −0.0100 (−0.0135, −0.0061) at 10% and −0.0004
(−0.0056, +0.0049) at 50%.

## Results

Selected epochs (report job on the completed runs; per-run metrics from `selection.json`):

| Pool | Arm | Seed | Epoch | Selection | State MAE | Interface paired MAE | pKPDB val MAE | pKPDB val MSE | Benchmark MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | drop10-bur1 (baseline) | 17 | 17 | 0.7195 | 0.4633 | 0.2562 | 0.5537 | 0.7246 | 0.5709 |
| 10% | drop10-bur1 (baseline) | 29 | 18 | 0.7198 | 0.4621 | 0.2577 | 0.5513 | 0.7116 | 0.5743 |
| 10% | crop 20 / 5 | 17 | 19 | 0.7418 | 0.4736 | 0.2682 | 0.5594 | 0.7283 | 0.5764 |
| 10% | crop 20 / 5 | 29 | 20 | 0.7337 | 0.4697 | 0.2639 | 0.5580 | 0.7336 | 0.5750 |
| 10% | crop 20 / 10 | 17 | 18 | 0.7316 | 0.4668 | 0.2648 | 0.5544 | 0.7174 | 0.5678 |
| 10% | crop 20 / 10 | 29 | 20 | 0.7304 | 0.4676 | 0.2628 | 0.5545 | 0.7198 | 0.5727 |
| 50% | drop10-bur1 (baseline) | 17 | 17 | 0.6777 | 0.4311 | 0.2466 | 0.5234 | 0.6459 | 0.5379 |
| 50% | drop10-bur1 (baseline) | 29 | 19 | 0.6746 | 0.4294 | 0.2452 | 0.5189 | 0.6262 | 0.5420 |
| 50% | crop 20 / 5 | 17 | 17 | 0.6703 | 0.4296 | 0.2407 | 0.5246 | 0.6451 | 0.5400 |
| 50% | crop 20 / 5 | 29 | 18 | 0.6751 | 0.4341 | 0.2410 | 0.5245 | 0.6475 | 0.5432 |
| 50% | crop 20 / 10 | 17 | 20 | 0.6723 | 0.4335 | 0.2387 | 0.5204 | 0.6284 | 0.5452 |
| 50% | crop 20 / 10 | 29 | 14 | 0.6717 | 0.4334 | 0.2382 | 0.5276 | 0.6669 | 0.5396 |

Contrasts against the baseline (seed means; 95% paired bootstrap over validation complexes and structures; lower is better):

| Contrast | Selection | State MAE | Interface paired MAE | pKPDB val MSE |
|---|---|---|---|---|
| crop 20 / 5, 10% | +0.0181 (+0.0137, +0.0223) | +0.0090 (+0.0069, +0.0111) | +0.0091 (+0.0057, +0.0123) | +0.0129 (+0.0031, +0.0240) |
| crop 20 / 10, 10% | +0.0113 (+0.0076, +0.0151) | +0.0045 (+0.0026, +0.0063) | +0.0068 (+0.0040, +0.0097) | +0.0005 (-0.0068, +0.0086) |
| crop 20 / 5, 50% | -0.0034 (-0.0076, +0.0007) | +0.0016 (-0.0002, +0.0035) | -0.0051 (-0.0086, -0.0017) | +0.0103 (+0.0023, +0.0186) |
| crop 20 / 10, 50% | -0.0042 (-0.0086, -0.0001) | +0.0032 (+0.0011, +0.0052) | -0.0074 (-0.0108, -0.0043) | +0.0116 (+0.0032, +0.0203) |

The crops behaved as designed in training: 25% of draws were cropped in both datasets, a crop kept 15% of PINDER
and 17–18% of pKPDB residues, and it supervised 1.5 / 1.3 sites (20 / 5) or 3.6 / 2.9 sites (20 / 10) per cropped
draw (PINDER / pKPDB). Epoch times were 62–74 s at 50%, against 62 s for the baseline. Loader wait rose from about
5% to 9–14% of step time, because the crop runs serially after the parallel assembly.

- **At 10% crops hurt every metric.** Selection is +0.018 at 20 / 5 and +0.011 at 20 / 10, with state and
  interface paired MAE worse in both. On the small pool, losing about 22–24% of the labels each epoch costs more
  than the augmentation gives. The arm with more labels (20 / 10) hurts less.
- **At 50% crops trade pKPDB for the interface.** Interface paired MAE improves by 0.005 (20 / 5) and 0.007
  (20 / 10), both intervals excluding zero; this is the largest interface gain at 50% of any variant in
  experiment 07. But pKPDB val MSE gets worse by about 0.010–0.012 (about 2%, significant), and state MAE by 0.002–0.003.
  Selection, which is state plus interface paired MAE, improves only marginally (−0.003 n.s.; −0.004, with the
  interval just excluding zero). Benchmark MAE is unchanged to slightly worse (seed means 0.540 baseline,
  0.542 for both crop arms).
- **20 / 10 vs 20 / 5:** 20 / 10 is better on every metric at 10%, and on interface paired MAE and selection at
  50%. It is slightly worse at 50% on state MAE (+0.003 vs +0.002) and pKPDB MSE (+0.012 vs +0.010). The extra
  supervised sites matter more than the 15 Å vs 10 Å context guarantee.
- The 20 / 10 seed-29 run at 50% selected epoch 14, and its pKPDB val MSE (0.667) is the worst of the 50% runs. So
  part of the pKPDB cost is seed noise; the interface gain is consistent across seeds (0.2387 and 0.2382 against
  0.2466 and 0.2452).

**Verdict:** not a production default. Crops hurt at 10%. At 50% they shift the balance towards the interface at
a cost to pKPDB and state accuracy, which is not a clear net gain. The improvement with pool size (from harmful at
10% to a mixed result at 50%) means a 100% run could differ, but that is untested. If crops are revisited, the
obvious next arm is PINDER-only crops at 50% (20 / 10). That would test whether the interface gain survives without
cropping pKPDB, which is where the pKPDB labels are lost. Validation never crops, so these are clean comparisons of
the training augmentation alone.
