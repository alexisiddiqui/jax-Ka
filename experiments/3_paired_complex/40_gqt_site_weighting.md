# Site-orientation target-range weighting

## Question

Can per-site loss weighting reduce pKa-shift range compression without losing the orientation model's overall validation improvement?

## Frozen design

The experiment keeps the cleaned 5k cohort, frozen train/validation split, backbone-only inputs, 20 A residue and site graphs, 67,725-parameter site-orientation architecture, batch size 8, float32 arithmetic, AdamW settings, and training schedule fixed. ARG remains an unsupervised context token. Seeds are 17, 29, and 43.

| Orientation arm | Per-labelled-site shift-MSE weight |
|---|---|
| Baseline | 1 |
| Balanced | Inverse effective bin frequency, normalized so all four bins have equal expected loss mass |
| Mild | Inverse square root effective bin frequency, normalized to mean 1 and capped at 3 |

The bins are defined by absolute pKPDB shift: `<0.5`, `0.5-1`, `1-2`, and `>=2` pKa. Frequencies use training labels only. Because the existing objective gives equal weight to structures and then averages sites within each structure, effective frequency is computed as `sum(1 / sites_in_structure)` for sites in each bin. This makes the balanced weights correspond to the actual sampler and loss rather than raw site counts alone.

The current 49,709-parameter GQT is a fixed external control using its completed matched three-seed 20 A runs. The three orientation arms use identical sampled structures and batch order within each seed; the report verifies the stored epoch-plan hashes.

## Selection and reporting

Checkpoint selection minimizes the mean of the four validation-bin group-macro MAEs. Overall MAE is not the selection criterion. Patience remains eight epochs with a 0.001 minimum improvement.

The final report records, for each arm and seed:

- overall validation MAE;
- MAE in every target-shift bin;
- pooled predicted-versus-target shift slope;
- predicted-shift standard deviation;
- group-macro MAE by residue type;
- three-seed mean and sample standard deviation.

No test records are read. Runtime artifacts are written under `_runtime/jax-Ka/pkabench/pretraining/gqt-site-weighting-v1`.

## Submitted run

| Stage | Slurm job |
|---|---:|
| Training-frequency registration | 745369 |
| Matched three-arm finite-update smokes | 745371 |
| Smoke verification | 745372 |
| Nine-run, four-GPU production array | 745373 |
| Three-seed report | 745374 |

Registration found effective training-loss frequencies of 0.5183, 0.2190, 0.1552, and 0.1075 from the smallest to largest shift bin. The corresponding balanced weights are 0.4823, 1.1415, 1.6109, and 2.3260; mild weights are 0.7273, 1.1189, 1.3292, and 1.5972. All three finite-update smokes passed and used an identical batch-plan digest.

## Result

All nine orientation runs and the dependent report completed successfully. Values below are mean and sample standard deviation across seeds 17, 29, and 43.

**Default decision:** adopt the site-orientation architecture as the orientation Graph Query Transformer (**oGQT**) for new backbone-only experiments. Use unweighted oGQT as the general-accuracy default and call the compression-aware mild-loss variant **oGQT-mild**. Historical arm identifiers remain unchanged for reproducibility.

| Arm | Overall MAE | Equal-bin MAE | `<0.5` | `0.5-1` | `1-2` | `>=2` | Shift slope | Prediction SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current GQT | 0.5838 +/- 0.0044 | 0.7616 +/- 0.0059 | 0.3606 | 0.4500 | 0.7383 | 1.4975 | 0.6143 | 1.1710 |
| Orientation baseline | **0.5760 +/- 0.0094** | 0.7207 +/- 0.0022 | 0.3847 | 0.4777 | 0.6983 | 1.3222 | 0.6904 | 1.2917 |
| Balanced | 0.6230 +/- 0.0190 | 0.7290 +/- 0.0052 | 0.5128 | 0.5311 | **0.6490** | **1.2232** | 0.7403 | 1.4019 |
| Mild | 0.5981 +/- 0.0126 | **0.7160 +/- 0.0097** | 0.4545 | 0.4985 | 0.6748 | 1.2362 | **0.7455** | **1.3994** |

The orientation baseline reproduces its overall advantage over the current GQT and reduces the large-shift MAE by 0.1753. Balanced weighting reduces the `>=2` MAE by another 0.0990 relative to the orientation baseline, but worsens overall MAE by 0.0470. Mild weighting preserves nearly all of the range benefit: it reduces `>=2` MAE by 0.0860, improves the shift slope from 0.6904 to 0.7455, and increases prediction spread from 1.2917 to 1.3994, while its overall penalty is 0.0221 rather than 0.0470.

Mild weighting gives the best equal-bin score and the best observed compromise. The remaining compression is measurable: the target-shift standard deviation is larger than every model's prediction standard deviation, and the fitted slopes remain below one. Balanced weighting is too aggressive for the overall objective; mild weighting is the candidate to carry forward.
