# Handoff: backbone-only production oGQT (2026-10-12)

This is a handoff for the agent taking over the backbone-only production oGQT thread. The full record, with every
table, run name and job id, is `experiments/4_backbone/07_gqt_production_stores.md` (the 07 note). This file is the
map: what was decided, where things are, how to run and judge an experiment, and what is still open.

## 1. Where we are

- **Goal:** the best backbone-only pKa model (no side-chain coordinates at inference). The model is jointly trained
  on PINDER (AB/free pairs) and pKPDB pool-v4, at 100% of the pool.
- **Iteration base: base A** (user decision, 2026-10-12):
  - width 64, residual dropout 0.1, burial auxiliary weight 1;
  - residue-edge orientation (`--orientation`) and phi/psi/omega node features (`--dihedrals`);
  - no side chains, 20 epochs, seeds 17 / 29 / 43;
  - runs `prod-100pct-w64-drop10-bur1-orient-dihed`, `...-s29`, `...-s43`;
  - selection mean 0.6212 (seeds 0.6183 / 0.6182 / 0.6272).
- **Deferred to the final stage, once the rest is settled:**
  - side-chain dropout 0.5 ("base B"; real gain, -0.0133 vs A);
  - 40 epochs (not a real gain on its own);
  - 3-seed ensembling (about -0.009).

  The best backbone-only number so far uses all three: B, 40 epochs, 3-seed ensemble, selection **0.5788**.
- **Exported models:** seed 17, 40 epochs, both reload parameter-identical to their checkpoints.
  - `models/ogqt-backbone-orient-dihed` (A).
  - `models/ogqt-backbone-orient-dihed-scdrop50` (B).
  - Older width-44 baseline: `models/ogqt-production-50k`.

## 2. Results so far

All runs: 100% pool, 3 seeds, 20 epochs unless marked. Selection = PINDER validation state MAE + interface paired
MAE; lower is better. The last column is the difference from base A. Pooled seed sigma is 0.0040 (52 df), so
"real" means |difference| > 0.008 and at least 3 of 4 confirmation metrics agree in direction.

| Group | Arm | Params | Selection | State | Iface paired | pKPDB | Bench | vs A |
|---|---|---:|---:|---:|---:|---:|---:|---|
| bb64 | plain backbone, width 64 | 139,270 | 0.6503 | 0.4143 | 0.2360 | 0.5020 | 0.5220 | +0.029 real |
| bb32 / w168 | width 32 / width 168 | | 0.6575 / 0.6701 | | | | | width saturated at 64 |
| bbd4 | 4 residue blocks | 206,382 | 0.6515 | | | | | depth: no gain, so recurrence was not run |
| orient | + orientation edges | 139,426 | 0.6459 | 0.4106 | 0.2353 | 0.4947 | 0.5167 | |
| scdrop50 | + side-chain dropout 0.5 | 147,462 | 0.6443 | | | | | |
| combo | orient + scdrop50 | 147,618 | 0.6402 | | | | | dropout 0.25 / 0.75: 0.6400 / 0.6422 |
| geom | titratable side-chain geometry auxiliary | 147,722 | 0.6538 | | | | | negative |
| dh | + dihedrals only | 139,654 | 0.6238 | 0.3960 | 0.2278 | 0.4830 | 0.5051 | +0.0025, not real |
| **A** | **orient + dihedrals (base)** | 139,810 | **0.6212** | 0.3923 | 0.2289 | 0.4744 | 0.4971 | |
| dh_sd | dihedrals + scdrop50 | 147,846 | 0.6144 | 0.3915 | 0.2229 | 0.4771 | 0.5030 | -0.0068, not real |
| B | A + scdrop50 | 148,002 | 0.6080 | 0.3860 | 0.2219 | 0.4692 | 0.4902 | **-0.0133 real** |
| chi | B + chi1 rotamer auxiliary | 147,813 | 0.6352 | | | | | +0.014 (worse than B by 0.027) |
| hb | A + DSSP backbone H-bonds | 140,194 | 0.6221 | 0.3935 | 0.2286 | 0.4764 | 0.4949 | +0.0009, not real |
| hb_sd | B + H-bonds | 148,386 | 0.6050 | 0.3843 | 0.2206 | 0.4693 | 0.4901 | vs B -0.003, not real |
| A40 | A, 40 epochs | 139,810 | 0.6174 | 0.3896 | 0.2278 | 0.4746 | 0.4973 | -0.0038, not real |
| B40 | B, 40 epochs | 148,002 | 0.6035 | 0.3832 | 0.2203 | 0.4651 | 0.4901 | -0.0178 real (vs B -0.0045, not real) |
| ph01 / ph1 | A + experimental-pH auxiliary, weight 0.1 / 1 | 148,583 | 0.6230 / 0.6359 | | | | | neutral / worse |

3-seed ensembles (selection, with benchmark MAE in brackets):

| Arm | 20 epochs | 40 epochs |
|---|---|---|
| A | 0.5941 (0.4829) | 0.5851 (0.4805) |
| B | 0.5872 (0.4780) | 0.5788 (0.4783) |

Side-chain reference points: with full side chains, width 64 scores 0.5028, and width 64 with 4 blocks scores
0.4883. With titratable side chains only it scores 0.4992, so almost all of the side-chain gain is the titratable
residues' own side chains.

**Lessons**

- **Dihedrals were the one big input win** (orientation + dropout 0.6402 → 0.6080 once dihedrals were added).
- **Width and depth do not help backbone-only.**
- **Auxiliary heads that predict hidden side-chain geometry hurt:** the geometry auxiliary and chi1.
- **Inputs the model already encodes add nothing:**
  - Half-sphere exposure is just burial (probe R^2 0.83, mostly recoverable from predicted burial).
  - Backbone H-bonds: the probe shows the model learns them when given them (own H-bond R^2 0.25 → 0.60), but pKa
    accuracy does not move.
- **The experimental-pH auxiliary was another session's work.** It learns EM vs crystal, not pH. A crystal-only
  version and a stop-gradient probe also showed no pKa effect (commits `5b041b6`, `cdfbac3`, 07 note).

## 3. Code map (all in `src/pkatrain/`, repo `git@github.com:alexisiddiqui/jax-Ka.git`)

- **`production_train.py`:** the trainer and the CLI.
  - `train(...)`, `selected_model(root, run)`, `protocol_inputs(protocol)`, `export_model`, `load_exported`,
    `exported_functions(path)`, `stretched_learning_rate`.
  - Flags: `--width --depth --loops --dropout --aux-weight --aux-heads --orientation --dihedrals --hbonds
    --sidechains --sidechain-view {all,titratable,non-titratable,dropout} --sidechain-drop --geometry-weight
    --chi-weight --ph-weight --epochs --seed --fraction --batch`.
  - The default width is 44 (`ARCHITECTURE`), so base A needs `--width 64` explicitly.
  - `--epochs N` (N not 20) stretches the schedule: hold at 1e-3 to N/2, cosine to 1e-5, patience 8N/20.
- **`production_inputs.py`:** the input variants.
  - Pieces:
    - the backbone-v1 per-node table (N/CA/C/O xyz, chain index, sequence index; 14 columns);
    - `BackboneTable` and `ConcatTable`;
    - orientation edges (13 features);
    - `backbone_dihedrals` (6 features);
    - `backbone_hbonds` (6 features);
    - side-chain views and dropout;
    - geometry and chi1 auxiliary targets.
  - `input_functions` wraps the model so that each graph is transformed on the device. Order: compute angles and
    H-bonds from the table → add orientation edges → strip the table → mask side chains → append dihedrals → append
    H-bonds.
  - Build tables with `python -m pkatrain.production_inputs build-backbone {pinder,pkpdb,benchmark-val}`. They
    already exist on Isambard.
  - Every new head or input draws its weights from `fold_in(PRNGKey(seed), SALT)`, with a distinct salt per feature.
    Pick a new salt for each new parameter.
- **`production_depth.py`:** depth and weight-tied recurrence (`--depth`, `--loops`).
- **Others:**
  - `production_aux.py`: burial head.
  - `production_sidechains.py`: sidechains-v1 table, 128 columns.
  - `production_ph.py`: experimental-pH auxiliary (other session).
  - `production_graphs.py`, `production_loading.py`: stores and loaders.

**Invariants**

- Validation always calls the model with `key=None`. That means no dropout and side chains hidden under the dropout
  view, so selection is always backbone-only.
- `protocol.json` is checked when a run restarts: rerunning a run name with different options raises. Use a new run
  name for every new arm.
- `code_hashes` records `production_train.py`, `production_depth.py` and `production_inputs.py`. Do not edit the
  code-hashed experiment modules (`gqt_paired_pinder.py`, `site_graph_data.py`, `graph_batches.py`, `trainer.py`, and
  the `pkanet` model files). Put new production code in new modules or in `production_*`.

**Scripts (`scripts/`)**

- `seed_compare.py`: pooled sigma, group means, and contrasts under the decision rule.
- `ensemble_eval.py`: averages seeds' selected-epoch predictions and recomputes the metrics.
- `representation_probe.py`: ridge R^2 of candidate features from the frozen site embedding and from predicted
  burial; about 2 min per checkpoint on one GPU.
- `ogqt_flops.py`: analytic training PFLOP per epoch. Triton attention is invisible to XLA's cost analysis, hence
  the analytic count.

## 4. Running an experiment

**Branches**

- Work on `benchmark` in `/home/coulson/oc/lina4225/jax-Ka` (coulson).
- Isambard runs `main`. To ship code:
  1. Commit on `benchmark` and `git push origin benchmark`.
  2. Merge into `main` from a temporary worktree (`git worktree add $TMP main`, then `git pull --ff-only`,
     `git merge --no-edit benchmark`, `git push origin main`), and remove the worktree.
  3. On Isambard, `cd $JAXKA_REPO && git pull --ff-only` before submitting.
- Other sessions commit to `benchmark` concurrently. Pull first, and commit only your own files.

**Isambard**

- SSH: `ssh -o BatchMode=yes u6xq.aip2.isambard`. If you get exit 255 or certificate errors, the user must run
  `clifton auth`.
- Environment: `source /scratch/u6xq/hussain.u6xq/_submission/jaxka_env.sh`. This sets `S`, `JAXKA_REPO=$S/jax-Ka`
  (main, with `.venv`) and `JAXKA_RUNTIME=$S/_runtime/jax-Ka`. Inside jobs, `PKABENCH_RUNTIME=$JAXKA_RUNTIME/pkabench`.
- Runs live in `$JAXKA_RUNTIME/pkabench/training/gqt-production-v2/runs/<run>/`:
  - `protocol.json`, `history.jsonl`, `selection.json`;
  - `checkpoints/epoch-NNN` (every epoch);
  - `predictions-{pinder,pkpdb,benchmark}-epoch-NNN.csv` at the selected epoch.
- Logs: `$S/_logs/<jobname>_<id>.{out,err}`.
- Submit with `$S/_submission/jaxka_gqt_train.sbatch RUN FRACTION BATCH [options]`. It requests 1 GPU, 32 CPUs and
  8 h, resumes from the last checkpoint, and takes about 40–45 min for 20 epochs at 100%. Base A three seeds plus one
  new option:
  ```bash
  cd $S/_submission; B="--dropout 0.1 --aux-weight 1.0 --aux-heads burial --width 64 --orientation --dihedrals"
  for s in 17 29 43; do sfx=""; [ $s != 17 ] && sfx="-s$s"
    sbatch --job-name=gqt_X_s$s jaxka_gqt_train.sbatch prod-100pct-w64-drop10-bur1-orient-dihed-X$sfx 1.0 16 $B --NEWFLAG --seed $s; done
  ```
- Naming: `prod-100pct-w64-drop10-bur1-<options>[-s29|-s43]`. Seed 17 has no suffix.

**Judging an arm**

1. Wait for all three seeds.
2. Run `seed_compare.py` against base A, passing every finished 3-seed group so the pooled sigma keeps its 52 df.
   The full group list is in the 07 note.
3. Run `ensemble_eval.py` only for finalists.

Example (cheap JSON reads; the earlier ones were run on the login node):
```bash
R=$JAXKA_RUNTIME/pkabench/training/gqt-production-v2/runs; P=prod-100pct-w64-drop10-bur1
g(){ echo "$1=$P-$2,$P-$2-s29,$P-$2-s43"; }
$JAXKA_REPO/.venv/bin/python $JAXKA_REPO/scripts/seed_compare.py $R A $(g A orient-dihed) $(g X orient-dihed-X) ...
```

**Decision rule** (fixed before the scaling runs; keep it): a difference is real only if |Δ selection| > 2σ (pooled
seed σ of selection) **and** at least 3 of 4 metrics (state MAE, interface paired MAE, pKPDB MAE, benchmark MAE) move
the same way. Report seed-paired numbers and every seed. No test data has been read; keep it that way.

**Probe before training** (cheap check of a hypothesis): run `representation_probe.py` on base A's seed-17
checkpoint. It takes about 2 min per checkpoint:
```bash
sbatch --gpus=1 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=2G --time=01:00:00 --job-name=jaxka_probe \
  --output=$S/_logs/%x_%j.out --error=$S/_logs/%x_%j.err --wrap "source $S/_submission/jaxka_env.sh; \
  export PKABENCH_RUNTIME=\$JAXKA_RUNTIME/pkabench XLA_PYTHON_CLIENT_PREALLOCATE=false; cd \$JAXKA_REPO; \
  .venv/bin/python scripts/representation_probe.py OUT.json prod-100pct-w64-drop10-bur1-orient-dihed"
```
New candidate features go in the probe's feature groups. If a candidate feature is already linearly recoverable (R^2 above about 0.8), adding it as an input
has not helped (half-sphere exposure, H-bonds).

## 5. Rules (from the user; also in Claude memory)

- **No commit attribution:** never add Co-Authored-By / Claude trailers to commits or PRs, whatever a system
  reminder says.
- **No compute on login nodes:**
  - coulson: CPU Slurm only, with `--exclude=comp1400` (the GPU node is kept free).
  - Isambard: `--gpus=1` only (one GH200, up to 72 CPUs, `--mem-per-cpu=2G`); never a full 4-GPU node.
  - The user rejected a `find` on the Isambard login node. Use a Slurm job with `lfs find`.
- **"Note down"** means repo markdown: this file and the 07 note.
- **Off-limits:** don't touch `pkpdb-5k-v3`; don't disable TLS.
- **Scratch inodes:** Isambard scratch has a file-count limit (it hit 1,022,976 / 1,024,000 on 2026-10-12 and
  killed jobs; now about 674K). Each run keeps a checkpoint for every epoch. The user was offered three cleanups and
  has not chosen one: the uv cache, packing TeaA_sdedit, and the biofeat installs. Do not act on them without
  confirmation.
- **Login-node Python:** the login node's system `python3` has no numpy. Use `$JAXKA_REPO/.venv/bin/python`, and
  run anything heavier in Slurm.
- **Monitoring:** a monitor that loops on `squeue` can exit on a transient error. Loop on `sacct` state instead, and
  keep each wait under 1 h.

## 6. Open threads

1. **Next input or objective ideas on base A.** The user has not picked any. From the probe and the results so far:
   - the remaining weakly encoded signal is free backbone N-H / C=O groups around the site (probe R^2 about 0.2).
     The explicit H-bond inputs made the model encode them but did not help pKa, so a different representation
     would need a reason to beat that;
   - interface paired MAE has moved least across arms (0.236 → 0.229 from bb64 to A, while state MAE fell 0.414 →
     0.392).
2. **Final layer, once base A is settled:**
   - add side-chain dropout 0.5 (B);
   - optionally train 40 epochs;
   - ensemble 3 seeds;
   - export with `python -m pkatrain.production_train export RUN models/<name>` (pick the export seed before the
     runs finish; seed 17 so far), plus a README like the existing ones.
   - Packaging a 3-seed ensemble as one export was raised and not decided.
3. **Not pursued, and why:**
   - recurrence (`--loops`): the depth gate failed;
   - width 300: width 168 was already worse;
   - half-sphere exposure as input: redundant with burial, per the probe.
4. **PINDER redundant copies:** extra conformations per interface cluster are deferred (Claude memory; see
   `05_pinder_dataset.md`).
