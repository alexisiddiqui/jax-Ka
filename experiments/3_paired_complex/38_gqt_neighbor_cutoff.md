# GQT global neighbour-cutoff screen

Authorized 2026-10-08. Compare strict-backbone and side-chain 50k GQT models
with global Cα graph cutoffs of 5, 10, 15, 20, and 25 Å. Every residue and every
eligible query remains in the graph; the cutoff applies to every encoder layer
in both training and validation. There is no query-centred cropping. An
all-to-all residue graph was considered and then explicitly dropped because it
would make this fast screen substantially more expensive.

The 25 Å source graphs are rebuilt from the resolved structures using the exact
frozen node order. All five arms use the same 16-channel 0–25 Å Gaussian RBF
basis. Smaller arms remove edges above their registered cutoff and recompute the
two-angstrom cosine taper. This keeps node features, labels, queries, optimizer,
and distance representation matched across the cutoff comparison.

Every input type and cutoff is first screened at seed 17. For each input type,
the lowest validation-MAE cutoff and the 20 Å reference are then confirmed at
seeds 29 and 43. Duplicate confirmations are removed if 20 Å wins. Runs use 20
maximum epochs, batch size eight, full float32, indexed Triton encoder
attention, AdamW with weight decay `1e-4`, and no dropout. The learning rate
remains `1e-3` through epoch 10 and then decays to `1e-5`. Selection uses
validation group-macro MAE with patience eight; no test data are read.

The staged experiment has 14–18 training runs rather than the full 30-run
factorial. With four A40s and sub-30-minute runs, the initial screen takes three
GPU waves and confirmation one or two further waves.

Runtime location: `pretraining/gqt-neighbor-cutoff-v1-triton`.

| Stage | Slurm job |
|---|---:|
| One-structure rebuild gate (passed) | 745266 |
| Full 25 Å graph rebuild, 48 CPUs | 745267 |
| Exact backbone/side-chain mmap conversion | 745289 |
| Cutoff invariant tests and registration | 745290 |
| Ten finite-update GPU smokes | 745292 |
| Smoke-result gate | 745293 |
| Ten-run seed-17 screen, maximum four A40s | 745294 |
| Winner selection and confirmation plan | 745295 |
| Four-to-eight confirmation runs | 745296 |
| Aggregate report | 745297 |
