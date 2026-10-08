# Native dataset export and replay results — 2026-10-05

The existing teacher calculations now provide a versioned experiment 03 dataset
covering all 778 frozen training and 151 validation complexes. No PB calculation
or structure prediction was rerun in this export. Test inputs and labels are absent.

| Quantity | Training | Validation |
|---|---:|---:|
| Selected complexes | 778 | 151 |
| Complexes with usable paired interface labels | 740 | 144 |
| Paired labels before recovery | 41,008 | 8,365 |
| Paired labels after recovery | 44,202 | 8,365 |
| Paired interface labels after recovery | 5,628 | 1,459 |
| Exported native states | 2,291 | 447 |
| States unavailable from teacher failure | 43 | 6 |

The earlier pilot recovery completed 54 of 61 retried states; seven still failed.
Those successes add 3,194 training pairs. Preparation failures were not silently
repaired. The export checked 249,048 unchanged prediction rows, including all
validation predictions, and preserved the frozen uncertainty masks and groups.
All 2,738 successful states passed native units, dimensions, symmetry, sentinel,
site mapping, intrinsic-energy and curve-round-trip checks. There were no rejected
intermediates in version 2. A failed first export attempt remains documented in
the [contract](04_native_data_contract.md).

## Saved-energy reload issue and verified workaround

An ordinary PypKa 2.10.0 MC-only reload reproduced a two-site state exactly. It failed
on the 16-site, histidine-containing AB state of complex `7b31a3bc6f3c8b9f`, despite
using the same energy file, seed, pH grid and MC settings. Maximum discrepancies were
0.99978 in occupancy and 7.42821 pKa units.

The installed `Molecule.loadSites()` calls `reorder_sites()`, moving a C-terminal
site to the end of its chain. In this saved file, each C-terminal site precedes a
lysine on the same residue. Reloading therefore changes the site-object order
without permuting the saved energy arrays. The issue is in saved-energy replay,
not evidence that the original PB/MC labels have these errors.

The replay wrapper restores the exact `all_sites` ordering before Monte Carlo.
On the same failed complex, this reproduces all 73-point curves and all midpoints
exactly: maximum occupancy and pKa differences both zero. The package installation,
energy values, mappings and original predictions were not edited. Failed and
successful replay artifacts are retained. This checks two representative states;
it does not assert that every possible reload edge case has been tested.

## Modelling consequence

Keep the native tautomer states and reference-state convention. The saved energies
are not a scalar binary-site interaction matrix. An unvalidated collapse to one
coupling per residue pair would change the teacher model. The next modelling gate
is a native-state intrinsic baseline and solver contract, or an explicitly tested
reduction. Frozen pKAI remains the main midpoint baseline.

For pair-target losses, both endpoint sites must pass their split's supervision
mask. Masked sites remain present in the complete physical system for MC replay.
Do not treat undefined same-site interaction blocks as training targets; use the
saved owner array. The output keeps mappings and masks rather than discarding
physical context.

## Artifacts and execution

Root: `/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/native-v2`.

- [Combined readiness gate](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/native-v2/readiness.json)
- [Export verification](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/native-v2/verification.json)
- [Paired labels and native charge curves](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/native-v2/paired_sites.parquet)
- [Native-state index with source hashes](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/native-v2/native_state_index.json)
- [Correctly ordered MC replay](/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/tierB/native-v2/replay-aligned/result.json)

Each `complexes/<id>/<state>/native_states.npz` contains microstate free energies,
proton occupancies, counts, ownership and the native interaction matrix. Its
`sites.json` supplies original site keys, intrinsic tautomer pKas, offsets and masks.
Manifest paths resolve the immutable source structures and annotations.

Jobs: initialization 733204; export shards 733206–733237; collection 733239;
MC checks 733238, 733240 and 733241; final readiness 733242. The native-format
regression test passed in 733205. Work ran on compute nodes with 2 CPUs/4 GB per job,
excluding comp1400, through the user-wide 400-core submission cap. No new neural
model was trained in this step. Experimental Set 2 remains awaiting primary
measurement and exact construct verification; no new experimental labels were admitted.
