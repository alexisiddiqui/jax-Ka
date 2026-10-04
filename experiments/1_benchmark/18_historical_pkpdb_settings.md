# Historical pKPDB configuration: evidence amendment (2026-10-04)

This records the user's supplied investigation of raw `fill.py`, `const.py`,
server routing and PypKa git history. The reported scratchpad `h/` files and
approximately 30 historical commits have not been independently re-audited in
this amendment. The statements below retain that provenance; they are not an
owner-provided settings export.

## Decision

Historical per-entry equivalence remains unresolved. Shared `PKPDB_PARAMS`
constants attached to responses are weak evidence, not independent confirmation
of deposited simulation settings. In particular, our earlier claim that
`ser_thr_titration=False` matches historical pKPDB is not established.
False remains our explicit current benchmark choice. No teacher parameters,
predictions, frozen inputs, masks or scoring results change in this amendment.
Current-label benchmarking does not depend on resolving historical equivalence.

## Reported source evidence

The deposited `fill.py` call sets epsin=15, ionicstr=0.1,
pbc_dimensions=0, pH="-6,20" and convergence=0.01 (DelPhi maxc).
Other settings depend on the PypKa version. DelPhi and Monte Carlo dictionaries
are stored whole; PypKa parameters retain ffID, ff_family, ffinput, clean_pdb,
LIPIDS, keep_ions, ser_thr_titration, cutoff, slice, CpHMD_mode and version.
Temperature and hydrogen optimisation are omitted.

The reported 2020–2024 defaults are temperature 298 K, scaleM=4, maxc=0.01,
solvent dielectric 80, grid 81, G54A7, MC steps 200000, equilibration 1000,
seed 1234567. Internal dielectric defaults to 20, requiring the explicit 15
override. SER/THR defaults reportedly were true except 2020-12-15–2021-01-05.
The pdb2pqr_h_opt option arrived in 2.1.2 (2021-04-21); earlier optimisation
was always on.

| Parameter | Generation call plus reported defaults | Shared server constant |
|---|---|---|
| pH range | −6 to 20 | 0 to 12 |
| maxc | 0.01 | 0.1 |
| scaleM | 4 | 2.0 |
| SER/THR | Usually true by default; actual override unresolved | False |
| Version | Stored per settings record | 2.1.0 |

The user reports that routes_direct.py attaches the same constants to every
response and that their file history begins with a 2025-01-25 refactor.
Live responses therefore do not resolve these conflicts.

The deposited `pk` is the first stored pKhalf, linearly interpolated on a
0.25-pH grid, absent when no crossing exists. Deposited `dpk` subtracts PK_MOD:
ASP 3.79, CTR 2.90, CYS 8.67, GLU 4.20, HIS 6.74, LYS 10.46,
NTR 7.99, TYR 9.59. This is not our bound-minus-free shift definition.
PK_MOD has no SER/THR entry: the user's inference is that a finite SER/THR pK
would cause a KeyError, suggesting titration was disabled. This is untested
and does not establish the actual per-entry setting.

Random entry selection, registration dates and deduplicated settid records
permit multiple settings/version groups within a database release. Reported
label-relevant changes include DelPhi binaries (2021-01-19), incomplete termini
and GLY starts (2021-02-10/11), numbering/NMR/terminal residue handling
(2021-03-02), pH ranges and numerical accumulation (2.4.0, 2021-09-21),
and insertion codes/free terminal CYS (2.8.0, 2022-03-07).

## Comparison and remaining provenance

Our current teacher uses 298.15 K, pH −2 to 16 at 0.25 spacing,
PypKa 2.10.0, explicit hydrogen optimisation and SER/THR off. The reported
historical defaults support several shared parameters but do not establish
per-entry equivalence. The 298 versus 298.15 K difference is explicit;
its numerical effect has not been measured. pH range, version and preparation
differences must not be attributed solely to missing residues.

The user reports different release/subset sizes across the original paper
(12M pKa, 120k structures), website (about 10M, 70% of PDB), pKAI
(about 3M in methods versus 6M/50k structures in abstract), and later server
paper (200k structures/20M pKa including over 100k AFDB structures).
These do not identify our downloaded CSV's snapshot or PDB-only scope.
That scope and the original entry filters remain to be checked from the
actual dataset provenance; AFDB use remains deferred to backbone-only work.

The owner export is the strongest available record of stored settings, but
cannot recover omitted temperature/optimisation values without accompanying
version and run provenance. The read-only
[SQL export](export_pkpdb_settings.sql) now includes a whole-database census
with complete stored dictionaries, simulation and distinct-structure counts,
and date ranges. It has not been run. Settings census plus the existing
per-entry mapping are needed to assign provenance to compared labels.
