# OPT3B

Pair energy is `(E[na]-E[oa])+(E[nb]-E[ob])`: ordered site differences, exact unchanged-site cancellation. Ref/Pallas uses two-nonzero masked differences; CPU uses scalar picks and ≤12 host threads. Deleted runtime/domain eligibility gates. Auto: supported GPU→Pallas; CPU→cpu; otherwise ref.

Ref/Pallas skips unchanged-site evaluation; CPU evaluates eagerly. RNG, moves, BLOCK, accept rule and E/X/count updates are unchanged.

| Gate | Result |
|---|---|
| CPU/new ref | Bitwise: 20/50k + 5/200k (counts/curves/pKa/shard fields) |
| Determinism/API | Repeats, fresh process and API pass |
| Compiles | Five warm executables; zero later |
| Tests | Pass: nst11, Mp23/2048, single-site, partial pairs, brute/interpreter/API |
| Speed/rebaseline | Pass |

Change / old-code seed noise:

| Arm | Differing curve counts | max ΔpKa |
|---|---:|---:|
| CPU 20/50k | 0 / 65522 | 0 / 0.0671164 |
| CPU 5/200k | 1 / 9324 | 0 / 0.0200468 |
| A40 20/200k | 0 / 71700 | 0 / 0.0435366 |
| A40 L3 104/200k | 0 / 316700 | 0 / 0.164512 |

A40 raw-count changes: bench 10, L3 18.

L3 teacher MAE: old **0.00843788**, new **0.00843788**, second seed **0.00839324**. Drift lies within seed noise.

Pallas/GPU ref agree bitwise on 20/50k + 5/200k. Pallas/CPU differs in 22 raw/2 curve counts at 50k; 16/0 on 5/200k; 147/25 on 20/200k. Locations are in JSON.

Remaining portability limits: invexp FMA/division. Other reductions are exact or sequential; arguments are in JSON.

200k MC seconds/state: CPU1 **197.47**, CPU12 **33.37**, versus phase 3 **238.5 / 41.7**. A40 bench **4.103→3.856**; L3 **1.873→1.838**. Non-exclusive allocations; CPU1 exposes two SMT threads, matching phase 3.

Direct-indexing lowering, four-pick overhead and scalar CPU branching were rejected. Saved energies; audits/evidence accompany `results/opt3b.json`. Uncommitted; stopped after OPT3B.
