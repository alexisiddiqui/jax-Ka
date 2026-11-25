# jaxpb M7: converged PB mode + driver memory trim + write-up (2026-11-25, user approved "implement M7")

Context: phases 1-4 are committed here as 31b30bc (phase 4 = CPU maps/geometry; jax-Ka jaxpb is at 2d8aa19, phase 4 not yet ported).
Do NOT commit; the user reviews `git diff 31b30bc`. Read DESIGN.md (sections 4 and 5, row M7), results/M6.json (design block,
`host_note`), results/opt4.json only as needed, src/jaxpb/{api.py,pb.py,sor.py}.
The CPU path is a fallback, so correctness and the GPU path come first; CPU only has to stay correct (tests pass, bitwise-unchanged
default mode).

## Background
The default PB mode "delphi" reproduces DelPhi's SOR exactly: zero start, red/black, relaxation factor from the spectral radius,
stop when max potential change < 0.01 kT/e at 10-sweep checks, 500-sweep cap. That stop is loose: PyPKA's labels (and so ours) carry
a solver error, previously estimated at 0.1-0.3 pK. Only "delphi" mode has ground truth (PyPKA). `Config.pb_mode` currently raises
NotImplementedError for anything but "delphi" (api.py:88).

## Deliverables
1. `pb_mode="converged"`: solve the same grids and coefficients to a tight tolerance, so the result is (close to) the
   exact discrete solution rather than DelPhi's stopped iterate. Choose by measurement between (a) the existing red/black SOR with
   a tight stop criterion on an equation-residual (not max-change) and a higher sweep cap, and (b) a PCG solver (`jaxpb.pcg`, f64
   accumulation) as the design says. The primary requirement is that the converged result is trustworthy and independent of the
   stopping heuristic; the secondary requirement is that it fits the existing lane pool/size-class design with flat compile counts.
   Keep "delphi" mode bit-for-bit unchanged (default). Mode goes into provenance and cfg_hash; label shards from different modes must
   never be mixed or resumed into each other (cfg change already refused; verify).
2. Reference and closing test: an independent f64 reference solve (PCG, tight tolerance, host or device) on real grids. Report on the
   5 OPT2 states and >= 20 bench states: converged-mode vs f64 reference (max |dphi|, energies esolvM/esolvS, g, W) and delphi-mode vs
   the same reference (this measures DelPhi's solver error directly), per solve type (coarse, focused, model).
3. Effect on labels: run the titration on converged-mode energies for the L3 set (104 states, A40) and report pKa/shift changes vs
   delphi mode (max, MAE, number of curve midpoints that move) against the PyPKA teacher and against MC seed noise (use the M6 noise
   reference). State clearly how large DelPhi's solver error is in pK terms; this is the main scientific result of M7.
4. Cost: converged-mode s/state and extra sweeps/iterations vs delphi mode on the A40 (and CPU for a few states); compile counts flat.
5. Driver memory trim (known item from results/M6.json `host_note`): the in-memory run summary grows with number of states
   (per-state timing list; +0.6 GB between 104 and 1,040 states, so unbounded for 30k). Make the summary memory bounded: stream the
   per-state/per-batch timings to the existing event log or a file and keep only aggregates in memory. The returned summary and
   run_<pid>.json keep the same keys needed by existing tests/results (document anything that moves). Resume and shard bytes
   unchanged. Verify with a long synthetic run that host RSS no longer grows with states.
6. Write-up: a concise pipeline README/paper-style document `JAXPB-WRITEUP.md` (<= 1,200 words) covering design, modes, parity
   bar and results across M0-M7 and the optimisation phases (timings: A40 ~344 states/GPU-h; CPU numbers from phases 3-4),
   CPU being a fallback, known limitations, and how to reproduce. Use the numbers in results/*.json; do not invent any.

## Gates
1. `delphi` mode outputs bitwise identical to 31b30bc on the 20 bench states (A40 and CPU) and on L3 (A40) energies; flat compile
   counts; determinism (two runs + fresh process); existing tests pass; new tests for converged mode (toy lattices with analytic
   solution, 1-site, convergence monotone in tolerance, mode in provenance/cfg_hash, shards not mixable).
2. converged mode: determinism, flat compiles, memory bounded, residual below the stated tolerance on all solves in the test sets
   (report the residual definition).
3. Memory trim: host RSS flat over a >= 2,000-state synthetic or replayed run (report numbers).
Report what you could not verify rather than guess.

## Compute rules (strict, unchanged)
Coulson only via `ssh Coulson-Chemistry`; CPU partition amd96 and A40 (oc, comp1400, gpu:a40:1), <= 12 CPUs, --mem-per-cpu=2G, env
envs/ogqt-main-7c3e3e9 + jobs/a40site shim. New frozen folders tierB/jaxpb-pipeline/m7-<tag>/, outputs tierB/jaxpb-pipeline/m7/.
Never modify existing folders or touch others' jobs (pkai-scale-*, abag-fv-*, cov-*, masif-*, pool5-*) or ~/jax-Ka; no login-node
compute, installs, downloads, commits or pushes. Keep file counts low (inode quota). Isambard credits are exhausted: Coulson only.

## Output
src/tests edits (uncommitted), jobs/m7*, results/m7.json, M7-REPORT.md (<= 400 words: design choice and why, gate table, DelPhi
solver error in pK terms, cost, memory trim result, open risks), JAXPB-WRITEUP.md. Stop after this.
