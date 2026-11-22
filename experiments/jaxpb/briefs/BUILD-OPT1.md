# jaxpb optimisation, phase 1: maps + setup memory/speed + quick fixes (2026-11-21, user approved)

You are implementing phase 1 of the merged plan in reviews/COMPARISON.md (read it, plus reviews/codex-astra.md and
reviews/gemini-neon.md for detail; evidence in results/profile2.json, bench.json, speed.json, M6.json). This folder is a
git working copy with a baseline commit: edit src/ freely, do NOT commit (the user reviews `git diff`). Stop after phase 1
and report; phases 2-5 (XLA SOR, lane packing, CPU MC, geometry) are NOT in scope.

## Changes (in this order; keep each separable and behind no flags unless stated)
A. Quick fixes:
   - pb.py ~378: bound the class-bump loop; past the last class raise OverflowError (driver logs failed_pb), never IndexError.
   - pb.py ~432: remove the redundant D2H `spec_h = np.asarray(j["spec"])` sync if unused (or use the host `sp`);
     gate timing-only block_until_ready calls behind a timing flag only if it doesn't change outputs.
   - api.py ~583: stop accumulating every state's timing dict in summary["per_state"] (keep per-shard timing in the shard
     files + bounded aggregates) so driver host memory does not grow with state count.
   - api.py ~174: make MemTrack lazy (no jax.devices() at import).
B. Maps (maps.py): replace the dense [ncell^3, K] cell table with per-cell start/stop ranges over the sorted order
   (CSR); reconstruct candidates per query tile; byte-budgeted query/reach tiles that account for B; padded sites do no
   work (today _site_layout repeats the last real site). Keep per-site origins (acent), the exact candidate order,
   first-candidate tie rule, distance expression and overflow detection. CPU: support a smaller internal site group
   (e.g. B=16 processed as 4 groups of 4) selected by backend/platform, fixed sizes only (flat compile count).
C. Setup (pb.py setup1/setup2/coulomb_bc/interp): evaluate boundary values only on the fixed face-point list
   (38,402 for n=81), scatter into a zero grid; same for the focused-grid interpolation. Keep the charge-reduction order.
   Build only what the selected SOR backend needs (XLA coefs vs Pallas codes) via a finite static variant; release
   consumed batch buffers.

## Hard gates (all must pass; if one fails, localise and fix, or revert that change and report)
1. Bitwise vs baseline on the A40, Pallas backend: energies, site potentials, sweep counts on the 104 L3 states
   (M6 inputs; see results/M6.json for shard paths) and on the 20 bench states (reference
   tierB/jaxpb-pipeline/bench/gpu/energies_gpu.npz). Also XLA backend bitwise vs the baseline XLA backend on >= 12 states.
2. CPU: bitwise vs the baseline CPU outputs on the 5 profile2 CPU states (tierB/jaxpb-pipeline/prof2/cpu/, bench/jaxcpu/).
3. Determinism: 2 passes + a fresh process bitwise identical (A40).
4. Compile count flat (no per-shape recompiles; report counts vs baseline 40/33).
5. CPU unit tests pass: tests/test_*.py that are pytest tests (test_api, test_assemble, test_records, test_titrate) on a
   Coulson CPU node (env has no pytest: use PYTHONPATH with the solver env's pytest as in
   tierB/jaxpb-pkgtest/pytestshim, or run their __main__ modes).
Report per change: s/state (geometry, maps, setup, SOR, total PB) on A40 Pallas, A40 XLA, CPU (5 states, 12 CPUs), peak
GPU bytes, peak host RSS (CPU arm especially: target well under 12 GB at --mem-per-cpu=2G x 12), vs baseline
(results/profile2.json numbers).

## Compute rules (strict)
Coulson only, via `ssh Coulson-Chemistry`; base ~/_runtime/jax-Ka/pkabench/. A40: partition oc, node comp1400,
--gres=gpu:a40:1, <= 12 CPUs, --mem-per-cpu=2G; CPU: partition amd96 as the bench (jobs/coulson_cpu.sbatch), <= 12 CPUs,
--mem-per-cpu=2G (4G only for a baseline comparison run that needs it). Env ~/_runtime/jax-Ka/pkabench/envs/ogqt-main-7c3e3e9
plus the A40 shim jobs/a40site/sitecustomize.py (see jobs/a40.sbatch, jobs/bench.sh, jobs/prof2.sh for working patterns).
Copy code to a NEW frozen folder per run: tierB/jaxpb-pipeline/opt1-<tag>/; outputs tierB/jaxpb-pipeline/opt1/. Never
modify existing folders there. Never touch, cancel or inspect-and-alter jobs you did not submit (abag-fv-*, pkai-scale-*,
cov-*, masif-*, pool5-*); never touch ~/jax-Ka on Coulson; no login-node compute (submit sbatch jobs; light file
inspection only); no installs, downloads, new envs, git commits or pushes anywhere. Keep file counts low.

## Output
- src/ edits in this folder (uncommitted), any new job scripts under jobs/.
- results/opt1.json: per change and cumulative: gate results, timings, memory, compile counts, job ids, frozen paths.
- OPT1-REPORT.md: <= 400 words: what changed (functions), gate table, timing/memory table vs baseline, anything reverted
  and why, open risks, what phase 2 should do first.
