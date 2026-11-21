# jaxpb-pipeline: geometry + maps speed-ups (levers 1–4) — 2026-11-20 (user approved)

Evidence: results/profile.json (A40, 20 states, frozen a40-2026-11-20-prof; jobs/prof_geomaps.py has per-stage timing code).
PB 15.87 s/state = geometry 5.77 + maps 5.14 (+ SOR etc.). Levers (profile estimates):
1. Remove per-chunk host syncs in geometry (~200 chunks × 3 syncs + dispatch; geometry GPU idle 31%, ~1.85 s/state): keep chunking on
   device (lax.fori_loop / scan over fixed-size chunks, device-side counters/compaction), one host sync per state at most.
2. Candidate slots K 32 → 16 for the burial/gather (max measured cell occupancy 10): keep an overflow counter wired to the existing
   XL/next-class retry; verify occupancy margins on all 104 states + the 1,040-state cycle set.
3. Reuse AB geometry for the A and B states of a triple: A/B probe candidates = AB's own-chain candidates (identical generation) +
   re-filter burial without the partner + chain-only pieces; must keep probe ORDER identical to a from-scratch A/B run (v2 order:
   vertices atom-major, then arcs pair-major) so nearest-probe tie-breaks and maps are unchanged. Driver must schedule triples
   together. If identical order/maps cannot be guaranteed, implement it behind a flag and report the label-level effect instead.
4. Block culling in maps: conservative per-block predictor (distance bounds from block centre/half-diagonal to nearest atom
   sphere / probe reach) classifies whole 4³ blocks as inside / outside; only mixed blocks get the per-point tests. Predictor must
   be conservative (never mis-classify) — proven by bitwise maps.
Optional if cheap: lever 6 (coarse probe table only for boundary projections) and 7 (reach_mask restricted to needed points).

## Closing tests
- Levers 1, 2, 4 (and 6, 7): maps (eps/ion/boundary codes, bidx, surf) and probe sets BITWISE identical to M6 on all 104 states;
  energies/site potentials/sweeps bitwise identical (so L3 unchanged by construction); determinism (2 passes + fresh compile) still
  bitwise; compile count still within budget and flat.
- Lever 3: probe sets/maps for A and B bitwise identical to from-scratch runs on all 34 triples (or, if behind a flag, L3 with the
  flag on within the M6 bars and the difference vs flag-off reported).
- Timing: per-lever and cumulative s/state (geometry, maps, total PB, end-to-end states per A40-hour) on the profile's 20 states and on
  the M6 104-state mix; GPU idle share in geometry.

## Rules
As BUILD-M5/M6: Coulson oc/comp1400, 1 A40, ≤ 12 CPUs, --mem-per-cpu=2G, env ogqt-main-7c3e3e9 + a40site shim; comp1400 CPUs may be
taken by other sessions' jobs (never touch them, nor pkai-scale-train 747977, masif-*, ~/jax-Ka) — queue and wait; frozen copies per
kept run; local src/ is the source of truth (list every changed function); no installs/downloads/envs/git commits; few files.
Results → results/speed.json (harness blocks .md). If the harness hands you back while jobs run, write numbers so far to
results/speed.json (IN PROGRESS) and name the job ids.
