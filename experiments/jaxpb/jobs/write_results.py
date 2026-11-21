"""Assemble results/M0.json, M1.json, M2.json from pulled run outputs (local helper).
usage: python jobs/write_results.py <pulled_dir> <results_dir>
<pulled_dir>: manifest.json (M0 shards), cpu_cachecheck.read.json, m1cpu/<sid>/m1_main.json, m2cpu/m2.json"""
import sys, os, json, glob

P, OUT = sys.argv[1], sys.argv[2]
L = lambda p: json.load(open(os.path.join(P, p))) if os.path.exists(os.path.join(P, p)) else None
COUL = "Coulson:~/_runtime/jax-Ka/pkabench/tierB/jaxpb-pipeline"
ISA = "Isambard:$R/audits/jaxpb-pipeline"
SUB = "$S/_submission"
SB = "jobs/isambard.sbatch"
man = L("manifest.json")
cc = L("cpu_cachecheck.read.json")
gpu_blocked = "Isambard sbatch rejects every job (AssocGrpCPUMinutesLimit: project CPU-minute allocation used up); GPU runs staged, not run"

m0 = {
    "milestone": "M0", "date": "2026-11-19",
    "status": "partial: goldens (a)+(b), compile counter and CPU cache-tarball check done; GPU closing test (v2 reproduction, Pallas cache) staged -- " + gpu_blocked,
    "goldens_pilot5": {"what": "5 pilot states: StateRecords from jaxpb-pilot-v1 rec.npz + saved mc-energies/result/gate + v2 outputs (v2_*), MC order",
                       "sha256": "ec537e59aa762cd0457b9b1015925882cccde53f6af5d9e3ba20bd4c6808e53e",
                       "paths": [f"{COUL}/m0/pilot/goldens_pilot5.npz", f"{ISA}/m0/goldens_pilot5.npz"]},
    "goldens_m0": {"what": "99 val states (33 complete AB+A+B triples, 11 Ab/Ag + 22 general complexes, pilot complexes excluded) recorded "
                           "with prep_pypka.py record on Coulson amd96 (DelPhi inputs/outputs, logged rho/sweeps/est. iterations, "
                           "mc-energies, result, pKmod, site type, chargediff, occ, MC order, pairs)",
                   "summary": man and man["summary"], "shards": man and [{k: s[k] for k in ("path", "sha256", "n_states", "bytes")} for s in man["shards"]],
                   "sha256_checked": "Coulson, Mac, Isambard (sha256sum -c OK on all three)",
                   "paths": [f"{COUL}/m0/shards/", f"{ISA}/m0/shards/", f"{COUL}/m0/rec/<sid>.npz (per state, with 2 full DelPhi log texts)"],
                   "gate": "re-run mc-energies.json bitwise equal to the campaign file on 99/99 states; pKa max diff 0.0",
                   "records_unit_test": "tests/test_records.py on the 3 shards: pass (99 states, 28568 tautomers; site order == mc-energies all_sites)"},
    "compile_counter": {"module": "src/jaxpb/compile_budget.py (CompileCounter: jax.monitoring backend_compile events minus persistent-cache hits, names from jax_log_compiles)"},
    "cache_tarball_check_cpu": cc and {"jax": cc["jax"], "device": cc["device"], "second_process_compiles": cc["compile"]["n_compiled"],
                                       "cache_hits": cc["compile"]["n_cache_hits"], "pass": cc["pass"],
                                       "note": "CPU stand-in (Coulson, JAX 0.11.1); Pallas in interpret mode lowers to plain XLA, so Pallas executable caching is NOT established here"},
    "closing_test_gpu": {"status": "staged, not run (" + gpu_blocked + ")",
                         "command": f"cd {SUB}/jaxpb-pipeline-m0 && sbatch -J jpp-m0gpu -t 01:00:00 {SB} {SUB}/jaxpb-pipeline-m0 bash jobs/m0_gpu.sh",
                         "does": "Pallas Triton cache write/read (2 processes); v2 runner re-run on 5 pilot states under CompileCounter vs saved v2 outputs (fresh cache), then 2 states from the tarball in a new process",
                         "outputs": f"{ISA}/m0/{{cachecheck.*.json, v2repro/v2repro.run1.json, v2repro/v2repro.run2cache.json}}"},
    "energy_track_requests": "all 8 handled, see results/pb-track-replies.json",
}

# ---------------------------------------------------------------- M1
m1s = {}
for f in sorted(glob.glob(os.path.join(P, "m1cpu", "*", "m1_main.json"))):
    d = json.load(open(f)); sid = f.split(os.sep)[-2]
    m1s[sid] = dict(summary=d["summary"], classes=d["classes"], budget=d["budget"], overflow_test=d.get("overflow_test"),
                    pass2_identical_sweeps=d.get("pass2_sweeps_identical"), pass2_max_abs=d.get("pass2_vs_pass1_max_abs"),
                    compile_pass1_names=d["compile_pass1"]["names"])
m1 = {
    "milestone": "M1", "date": "2026-11-19",
    "status": "code complete; CPU stand-in tests on Coulson; GPU closing test staged -- " + gpu_blocked,
    "files": ["src/jaxpb/records.py", "src/jaxpb/geometry.py", "src/jaxpb/maps.py", "src/jaxpb/sor.py", "src/jaxpb/pb.py",
              "src/jaxpb/compile_budget.py", "tests/test_pb_m1.py", "tests/test_records.py", "jobs/m1_gpu.sh"],
    "criteria": {"energies_rel": "<= 1e-6 vs v2", "sitpot": "<= 1e-6 (relative to the tautomer's potential scale; abs reported)",
                 "sweeps": "100% equal to v2", "compiles": "<= compile_budget.budget(classes), 0 on a second pass",
                 "overflow": "Ka 24 / Kp 768, forced overflow -> automatic XL retry, identical outputs"},
    "cpu_standin_per_state": m1s,
    "probe_centres": "geometry_state bitwise equal (same order) to v2 probe_centres_gpu on d0d5 B (222,581 probes), Mac CPU",
    "closing_test_gpu": {"status": "staged, not run", "command": f"cd {SUB}/jaxpb-pipeline-m1 && sbatch -J jpp-m1 -t 02:00:00 {SB} {SUB}/jaxpb-pipeline-m1 bash jobs/m1_gpu.sh",
                         "outputs": f"{ISA}/m1/{{m1_main.json, m1_cache.json, cc_m1.tar}}"},
}
# ---------------------------------------------------------------- M2
m2r = L("m2cpu/m2.json")
m2 = {
    "milestone": "M2", "date": "2026-11-19",
    "status": "estimator found and ported; CPU stand-in on the 99 M0 states; GPU closing test staged -- " + gpu_blocked,
    "method": "DelPhi 5.1 relfac (disassembled from delphi4py rundelphi .so): v = product of normalised sines sinf(i pi/80) sqrt2/sqrt80 "
              "(float32), w = J(J(v)) with the point-Jacobi PB operator, rho = 2 * float32 SEQUENTIAL sum over odd points (Fortran order) "
              "of v*w; rho > 1 -> 0.995. The float32 sequential sum lowers rho by ~1.2e-4 vs the exact dot product and is required.",
    "estimator_vs_logged_on_identical_maps": "Mac CPU, d0d5 B sites 0-3: |rho - logged| <= 9.5e-7 (4 of 8 exact to float32 ulps)",
    "criteria": {"sweeps_focused": ">= 0.87", "sweeps_model": ">= 0.83", "solv_median_rel": "<= 1e-3", "solv_p99_rel": "<= 1e-2", "sitpot_other_p99": "<= 0.02 kT/e"},
    "cpu_standin": m2r and m2r["summary"],
    "cpu_standin_path": f"{COUL}/m2cpu/",
    "closing_test_gpu": {"status": "staged, not run", "command": f"cd {SUB}/jaxpb-pipeline-m2 && sbatch -J jpp-m2 -t 04:00:00 {SB} {SUB}/jaxpb-pipeline-m2 bash jobs/m2_gpu.sh",
                         "outputs": f"{ISA}/m2/m2.json"},
}
os.makedirs(OUT, exist_ok=True)
for name, d in (("M0", m0), ("M1", m1), ("M2", m2)):
    json.dump(d, open(os.path.join(OUT, name + ".json"), "w"), indent=1, default=str)
print("written", OUT)
