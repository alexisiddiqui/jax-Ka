"""Short ANM/repair CPU-versus-CUDA probe using audited, completed templates.

This measures generation after source completion, not a production generator or
an end-to-end estimate. Inputs are packed; conformers never become loose files.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import queue
import subprocess
import tarfile
import time
import traceback

import numpy as np
import zstandard

from pkabench.runtime import atomic_json, digest, require_compute
from .augmentation_quality import quality, rmsd, verify_coordinate_record
from .augmentation_store import AugmentationStore, ReleaseWriter, decode, encode


def bundle(pilot, coordinates, output, per_dataset=8, all_complete=False):
    """Select size-spread complete parents; include zero-control failures."""
    rows = []
    with AugmentationStore(coordinates) as store, tarfile.open(output, "x:") as archive:
        for dataset in ("pkpdb", "pinder", "atlas"):
            cohort = sorted((r for r in store.rows if r["summary"]["dataset"] == dataset
                             and r["summary"]["status"] != "preparation_failed"),
                            key=lambda r: (r["summary"]["n"], r["id"]))
            if len(cohort) < per_dataset: raise ValueError("pilot cohort too small")
            positions = range(len(cohort)) if all_complete else np.linspace(0, len(cohort)-1, per_dataset).round().astype(int)
            for position in positions:
                row = cohort[position]; record = store.raw(row["id"])
                index = record["metadata"]["source"]["index"]
                unit = Path(pilot)/"units"/f"{index:03d}"
                prefix = f"{index:03d}"
                frame, raw_size = encode(record["metadata"], record["arrays"])
                files = {"record.zst": frame}
                for name in ("preparation.json", "prepared.npz", "system.xml"):
                    files[name] = (unit/name).read_bytes()
                members = {}
                for name, data in sorted(files.items()):
                    compressed = data if name == "record.zst" else zstandard.ZstdCompressor(level=3).compress(data)
                    entry = tarfile.TarInfo(prefix+"/"+name); entry.size = len(compressed)
                    members[name] = dict(offset=archive.offset+512, bytes=len(compressed),
                        sha256=hashlib.sha256(compressed).hexdigest(), compressed=name != "record.zst")
                    archive.addfile(entry, io.BytesIO(compressed))
                rows.append(dict(parent=row["id"], prefix=prefix, raw_size=raw_size,
                                 source=record["metadata"]["source"], files=members))
        manifest = dict(parents=rows, coordinate_manifest_sha256=digest(Path(coordinates)/"manifest.json"),
                        format="indexed-zstd-templates-v1",
                        selection="all complete parents" if all_complete else "size-spread complete parents, including failed zero controls",
                        scope="Audited completed templates; source preparation and CG2ALL excluded")
        data = json.dumps(manifest, sort_keys=True).encode()
        entry = tarfile.TarInfo("manifest.json"); entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    return manifest


def read_template(archive_path, row):
    if "files" in row:
        result = {}; fd = os.open(archive_path, os.O_RDONLY)
        try:
            for name, field in row["files"].items():
                data = os.pread(fd, field["bytes"], field["offset"])
                if len(data) != field["bytes"] or hashlib.sha256(data).hexdigest() != field["sha256"]:
                    raise ValueError("packed template checksum changed")
                result[name] = zstandard.ZstdDecompressor().decompress(data) if field["compressed"] else data
        finally: os.close(fd)
        return result
    with tarfile.open(archive_path, "r:gz") as archive:
        return {name: archive.extractfile(row["prefix"]+"/"+name).read()
                for name in ("record.zst", "preparation.json", "prepared.npz", "system.xml")}


def _stereo(system, xyz, info):
    import openmm as mm
    lookup = {(r, a): j for j, (r, a, native) in enumerate(zip(
        info["residue"], info["atom_names"], info["is_native"])) if native}
    force = mm.CustomCompoundBondForce(4,
        "0.5*stereo_k*min(0,v/vref-0.25)^2;"
        "v=(x2-x1)*((y3-y1)*(z4-z1)-(z3-z1)*(y4-y1))"
        "+(y2-y1)*((z3-z1)*(x4-x1)-(x3-x1)*(z4-z1))"
        "+(z2-z1)*((x3-x1)*(y4-y1)-(y3-y1)*(x4-x1))")
    force.addGlobalParameter("stereo_k", 1000.); force.addPerBondParameter("vref")
    for i, name in enumerate(info["names"]):
        centres = []
        if (i, "CB") in lookup: centres.append(["CA", "N", "C", "CB"])
        if name in ("THR", "ILE"):
            centres.append(["CB", "CA", "OG1" if name == "THR" else "CG1", "CG2"])
        for names in centres:
            ids = [lookup[i, a] for a in names]
            a, b, c = (xyz[ids[1:]]-xyz[ids[0]])*.1
            volume = float(np.dot(np.cross(a, b), c))
            if abs(volume) < 1e-6: raise ValueError("degenerate source stereocentre")
            force.addBond(ids, [volume])
    system.addForce(force)


def _unit(task):
    import openmm as mm
    from openmm import unit
    from prody import ANM
    archive_path, row, platform = task; started = time.perf_counter()
    try:
        inputs = read_template(archive_path, row)
        record = decode(inputs["record.zst"], max_output_size=row["raw_size"])
        info = json.loads(inputs["preparation.json"])
        with np.load(io.BytesIO(inputs["prepared.npz"])) as z:
            xyz = z["xyz"].copy(); bb = z["source_bb"].copy()
        system = mm.XmlSerializer.deserialize(inputs["system.xml"].decode())
        meta = record["metadata"]; arrays = record["arrays"]
        heavy = np.flatnonzero(np.asarray(info["is_native"]) & np.asarray(info["is_heavy"]))
        if not np.allclose(xyz[heavy], arrays["native_coordinates"], atol=0.001, rtol=0):
            raise ValueError("prepared template changed")
        ts = time.perf_counter(); anm = ANM(); anm.buildHessian(bb[:, 1], cutoff=15.)
        anm.calcModes(n_modes=5, zeros=False); values = anm.getEigvals()
        if len(values) != 5 or np.any(values <= 1e-6): raise ValueError("insufficient ANM modes")
        rng = np.random.default_rng(20261013+meta["source"]["index"])
        delta = ((rng.normal(size=(8, 5))/np.sqrt(values))@anm.getEigvecs().T).reshape(8, len(bb), 3)
        delta /= np.sqrt(np.mean(np.sum(delta**2, axis=-1), axis=-1))[:, None, None]
        targets = [bb[:, 1]]+[bb[:, 1]+amp*d for amp in (.5, 1., 2.) for d in delta]
        # Match the historical benchmark's PDB-rounded CA targets.
        targets = [np.round(ca, 3) for ca in targets]; anm_s = time.perf_counter()-ts
        force = system.getForce(system.getNumForces()-1)
        if not isinstance(force, mm.CustomExternalForce): raise ValueError("CA restraint missing")
        _stereo(system, xyz, info); ts = time.perf_counter()
        integrator = mm.VerletIntegrator(.001*unit.picosecond)
        props = {"DeviceIndex": "0", "Precision": "mixed"} if platform == "CUDA" else {"Threads": "1"}
        context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), props)
        context_s = time.perf_counter()-ts; metrics = []; coords = []
        for c, ca in enumerate(targets):
            ts = time.perf_counter()
            for i, j in enumerate(info["ca_ids"]): force.setParticleParameters(i, j, (ca[i]/10).tolist())
            force.updateParametersInContext(context)
            moved = xyz+(ca-bb[:, 1])[np.asarray(info["residue"])]
            context.setPositions(moved*.1*unit.nanometer)
            before = float(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            mt = time.perf_counter(); mm.LocalEnergyMinimizer.minimize(context, 10., 100)
            minimise_s = time.perf_counter()-mt
            state = context.getState(getPositions=True, getEnergy=True, getForces=True)
            post = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))[heavy].astype(np.float32)
            energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            forces = np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))
            if not np.isfinite(post).all() or not np.isfinite(energy) or not np.isfinite(forces).all():
                raise ValueError("nonfinite minimisation")
            if energy > before+max(1e-3, abs(before)*1e-6): raise ValueError("minimisation raised energy")
            repair_s = time.perf_counter()-ts; qt = time.perf_counter(); checked = quality(meta, arrays, post)
            metrics.append(dict(candidate=c, minimisation_call_s=minimise_s, repair_s=repair_s,
                                quality_s=time.perf_counter()-qt, energy_before=before, energy_after=energy,
                                final_rms_force=float(np.sqrt(np.mean(np.sum(forces**2, axis=1)))), **checked))
            coords.append(post)
        chosen = []; candidates = [m["candidate"] for m in metrics[1:] if m["geometry_valid"]]
        ids = arrays["backbone_atoms"][:, 1]
        while candidates and len(chosen) < 4:
            best = max(candidates, key=lambda c: (min([metrics[c]["ca_rmsd_A"]]+
                [rmsd(coords[c][ids], coords[j][ids]) for j in chosen]), -c))
            chosen.append(best); candidates.remove(best)
        if not metrics[0]["geometry_valid"]: chosen = []
        packed = dict(arrays); packed["zero_coordinates"] = coords[0]
        packed["coordinates"] = np.stack([coords[c] for c in chosen]) if chosen else np.empty((0, len(heavy), 3), np.float32)
        updated = dict(meta, selected=chosen, status="usable" if len(chosen) >= 3 else
            ("zero_control_failed" if not metrics[0]["geometry_valid"] else "insufficient_candidates"),
            zero_quality=metrics[0], selected_quality={str(c): metrics[c] for c in chosen},
            selected_pairwise_ca_rmsd_A=[rmsd(coords[a][ids], coords[b][ids])
                for i, a in enumerate(chosen) for b in chosen[i+1:]],
            config=dict(meta["config"], platform=platform, openmm=mm.__version__),
            benchmark_scope="Completed-template ANM/repair; source preparation and CG2ALL excluded")
        verify_coordinate_record(dict(metadata=updated, arrays=packed))
        return dict(parent=row["parent"], passed=True, dataset=meta["source"]["dataset"], n=len(bb),
                    anm_s=anm_s, context_s=context_s, wall_s=time.perf_counter()-started,
                    target_sha256=hashlib.sha256(np.asarray(targets, np.float64).tobytes()).hexdigest(),
                    peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                    selected=chosen, zero_valid=metrics[0]["geometry_valid"], metrics=metrics,
                    metadata=updated, arrays=packed)
    except Exception:
        return dict(parent=row["parent"], passed=False, error=traceback.format_exc())


def worker_layout(cpus, cpu_workers, gpu_workers):
    if cpu_workers < 0 or gpu_workers < 0 or not cpu_workers+gpu_workers:
        raise ValueError("invalid worker counts")
    if cpu_workers+gpu_workers >= len(cpus):
        raise ValueError("workers must leave at least one CPU for the writer")
    return [("CPU", cpu) for cpu in cpus[:cpu_workers]]+[
        ("CUDA", cpu) for cpu in cpus[cpu_workers:cpu_workers+gpu_workers]]


def _worker(archive, inputs, outputs, platform, cpu, number):
    os.sched_setaffinity(0, {cpu})
    while True:
        row = inputs.get()
        if row is None: return
        try: result = _unit((archive, row, platform))
        except Exception: result = dict(parent=row["parent"], passed=False, error=traceback.format_exc())
        result.update(platform=platform, worker=number, affinity_cpu=cpu)
        outputs.put(result)


def _rss_gib(pids):
    total = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"): total += int(line.split()[1])*1024
        except FileNotFoundError: pass
    return total/1024**3


def _phase(bundle_path, rows, output, provenance, label, cpu_workers, gpu_workers, cpus, gpu_id):
    layout = worker_layout(cpus, cpu_workers, gpu_workers)
    context = mp.get_context("spawn"); inputs = context.Queue(); outputs = context.Queue(maxsize=len(layout))
    # Largest first reduces the idle tail for both standalone and mixed phases.
    for row in sorted(rows, key=lambda r: (r["source"]["n"], r["parent"]), reverse=True): inputs.put(row)
    for _ in layout: inputs.put(None)
    processes = [context.Process(target=_worker, args=(str(bundle_path), inputs, outputs, platform, cpu, i))
                 for i, (platform, cpu) in enumerate(layout)]
    results = []; peak_rss = 0.; gpu_peak = None; last_gpu_probe = 0.; started = time.perf_counter()
    try:
        for process in processes: process.start()
        os.sched_setaffinity(0, set(cpus[len(layout):]))
        with ReleaseWriter(output/label, dict(provenance, cpu_workers=cpu_workers, gpu_workers=gpu_workers)) as writer:
            while len(results) < len(rows):
                peak_rss = max(peak_rss, _rss_gib([os.getpid()]+[p.pid for p in processes]))
                now = time.perf_counter()
                if gpu_workers and now-last_gpu_probe >= 3:
                    try:
                        memory = int(subprocess.check_output(["nvidia-smi", "-i", gpu_id,
                            "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
                        gpu_peak = max(gpu_peak or 0, memory)
                    except (subprocess.SubprocessError, ValueError): pass
                    last_gpu_probe = now
                try: result = outputs.get(timeout=.5)
                except queue.Empty:
                    if any(p.exitcode not in (None, 0) for p in processes):
                        raise RuntimeError("repair worker exited unexpectedly")
                    if all(p.exitcode is not None for p in processes):
                        raise RuntimeError("workers finished without complete inventory")
                    continue
                if result["passed"]:
                    writer.add(result["parent"], result.pop("metadata"), result.pop("arrays"),
                        summary=dict(dataset=result["dataset"], selected=len(result["selected"]), platform=result["platform"]))
                else: writer.add(result["parent"], result, {}, summary=dict(status="error"))
                results.append(result)
                print(json.dumps(dict(phase=label, completed=len(results), total=len(rows),
                    parent=result["parent"], platform=result["platform"], passed=result["passed"])), flush=True)
            for process in processes:
                process.join(timeout=30)
                if process.exitcode != 0: raise RuntimeError("repair worker did not shut down cleanly")
            writer.finish(expected_ids=[r["parent"] for r in rows])
    finally:
        for process in processes:
            if process.is_alive(): process.terminate()
            if process.pid is not None: process.join(timeout=5)
        inputs.close(); outputs.close(); os.sched_setaffinity(0, set(cpus))
    return dict(label=label, cpu_workers=cpu_workers, gpu_workers=gpu_workers,
        elapsed_s=time.perf_counter()-started, passed=all(r["passed"] for r in results), results=results,
        peak_sum_process_rss_GiB=peak_rss, peak_gpu_memory_MiB=gpu_peak,
        affinity=dict(CPU=[cpu for platform, cpu in layout if platform == "CPU"],
                      CUDA=[cpu for platform, cpu in layout if platform == "CUDA"]))


def run(bundle_path, output, cpu_workers=24, mixed=False, mixed_cpu_workers=(32, 48, 64)):
    import openmm as mm
    import prody
    require_compute(threads=int(os.environ["SLURM_CPUS_PER_TASK"]), gpu_benchmark=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or not visible[0]: raise RuntimeError("pilot requires exactly one visible GPU")
    gpu = subprocess.check_output(["nvidia-smi", "-i", visible[0],
        "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"], text=True).strip().splitlines()
    if len(gpu) != 1: raise RuntimeError("pilot requires exactly one visible GPU")
    with tarfile.open(bundle_path, "r:*") as archive: manifest = json.load(archive.extractfile("manifest.json"))
    rows = manifest["parents"]; output = Path(output); output.mkdir(parents=True, exist_ok=True)
    report = dict(input_sha256=digest(bundle_path), job=os.environ["SLURM_JOB_ID"],
        node=os.environ["SLURMD_NODENAME"], gpu=gpu, openmm=mm.__version__, prody=prody.__version__,
        cpus=int(os.environ["SLURM_CPUS_PER_TASK"]), parents=len(rows),
        git_commit=subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip(),
        scope="Completed-template generation, including ANM, zero control, 24 candidates, quality, selection and packing. Source preparation/CG2ALL, full random I/O and graphs excluded.", phases=[])
    cpus = sorted(os.sched_getaffinity(0))
    phases = [(f"cpu-{cpu_workers}", cpu_workers, 0), ("cuda-1", 0, 1), ("cuda-4", 0, 4)]
    if mixed:
        phases = [("cuda-4-warmup", 0, 4), ("cuda-4-before", 0, 4), (f"cpu-{cpu_workers}", cpu_workers, 0)]+[
            (f"mixed-{n}+4", n, 4) for n in mixed_cpu_workers]+[("cuda-4-after", 0, 4)]
    for _, cpu_count, gpu_count in phases: worker_layout(cpus, cpu_count, gpu_count)
    report.update(order="largest first, common queue dynamically claimed by CPU and CUDA workers",
        cache_scope="Pilot working set; CUDA warmup excluded from comparisons; before/after GPU controls detect drift",
        phase_plan=[dict(label=label, cpu_workers=c, gpu_workers=g) for label, c, g in phases])
    atomic_json(output/"report.json", report)
    for label, cpu_count, gpu_count in phases:
        provenance = dict(kind="gh200-repair-probe-v2", input_sha256=report["input_sha256"],
                          openmm=mm.__version__, code_sha256=digest(__file__))
        phase = _phase(bundle_path, rows, output, provenance, label, cpu_count, gpu_count, cpus, visible[0])
        report["phases"].append(phase); atomic_json(output/"report.json", report)
    report["passed"] = all(p["passed"] for p in report["phases"])
    report["summary"] = {}
    for phase in report["phases"]:
        complete = [r for r in phase["results"] if r["passed"]]
        selected_rmsds = [r["metrics"][c]["ca_rmsd_A"] for r in complete for c in r["selected"]]
        report["summary"][phase["label"]] = dict(
            elapsed_s=phase["elapsed_s"], parents_per_s=len(complete)/phase["elapsed_s"],
            complete=len(complete), zero_valid=sum(r["zero_valid"] for r in complete),
            yield_four=sum(len(r["selected"]) == 4 for r in complete),
            mean_unit_s=float(np.mean([r["wall_s"] for r in complete])) if complete else None,
            max_worker_rss_GiB=max((r["peak_rss_GiB"] for r in complete), default=None),
            peak_sum_process_rss_GiB=phase["peak_sum_process_rss_GiB"],
            peak_gpu_memory_MiB=phase["peak_gpu_memory_MiB"],
            platform_counts={p: sum(r["platform"] == p for r in complete) for p in ("CPU", "CUDA")},
            selected_ca_rmsd_A=dict(zip(("p05", "median", "p95"),
                np.quantile(selected_rmsds, [.05, .5, .95]).tolist())) if selected_rmsds else None)
    atomic_json(output/"report.json", report)
    if not report["passed"]: raise RuntimeError("repair probe failures; inspect report")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bundle"); b.add_argument("pilot", type=Path); b.add_argument("coordinates", type=Path)
    b.add_argument("output", type=Path); b.add_argument("--per-dataset", type=int, default=8)
    b.add_argument("--all-complete", action="store_true")
    r = sub.add_parser("run"); r.add_argument("bundle", type=Path); r.add_argument("output", type=Path)
    r.add_argument("--cpu-workers", type=int, default=24)
    r.add_argument("--mixed", action="store_true")
    r.add_argument("--mixed-cpu-workers", type=int, nargs="+", default=[32, 48, 64])
    args = parser.parse_args()
    if args.command == "bundle":
        require_compute()
        manifest = bundle(args.pilot, args.coordinates, args.output, args.per_dataset, args.all_complete)
        print(json.dumps(dict(parents=len(manifest["parents"]), format=manifest["format"],
            input_bytes=args.output.stat().st_size, input_sha256=digest(args.output))))
    else:
        if not 1 <= args.cpu_workers <= int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
            raise ValueError("CPU workers exceed allocation")
        run(args.bundle, args.output, args.cpu_workers, args.mixed, args.mixed_cpu_workers)


if __name__ == "__main__": main()
