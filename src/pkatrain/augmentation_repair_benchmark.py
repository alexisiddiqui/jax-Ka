"""Short ANM/repair CPU-versus-CUDA probe using audited, completed templates.

This measures generation after source completion, not a production generator or
an end-to-end estimate. Inputs are packed; conformers never become loose files.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import tarfile
import time
import traceback

import numpy as np

from pkabench.runtime import atomic_json, digest, require_compute
from .augmentation_quality import quality, rmsd, verify_coordinate_record
from .augmentation_store import AugmentationStore, ReleaseWriter, decode, encode


def bundle(pilot, coordinates, output, per_dataset=8):
    """Select size-spread complete parents; include zero-control failures."""
    rows = []; files = {}
    with AugmentationStore(coordinates) as store:
        for dataset in ("pkpdb", "pinder", "atlas"):
            cohort = sorted((r for r in store.rows if r["summary"]["dataset"] == dataset
                             and r["summary"]["status"] != "preparation_failed"),
                            key=lambda r: (r["summary"]["n"], r["id"]))
            if len(cohort) < per_dataset: raise ValueError("pilot cohort too small")
            positions = np.linspace(0, len(cohort)-1, per_dataset).round().astype(int)
            for position in positions:
                row = cohort[position]; record = store.raw(row["id"])
                index = record["metadata"]["source"]["index"]
                unit = Path(pilot)/"units"/f"{index:03d}"
                prefix = f"{index:03d}"
                frame, raw_size = encode(record["metadata"], record["arrays"])
                files[prefix+"/record.zst"] = frame
                for name in ("preparation.json", "prepared.npz", "system.xml"):
                    files[prefix+"/"+name] = (unit/name).read_bytes()
                rows.append(dict(parent=row["id"], prefix=prefix, raw_size=raw_size,
                                 source=record["metadata"]["source"]))
        manifest = dict(parents=rows, coordinate_manifest_sha256=digest(Path(coordinates)/"manifest.json"),
                        selection="size-spread complete parents, including failed zero controls",
                        scope="Audited completed templates; source preparation and CG2ALL excluded")
    files["manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
    with tarfile.open(output, "x:gz") as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return manifest


def _initialise():
    # Distinct physical affinity slots; OpenMM/BLAS each use one thread.
    available = sorted(os.sched_getaffinity(0))
    slot = (mp.current_process()._identity[0]-1) % len(available)
    os.sched_setaffinity(0, {available[slot]})


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
        with tarfile.open(archive_path, "r:gz") as archive:
            def read(name): return archive.extractfile(row["prefix"]+"/"+name).read()
            record = decode(read("record.zst"), max_output_size=row["raw_size"])
            info = json.loads(read("preparation.json"))
            with np.load(io.BytesIO(read("prepared.npz"))) as z:
                xyz = z["xyz"].copy(); bb = z["source_bb"].copy()
            system = mm.XmlSerializer.deserialize(read("system.xml").decode())
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
                    peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                    selected=chosen, zero_valid=metrics[0]["geometry_valid"], metrics=metrics,
                    metadata=updated, arrays=packed)
    except Exception:
        return dict(parent=row["parent"], passed=False, error=traceback.format_exc())


def run(bundle_path, output, cpu_workers=24):
    import openmm as mm
    import prody
    require_compute(threads=int(os.environ["SLURM_CPUS_PER_TASK"]), gpu_benchmark=True)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or not visible[0]: raise RuntimeError("pilot requires exactly one visible GPU")
    gpu = subprocess.check_output(["nvidia-smi", "-i", visible[0],
        "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"], text=True).strip().splitlines()
    if len(gpu) != 1: raise RuntimeError("pilot requires exactly one visible GPU")
    with tarfile.open(bundle_path, "r:gz") as archive: manifest = json.load(archive.extractfile("manifest.json"))
    rows = manifest["parents"]; output = Path(output); output.mkdir(parents=True, exist_ok=True)
    report = dict(input_sha256=digest(bundle_path), job=os.environ["SLURM_JOB_ID"],
        node=os.environ["SLURMD_NODENAME"], gpu=gpu, openmm=mm.__version__, prody=prody.__version__,
        cpus=int(os.environ["SLURM_CPUS_PER_TASK"]), parents=len(rows),
        git_commit=subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True).strip(),
        scope="Completed-template generation, including ANM, zero control, 24 candidates, quality, selection and packing. Source preparation/CG2ALL, full random I/O and graphs excluded.", phases=[])
    for platform, workers in (("CPU", cpu_workers), ("CUDA", 1), ("CUDA", 4)):
        label = f"{platform.lower()}-{workers}"; results = []; ts = time.perf_counter()
        provenance = dict(kind="gh200-repair-probe-v1", input_sha256=report["input_sha256"], platform=platform,
                          workers=workers, openmm=mm.__version__, code_sha256=digest(__file__))
        with ReleaseWriter(output/label, provenance) as writer:
            with mp.get_context("spawn").Pool(workers, initializer=_initialise) as pool:
                for result in pool.imap_unordered(_unit, [(str(bundle_path), r, platform) for r in rows]):
                    if result["passed"]:
                        writer.add(result["parent"], result.pop("metadata"), result.pop("arrays"),
                                   summary=dict(dataset=result["dataset"], selected=len(result["selected"])))
                    else:
                        writer.add(result["parent"], result, {}, summary=dict(status="error"))
                    results.append(result)
                    print(json.dumps(dict(phase=label, completed=len(results), parent=result["parent"], passed=result["passed"])), flush=True)
            writer.finish(expected_ids=[r["parent"] for r in rows])
        phase = dict(label=label, workers=workers, platform=platform, elapsed_s=time.perf_counter()-ts,
                     passed=all(r["passed"] for r in results), results=results)
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
            selected_ca_rmsd_A=dict(zip(("p05", "median", "p95"),
                np.quantile(selected_rmsds, [.05, .5, .95]).tolist())) if selected_rmsds else None)
    atomic_json(output/"report.json", report)
    if not report["passed"]: raise RuntimeError("repair probe failures; inspect report")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bundle"); b.add_argument("pilot", type=Path); b.add_argument("coordinates", type=Path)
    b.add_argument("output", type=Path); b.add_argument("--per-dataset", type=int, default=8)
    r = sub.add_parser("run"); r.add_argument("bundle", type=Path); r.add_argument("output", type=Path)
    r.add_argument("--cpu-workers", type=int, default=24)
    args = parser.parse_args()
    if args.command == "bundle":
        require_compute(); print(json.dumps(bundle(args.pilot, args.coordinates, args.output, args.per_dataset)))
    else:
        if not 1 <= args.cpu_workers <= int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
            raise ValueError("CPU workers exceed allocation")
        run(args.bundle, args.output, args.cpu_workers)


if __name__ == "__main__": main()
