"""Compare global and local surface changes in an existing coordinate pilot.

No training, regeneration, or release mutation. Surface = Shrake--Rupley
solvent-accessible spheres, not a solvent-excluded/reentrant molecular mesh.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pkabench.runtime import atomic_json, digest, require_compute
from pkatrain.augmentation_store import AugmentationStore

RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
AMINO = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
CHEM = {name: (1 if name in {"ARG", "LYS"} else 2 if name in {"ASP", "GLU"}
               else 0 if name in {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "CYS"}
               else 3) for name in AMINO}


def fit(moving, reference):
    """Proper row-vector Kabsch: moving @ rotation + translation."""
    mc, rc = moving.mean(0), reference.mean(0)
    u, _, vt = np.linalg.svd((moving - mc).T @ (reference - rc))
    d = np.eye(3); d[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ d @ vt
    return rotation, rc - mc @ rotation


def rms(moving, reference):
    return float(np.sqrt(np.mean(np.sum((moving - reference) ** 2, axis=-1))))


def frames(xyz, bb):
    ca = xyz[bb[:, 1]]
    x = xyz[bb[:, 2]] - ca
    x /= np.linalg.norm(x, axis=1)[:, None]
    y = xyz[bb[:, 0]] - ca
    y -= np.sum(x * y, axis=1)[:, None] * x
    y /= np.linalg.norm(y, axis=1)[:, None]
    return np.stack((x, y, np.cross(x, y)), axis=1)


def sphere(n):
    i = np.arange(n, dtype=float)
    z = 1 - 2 * (i + .5) / n
    phi = i * np.pi * (3 - np.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=1)


def surface(xyz, bb, owner, radius, npoints):
    """Equal-area tests on inflated vdW spheres; directions follow residue frames.

    Following frames makes identical rigid motions exactly invariant at finite
    sampling density. Surface construction always includes the WHOLE protein,
    so extracting a patch does not expose atoms buried by its surroundings.
    """
    directions = np.einsum("pi,aij->apj", sphere(npoints), frames(xyz, bb)[owner])
    neighbors = cKDTree(xyz).query_ball_point(xyz, radius + radius.max() + 1e-8)
    visible = np.zeros((len(xyz), npoints), bool)
    for a, nearby in enumerate(neighbors):
        nearby = np.asarray([b for b in nearby if b != a], dtype=int)
        points = xyz[a] + radius[a] * directions[a]
        if not len(nearby):
            visible[a] = True
            continue
        delta = points[:, None, :] - xyz[nearby][None, :, :]
        dist2 = np.einsum("pni,pni->pn", delta, delta)
        visible[a] = ~np.any(dist2 < radius[nearby][None, :] ** 2 - 1e-9, axis=1)
    atoms, samples = np.nonzero(visible)
    normals = directions[atoms, samples]
    area = 4 * np.pi * radius ** 2 / npoints
    return dict(points=xyz[atoms] + radius[atoms, None] * normals,
                normal=normals, atom=atoms, sample_id=atoms * npoints + samples,
                weight=area[atoms], atom_area=visible.sum(1) * area)


def shape(a, b, rotation, translation):
    """Area-weighted symmetric nearest-surface distance, in Angstrom, not squared."""
    ap, bp = a["points"], b["points"] @ rotation + translation
    dab, iab = cKDTree(bp).query(ap, workers=1)
    dba, iba = cKDTree(ap).query(bp, workers=1)
    wa, wb = a["weight"], b["weight"]
    distance = .5 * (np.average(dab, weights=wa) + np.average(dba, weights=wb))
    na, nb = a["normal"], b["normal"] @ rotation
    angle = .5 * (np.average(np.degrees(np.arccos(np.clip(np.sum(na * nb[iab], 1), -1, 1))), weights=wa)
                  + np.average(np.degrees(np.arccos(np.clip(np.sum(nb * na[iba], 1), -1, 1))), weights=wb))
    mismatch = .5 * (np.average(a["chem"] != b["chem"][iab], weights=wa)
                     + np.average(b["chem"] != a["chem"][iba], weights=wb))
    # Same atom/sample, exposed in both; complementary to NN distances, which
    # can match distinct atoms and hide chemically significant rearrangements.
    _, ai, bi = np.intersect1d(a["sample_id"], b["sample_id"], return_indices=True)
    matched = rms(bp[bi], ap[ai]) if len(ai) else float("nan")
    return dict(distance_A=float(distance), normal_angle_deg=float(angle),
                chemistry_mismatch_fraction=float(mismatch), matched_surface_rmsd_A=matched)


def crop(s, center, radius):
    mask = np.sum((s["points"] - center) ** 2, axis=1) <= radius ** 2
    return {k: v[mask] for k, v in s.items() if k != "atom_area"}


def composition(s):
    hist = np.bincount(s["aa"], weights=s["weight"], minlength=20)
    return hist / hist.sum()


def atom_exposure(a, b):
    return float(np.minimum(a, b).sum() / np.maximum(a, b).sum())


def anchors(xyz, bb, owner, surf, radius, limit):
    ca = xyz[bb[:, 1]]
    areas = np.bincount(owner, weights=surf["atom_area"], minlength=len(bb))
    eligible = np.flatnonzero(areas >= 20.)
    if not len(eligible): return []
    # Deterministic farthest point centers, independent of augmented structures.
    chosen = [int(eligible[np.argmax(areas[eligible])])]
    distances = np.full(len(eligible), np.inf)
    while len(chosen) < min(limit, len(eligible)):
        distances = np.minimum(distances, np.sum((ca[eligible] - ca[chosen[-1]]) ** 2, 1))
        nxt = int(np.argmax(distances))
        if distances[nxt] < 36.: break
        chosen.append(int(eligible[nxt]))
    result = []
    for r in chosen:
        points = surf["points"][owner[surf["atom"]] == r]
        center = points[np.argmin(np.sum((points - ca[r]) ** 2, 1))]
        members = np.flatnonzero(np.sum((ca - center) ** 2, 1) <= radius ** 2)
        if len(members) < 4 or np.linalg.matrix_rank(ca[members] - ca[members].mean(0)) < 2:
            continue
        result.append((r, center, members))
    return result


def annotate(s, atom_aa, atom_chem):
    s["aa"] = atom_aa[s["atom"]]
    s["chem"] = atom_chem[s["atom"]]
    return s


def compare_pair(meta, arrays, reference, moving, group, candidate, config):
    bb, owner = arrays["backbone_atoms"], arrays["atom_residue"]
    radius = np.array([RADII[a[0]] + 1.4 for a in meta["atom_names"]])
    names = meta["residue_names"]
    atom_aa = np.array([AMINO.index(names[r]) for r in owner])
    atom_chem = np.array([CHEM[names[r]] for r in owner])
    a = annotate(surface(reference, bb, owner, radius, config["points"]), atom_aa, atom_chem)
    b = annotate(surface(moving, bb, owner, radius, config["points"]), atom_aa, atom_chem)
    ca_a, ca_b = reference[bb[:, 1]], moving[bb[:, 1]]
    rotation, translation = fit(ca_b, ca_a)
    common = dict(parent=meta["parent"], dataset=meta["source"]["dataset"],
                  split=meta["source"]["split"], size_bin=meta["source"]["bin"],
                  group=group, candidate=candidate,
                  zero_valid=meta["zero_quality"]["geometry_valid"],
                  n_residues=len(bb), completed_fraction=float(arrays["completed"].mean()))
    whole = dict(common, global_ca_rmsd_A=rms(ca_b @ rotation + translation, ca_a),
                 global_heavy_rmsd_A=rms(moving @ rotation + translation, reference),
                 global_sasa_relative_change=float((b["weight"].sum() - a["weight"].sum()) / a["weight"].sum()),
                 global_atom_exposure_jaccard=atom_exposure(a["atom_area"], b["atom_area"]),
                 global_composition_TV=float(.5 * np.abs(composition(a) - composition(b)).sum()),
                 **{"global_surface_" + k: v for k, v in shape(a, b, rotation, translation).items()})
    rows = []
    fa, fb = frames(reference, bb), frames(moving, bb)
    for r, center, members in anchors(reference, bb, owner, a, config["radius"], config["patches"]):
        # Anchor follows its residue backbone frame; atom/residue correspondence
        # and alignment members are fixed from the reference, not NN reassigned.
        moved_center = ca_b[r] + (center - ca_a[r]) @ fa[r].T @ fb[r]
        pa, pb = crop(a, center, config["radius"]), crop(b, moved_center, config["radius"])
        if min(len(pa["points"]), len(pb["points"])) < 20: continue
        lr, lt = fit(ca_b[members], ca_a[members])
        atoms = np.isin(owner, members)
        globally = shape(pa, pb, rotation, translation)
        locally = shape(pa, pb, lr, lt)
        da = np.linalg.norm(ca_a[members, None] - ca_a[None, members], axis=-1)
        db = np.linalg.norm(ca_b[members, None] - ca_b[None, members], axis=-1)
        pairmask = np.triu(np.ones_like(da, bool), 1)
        row = dict(common, anchor_residue=int(r), anchor_key=json.dumps(meta["residue_keys"][r]),
                   patch_residues=len(members), patch_reference_area_A2=float(pa["weight"].sum()),
                   global_ca_rmsd_A=whole["global_ca_rmsd_A"],
                   patch_ca_after_global_rmsd_A=rms(ca_b[members] @ rotation + translation, ca_a[members]),
                   patch_ca_after_local_rmsd_A=rms(ca_b[members] @ lr + lt, ca_a[members]),
                   patch_heavy_after_local_rmsd_A=rms(moving[atoms] @ lr + lt, reference[atoms]),
                   patch_ca_distance_change_rms_A=float(np.sqrt(np.mean((da[pairmask] - db[pairmask]) ** 2))),
                   patch_sasa_relative_change=float((pb["weight"].sum() - pa["weight"].sum()) / pa["weight"].sum()),
                   patch_atom_exposure_jaccard=atom_exposure(a["atom_area"][atoms], b["atom_area"][atoms]),
                   patch_composition_TV=float(.5 * np.abs(composition(pa) - composition(pb)).sum()),
                   **{"patch_after_global_" + k: v for k, v in globally.items()},
                   **{"patch_after_local_" + k: v for k, v in locally.items()})
        rows.append(row)
    return whole, rows


def run_parent(task):
    path, parent, config = task
    start = time.monotonic()
    with AugmentationStore(path) as store:
        record = store.raw(parent)
    m, a = record["metadata"], record["arrays"]
    if not a: return dict(parent=parent, skipped="no coordinates")
    native, zero = a["native_coordinates"].astype(float), a["zero_coordinates"].astype(float)
    pairs = [("repair_only", 0, native, zero)]
    for candidate, xyz in zip(m["selected"], a["coordinates"]):
        pairs.extend([("selected_vs_native", candidate, native, xyz.astype(float)),
                      ("selected_vs_zero", candidate, zero, xyz.astype(float))])
    # Historical CPU pilot retains all amplitudes. One direction, identical
    # across the three amplitudes, avoids choosing a favorable deformation.
    old = Path(config["historical"]) / f'{m["source"]["index"]:03d}'
    historical = []
    if old.exists() and m["zero_quality"]["geometry_valid"]:
        prep = json.loads((old / "preparation.json").read_text())
        heavy = np.flatnonzero(np.asarray(prep["is_native"]) & np.asarray(prep["is_heavy"]))
        if ([prep["atom_names"][j] for j in heavy] != m["atom_names"]
                or not np.array_equal(np.asarray(prep["residue"])[heavy], a["atom_residue"])):
            raise ValueError("historical atom correspondence changed")
        prepared = np.load(old / "prepared.npz")["xyz"][heavy]
        if not np.allclose(prepared, native, atol=.001, rtol=0):
            raise ValueError("historical native does not match current coordinate record")
        report = json.loads((old / "report.json").read_text())
        metrics = {v["candidate"]: v for v in report["metrics"]}
        histzero = np.load(old / "native/c00/min.npy")[heavy].astype(float)
        for candidate, amp in ((1, .5), (9, 1.), (17, 2.)):
            if metrics[0]["geometry_valid"] and metrics[candidate]["geometry_valid"]:
                xyz = np.load(old / f"native/c{candidate:02d}/min.npy")[heavy].astype(float)
                pairs.append((f"amplitude_{amp:g}_vs_zero", candidate, histzero, xyz))
                historical.append(dict(candidate=candidate, amplitude_A=amp, path=str(old),
                                       coordinates_sha256=digest(old / f"native/c{candidate:02d}/min.npy"),
                                       zero_sha256=digest(old / "native/c00/min.npy"),
                                       preparation_sha256=digest(old / "preparation.json"),
                                       report_sha256=digest(old / "report.json")))
    # Cache surfaces for this parent; the same reference occurs repeatedly.
    original_surface = globals()["surface"]
    cache = {}
    def cached(xyz, bb, owner, radius, points):
        key = hashlib.sha256(xyz.tobytes()).hexdigest()
        if key not in cache: cache[key] = original_surface(xyz, bb, owner, radius, points)
        return cache[key]
    globals()["surface"] = cached
    try:
        wholes, patches = [], []
        for group, candidate, ref, xyz in pairs:
            w, p = compare_pair(m, a, ref, xyz, group, candidate, config)
            wholes.append(w); patches.extend(p)
    finally:
        globals()["surface"] = original_surface
    return dict(parent=parent, wholes=wholes, patches=patches, historical=historical,
                wall_s=time.monotonic() - start)


def checks():
    # Numerical controls that matter: proper alignment, analytic isolated-atom
    # area, burial, and finite-sampling rigid-motion invariance.
    rng = np.random.default_rng(91)
    xyz = rng.normal(size=(12, 3)) * 3
    bb = np.arange(12).reshape(3, 4)
    owner = np.repeat(np.arange(3), 4)
    radius = np.full(12, 3.1)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0: q[:, 0] *= -1
    moved = xyz @ q + [8, -2, 5]
    rotation, translation = fit(moved, xyz)
    assert rms(moved @ rotation + translation, xyz) < 1e-12
    a, b = surface(xyz, bb, owner, radius, 256), surface(moved, bb, owner, radius, 256)
    assert np.array_equal(a["sample_id"], b["sample_id"])
    assert rms(b["points"] @ rotation + translation, a["points"]) < 1e-11
    assert np.allclose(a["atom_area"], b["atom_area"], atol=1e-10)
    for s in (a, b): s["chem"] = owner[s["atom"]] % 4
    rigid = shape(a, b, rotation, translation)
    assert rigid["distance_A"] < 1e-11 and rigid["chemistry_mismatch_fraction"] == 0
    isolated = np.array([[0., 0., 0.], [100., 0., 0.], [0., 100., 0.], [0., 0., 100.]])
    iso = surface(isolated, np.arange(4)[None], np.zeros(4, int), np.full(4, 3.1), 256)
    assert np.allclose(iso["atom_area"], 4 * np.pi * 3.1 ** 2)
    buried = surface(isolated, np.arange(4)[None], np.zeros(4, int), np.array([200., 3., 3., 3.]), 256)
    assert np.all(buried["atom_area"][1:] == 0)
    return dict(passed=True, rigid_surface_error_A=rigid["distance_A"],
                checks=["proper Kabsch", "isolated sphere area", "buried atoms", "rigid surface and chemistry invariance"])


def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summary(wholes, patches):
    # Equal parent weight, then equal conformer weight, then equal patch weight.
    # Pooling patches directly would give large proteins excess influence.
    output = {}
    for group in sorted({w["group"] for w in wholes}):
        ws = [w for w in wholes if w["group"] == group and w["zero_valid"]]
        ps = [p for p in patches if p["group"] == group and p["zero_valid"]]
        by_parent = {}
        for key in ws[0] if ws else []:
            if key in {"candidate", "size_bin", "n_residues", "completed_fraction", "zero_valid"} or not isinstance(ws[0][key], (float, int)):
                continue
            values = [np.mean([w[key] for w in ws if w["parent"] == parent]) for parent in sorted({w["parent"] for w in ws})]
            by_parent[key] = dict(zip(("p05", "median", "p95"), map(float, np.quantile(values, [.05, .5, .95]))))
        if ps:
            for key in ps[0]:
                if not key.startswith("patch_") or key == "patch_residues": continue
                values = [np.mean([p[key] for p in ps if p["parent"] == parent]) for parent in sorted({p["parent"] for p in ps})]
                by_parent[key] = dict(zip(("p05", "median", "p95"), map(float, np.quantile(values, [.05, .5, .95]))))
            per_parent = []
            for parent in sorted({p["parent"] for p in ps}):
                pp = [p for p in ps if p["parent"] == parent]
                per_parent.append(float(np.mean([p["patch_after_local_distance_A"] < .5 for p in pp])))
            by_parent["fraction_patches_local_surface_distance_below_0p5_A"] = dict(mean=float(np.mean(per_parent)))
        output[group] = dict(parents=len({w["parent"] for w in ws}), frames=len(ws), patches=len(ps), metrics=by_parent)
    return output


def plots(out, ws, ps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = ["repair_only", "amplitude_0.5_vs_zero", "amplitude_1_vs_zero", "amplitude_2_vs_zero", "selected_vs_zero"]
    labels = ["Repair only", "0.5 Å ANM", "1 Å ANM", "2 Å ANM", "Selected ANM"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for group, label in zip(groups, labels):
        wp = [w for w in ws if w["group"] == group and w["zero_valid"]]
        pp = [p for p in ps if p["group"] == group and p["zero_valid"]]
        if not pp: continue
        axes[0, 0].scatter([w["global_ca_rmsd_A"] for w in wp], [w["global_surface_distance_A"] for w in wp], s=8, alpha=.3, label=label)
        # One average per parent for legible, parent-balanced scatter.
        parents = sorted({p["parent"] for p in pp})
        x = [np.mean([p["patch_after_global_distance_A"] for p in pp if p["parent"] == k]) for k in parents]
        y = [np.mean([p["patch_after_local_distance_A"] for p in pp if p["parent"] == k]) for k in parents]
        axes[0, 1].scatter(x, y, s=9, alpha=.4, label=label)
        for ax, key in ((axes[1, 0], "patch_after_local_distance_A"), (axes[1, 1], "patch_composition_TV")):
            values = np.sort([np.mean([p[key] for p in pp if p["parent"] == k]) for k in parents])
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=label)
    axes[0, 0].set(xlabel="Whole-protein aligned Cα RMSD (Å)", ylabel="Whole-surface distance (Å)")
    axes[0, 1].set(xlabel="Patch surface distance after global alignment (Å)", ylabel="After local patch alignment (Å)")
    axes[0, 1].plot([0, 2], [0, 2], "k--", lw=.7)
    axes[1, 0].set(xlabel="Mean locally aligned patch surface distance (Å)", ylabel="Fraction of proteins")
    axes[1, 1].set(xlabel="Mean patch exposed-residue composition change (TV)", ylabel="Fraction of proteins")
    for ax in axes.flat: ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Existing ANM pilot: global versus 10 Å local surfaces\nSolvent-accessible surface; geometry and chemistry proxies; no training")
    fig.savefig(out / "global-vs-local.png", dpi=180)
    fig.savefig(out / "global-vs-local.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--patches", type=int, default=32)
    parser.add_argument("--radius", type=float, default=10.)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps(checks())); return
    require_compute(threads=args.workers)
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    numerical = checks()
    with AugmentationStore(args.store) as store:
        rows = sorted(store.rows, key=lambda r: (r["summary"].get("dataset", ""), r["id"]))
    if args.limit:
        # Size-spread per dataset for smoke and resolution controls.
        selected = []
        for dataset in sorted({r["summary"].get("dataset") for r in rows}):
            sub = [r for r in rows if r["summary"].get("dataset") == dataset]
            count = max(1, args.limit // 3)
            selected.extend(sub[i] for i in np.linspace(0, len(sub)-1, count).astype(int))
        rows = selected
    config = dict(points=args.points, patches=args.patches, radius=args.radius, historical=str(args.historical))
    tasks = [(str(args.store), r["id"], config) for r in rows]
    # Largest first using saved heavy-atom record size as a scheduling proxy.
    tasks.sort(key=lambda t: next(r["raw_bytes"] for r in rows if r["id"] == t[1]), reverse=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_parent, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(json.dumps(dict(done=len(results), total=len(tasks), parent=result["parent"], wall_s=result.get("wall_s"))), flush=True)
    ws = [w for r in results for w in r.get("wholes", [])]
    ps = [p for r in results for p in r.get("patches", [])]
    write_csv(args.output / "structures.csv", ws)
    write_csv(args.output / "patches.csv", ps)
    atomic_json(args.output / "summary.json", summary(ws, ps))
    atomic_json(args.output / "provenance.json", dict(config=config, script_sha256=digest(__file__),
        input_manifest_sha256=digest(args.store / "manifest.json"), numerical_controls=numerical,
        parents=len(results), datasets=dict(Counter(w["dataset"] for w in ws if w["group"] == "repair_only")),
        frames=len(ws), patches=len(ps), wall_s=time.monotonic()-start,
        excluded_zero_failures=sorted({w["parent"] for w in ws if not w["zero_valid"]}),
        historical=[dict(parent=r["parent"], files=r.get("historical", [])) for r in results],
        limitations=["Solvent-accessible spheres, not solvent-excluded molecular surface",
                     "Chemistry = residue-class labels and exposed amino-acid composition, not electrostatic potential",
                     "Patch centers fixed by reference surface; moving centers follow central residue frame",
                     "Historical amplitude sweep uses one preselected common direction per protein",
                     "Equal parent aggregation; pilot not weighted to full dataset population",
                     "Prepared native may contain completed atoms; source-present mask retained in input",
                     "Geometric robustness does not establish learned representation robustness"]))
    plots(args.output, ws, ps)
    print(json.dumps(dict(done=True, output=str(args.output), frames=len(ws), patches=len(ps))), flush=True)


if __name__ == "__main__":
    main()
