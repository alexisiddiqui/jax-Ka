"""Heavy-atom checks matching the audited native+min pilot (not convergence)."""
import numpy as np
from scipy.spatial import cKDTree

LIMITS = {"N-CA": (1.30, 1.65), "CA-C": (1.35, 1.70), "C-O": (1.10, 1.35), "C-N": (1.20, 1.50)}
FLAGS = ("geometry_valid", "bond_violations", "nonbonded_heavy_pairs_below_1p2_A",
         "chirality_flips", "new_cis_nonpro")
RMSDS = ("ca_rmsd_A", "backbone_rmsd_A", "heavy_atom_rmsd_A")


def rmsd(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a-a.mean(0); b = b-b.mean(0)
    u, _, vt = np.linalg.svd(a.T@b); u[:, -1] *= np.linalg.det(u@vt)
    return float(np.sqrt(np.mean(np.sum((a@u@vt-b)**2, axis=1))))


def omega(a, b, c, d):
    v = b-a; w = c-b; x = d-c; u = np.cross(v, w); z = np.cross(w, x)
    return float(np.degrees(np.arctan2(np.dot(np.cross(u, z), w/np.linalg.norm(w)), np.dot(u, z))))


def quality(metadata, arrays, xyz):
    xyz = np.asarray(xyz, np.float64); native = np.asarray(arrays["native_coordinates"], np.float64)
    ids = arrays["backbone_atoms"]; bb = xyz[ids]; source = np.asarray(arrays["source_backbone"], np.float64)
    seg = arrays["segments"]; names = metadata["residue_names"]
    bonds = arrays["bonds"]; adjacency = {i: set() for i in range(len(xyz))}
    for a, b in bonds: adjacency[int(a)].add(int(b)); adjacency[int(b)].add(int(a))
    excluded = set()
    for a, near in adjacency.items():
        for b in near | set().union(*(adjacency[j] for j in near)):
            if a != b: excluded.add(tuple(sorted((a, b))))
    lengths = {"N-CA": np.linalg.norm(bb[:, 0]-bb[:, 1], axis=1),
               "CA-C": np.linalg.norm(bb[:, 1]-bb[:, 2], axis=1),
               "C-O": np.linalg.norm(bb[:, 2]-bb[:, 3], axis=1),
               "C-N": np.linalg.norm(bb[:-1, 2]-bb[1:, 0], axis=1)[seg[:-1] == seg[1:]]}
    bad = {k: int(((v < LIMITS[k][0]) | (v > LIMITS[k][1])).sum()) for k, v in lengths.items()}
    clashes = len(cKDTree(xyz).query_pairs(1.2)-excluded)
    lookup = {(int(r), a): j for j, (r, a) in enumerate(zip(arrays["atom_residue"], metadata["atom_names"]))}
    chirality = []; cis = []
    def volume(coords, center, neighbors):
        v = [coords[j]-coords[center] for j in neighbors]
        return np.dot(np.cross(v[0], v[1]), v[2])
    for i, name in enumerate(names):
        cb = lookup.get((i, "CB"))
        if cb is not None:
            n, ca, c = ids[i, :3]
            if volume(native, ca, [n, c, cb])*volume(xyz, ca, [n, c, cb]) <= 0: chirality.append(i)
        if name in ("THR", "ILE"):
            center = lookup[i, "CB"]
            neighbors = [lookup[i, a] for a in (["CA", "OG1", "CG2"] if name == "THR" else ["CA", "CG1", "CG2"])]
            if volume(native, center, neighbors)*volume(xyz, center, neighbors) <= 0: chirality.append(i)
        if i+1 < len(bb) and seg[i] == seg[i+1] and names[i+1] != "PRO":
            if abs(omega(bb[i, 1], bb[i, 2], bb[i+1, 0], bb[i+1, 1])) < 90 and \
                    abs(omega(source[i, 1], source[i, 2], source[i+1, 0], source[i+1, 1])) >= 90:
                cis.append(i)
    ss = [float(np.linalg.norm(xyz[lookup[i, "SG"]]-xyz[lookup[j, "SG"]])) for i, j in metadata["disulfides"]]
    valid = not any(bad.values()) and not clashes and not chirality and not cis and \
            all(1.8 <= x <= 2.3 for x in ss) and np.isfinite(xyz).all()
    common = arrays["source_present"]
    return dict(geometry_valid=bool(valid), bond_violations=bad,
        nonbonded_heavy_pairs_below_1p2_A=clashes, chirality_flips=chirality, new_cis_nonpro=cis,
        disulfide_lengths_A=ss, ca_rmsd_A=rmsd(bb[:, 1], source[:, 1]),
        backbone_rmsd_A=rmsd(bb.reshape(-1, 3), source.reshape(-1, 3)),
        heavy_atom_rmsd_A=rmsd(xyz[common], native[common]))


def check_metrics(got, expected, tolerance=0.001):
    for name in FLAGS:
        if got[name] != expected[name]: raise ValueError(f"quality classification changed: {name}")
    if len(got["disulfide_lengths_A"]) != len(expected["disulfide_lengths_A"]):
        raise ValueError("disulfide count changed")
    if any(abs(a-b) > tolerance for a, b in zip(got["disulfide_lengths_A"], expected["disulfide_lengths_A"])):
        raise ValueError("disulfide lengths changed")
    for name in RMSDS:
        if abs(got[name]-expected[name]) > tolerance: raise ValueError(f"RMSD precision changed: {name}")


def verify_coordinate_record(record):
    meta = record["metadata"]; arrays = record["arrays"]
    if not arrays: return 0
    n = len(meta["residue_names"]); atoms = len(meta["atom_names"])
    selected = meta["selected"]; coords = arrays["coordinates"]
    if coords.dtype != np.float32 or coords.shape != (len(selected), atoms, 3):
        raise ValueError("invalid conformer dimensions/dtype")
    if len(selected) > 4 or len(set(selected)) != len(selected) or any(c <= 0 for c in selected):
        raise ValueError("invalid selected candidate IDs")
    owner = arrays["atom_residue"]; bbids = arrays["backbone_atoms"]
    if owner.shape != (atoms,) or bbids.shape != (n, 4) or len(meta["residue_keys"]) != n:
        raise ValueError("atom/residue mapping shape changed")
    if np.any(owner < 0) or np.any(owner >= n) or np.any(bbids < 0) or np.any(bbids >= atoms):
        raise ValueError("invalid atom/residue mapping")
    if not np.array_equal(owner[bbids], np.broadcast_to(np.arange(n)[:, None], (n, 4))):
        raise ValueError("backbone atoms map to wrong residues")
    if [[meta["atom_names"][j] for j in row] for row in bbids] != [["N", "CA", "C", "O"]]*n:
        raise ValueError("backbone atom identity changed")
    if len({(int(r), a) for r, a in zip(owner, meta["atom_names"])}) != atoms:
        raise ValueError("duplicate heavy-atom identity")
    check_metrics(quality(meta, arrays, arrays["zero_coordinates"]), meta["zero_quality"])
    if selected and not meta["zero_quality"]["geometry_valid"]:
        raise ValueError("selected conformers lack a valid zero control")
    for c, xyz in zip(selected, coords):
        checked = quality(meta, arrays, xyz)
        check_metrics(checked, meta["selected_quality"][str(c)])
        if not checked["geometry_valid"]: raise ValueError("invalid selected conformer")
    pairs = [rmsd(coords[i, bbids[:, 1]], coords[j, bbids[:, 1]])
             for i in range(len(coords)) for j in range(i+1, len(coords))]
    if len(pairs) != len(meta["selected_pairwise_ca_rmsd_A"]) or \
            any(abs(a-b) > 0.001 for a, b in zip(pairs, meta["selected_pairwise_ca_rmsd_A"])):
        raise ValueError("selected pairwise RMSD changed")
    return len(coords)+1
