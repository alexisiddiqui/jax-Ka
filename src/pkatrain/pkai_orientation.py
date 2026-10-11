"""Backbone-plane tilt in a residue-local frame, invariant to proper rigid motions.

x = unit(C-CA); z = unit(x cross (N-CA)); y = z cross x. Each N/O
neighbour slot gets its residue's z projected onto the query's (x,y,z),
scaled by 1/d^2. This describes plane tilt (two degrees of freedom), not
full phi/psi torsions or rotation about the plane normal. Missing or
degenerate frames give a zero vector, distinct from a valid unit normal.
No side-chain atoms, chain adjacency or peptide connectivity are used.
"""
import numpy as np


def backbone_frames(atoms):
    groups = {}; invalid = set()
    for i in np.flatnonzero(~np.asarray(atoms.hetero)):
        name = str(atoms.atom_name[i])
        if name not in ("N", "CA", "C"): continue
        key = (str(atoms.chain_id[i]), int(atoms.res_id[i]), str(atoms.ins_code[i]))
        group = groups.setdefault(key, {})
        if name in group: invalid.add(key)
        group[name] = np.asarray(atoms.coord[i], np.float64)
    frames = {}
    for key, group in groups.items():
        if key in invalid or not all(name in group for name in ("N", "CA", "C")): continue
        x = group["C"] - group["CA"]; norm = np.linalg.norm(x)
        if not np.isfinite(norm) or norm < 1e-8: continue
        x = x / norm; z = np.cross(x, group["N"] - group["CA"]); norm = np.linalg.norm(z)
        if not np.isfinite(norm) or norm < 1e-8: continue
        z = z / norm; frames[key] = np.stack((x, np.cross(z, x), z))
    return frames


def orientation_features(atoms, keys):
    from pkabench.pkai_backbone_pinder_eval import _features
    from .pkai_scratch import CUTOFF, SLOTS, aa20_index
    base, retained = _features(atoms, keys, "atom16aa20")
    matrix = np.zeros((len(keys), SLOTS * 39 + 8), np.float32)
    blocks = matrix[:, :SLOTS * 39].reshape(len(keys), SLOTS, 39)
    blocks[..., :36] = base[:, :SLOTS * 36].reshape(len(keys), SLOTS, 36)
    matrix[:, -8:] = base[:, -8:]
    frames = backbone_frames(atoms)
    context = np.flatnonzero(~np.asarray(atoms.hetero) & np.isin(atoms.atom_name, ("N", "O")))
    ca = {(str(atoms.chain_id[i]), int(atoms.res_id[i]), str(atoms.ins_code[i])): np.asarray(atoms.coord[i], np.float64)
          for i in np.flatnonzero(~np.asarray(atoms.hetero) & (np.asarray(atoms.atom_name) == "CA"))}
    context_keys = [(str(atoms.chain_id[i]), int(atoms.res_id[i]), str(atoms.ins_code[i])) for i in context]
    coords = np.asarray(atoms.coord[context], np.float64)
    for row, key in enumerate(keys):
        query = (str(key[0]), int(key[1]), str(key[2]))
        if not retained[row] or query not in frames: continue
        distance = np.sqrt(np.sum((coords - ca[query]) ** 2, axis=1))
        ordered = sorted((float(distance[j]), 0 if str(atoms.atom_name[i]) == "N" else 9,
                          aa20_index(atoms.res_name[i]), j)
                         for j, i in enumerate(context) if context_keys[j] != query and distance[j] < CUTOFF)[:SLOTS]
        for slot, (d, _, _, j) in enumerate(ordered):
            frame = frames.get(context_keys[j])
            if frame is not None:
                blocks[row, slot, 36:] = (frames[query] @ frame[2]) / d**2
    assert np.isfinite(matrix).all()
    return matrix, retained
