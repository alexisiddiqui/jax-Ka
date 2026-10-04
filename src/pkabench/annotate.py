"""Fixed-coordinate burial of two partners, each containing one or more chains."""
import numpy as np
from scipy.spatial import cKDTree
from biotite.structure import sasa

SITE_ATOMS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"), "HIS": ("ND1", "NE2"),
    "CYS": ("SG",), "TYR": ("OH",), "LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2"),
    "NTERM": ("N",), "CTERM": ("O", "OXT")}


def annotate(topology, partners):
    atoms = topology.atoms
    masks = {p: np.isin(atoms.chain_id, chains) for p, chains in partners.items()}
    if set(masks) != {"A", "B"} or np.any(masks["A"] & masks["B"]) or not np.all(masks["A"] | masks["B"]):
        raise ValueError("partners must partition all selected chains")
    bound = np.asarray(sasa(atoms, probe_radius=1.4, point_number=1000), float)
    free = np.empty(len(atoms)); distance = np.empty(len(atoms))
    for p, mask in masks.items():
        if not mask.any(): raise ValueError("empty partner")
        free[mask] = sasa(atoms[mask], probe_radius=1.4, point_number=1000)
        distance[mask] = cKDTree(atoms.coord[~mask]).query(atoms.coord[mask])[0]
    delta = free - bound
    if not np.isfinite(delta).all() or delta.min() < -1e-3:
        raise ValueError("invalid fixed-coordinate burial")
    residues = []
    for i, k in enumerate(topology.keys):
        s, e = topology.starts[i:i+2]
        residues.append({"chain": k.chain, "resnum": k.number, "icode": k.insertion,
            "partner": "A" if k.chain in partners["A"] else "B",
            "residue_delta_sasa": float(delta[s:e].sum()), "min_partner_distance": float(distance[s:e].min())})
    return residues, delta, {"half_sum_buried_area": float(delta.sum()/2),
        "interface_residues": sum(r["residue_delta_sasa"] > 10 for r in residues)}
