"""Validated, portable NumPy structural constants. No pickle or Biotite at runtime."""
from __future__ import annotations
from dataclasses import dataclass, fields
from pathlib import Path
import hashlib
import json
import numpy as np
from .parameters import ALPHABET, GROUPS


@dataclass(frozen=True, order=True)
class ResidueKey:
    chain: str
    number: int
    insertion: str = ""

    def __str__(self):
        return f"{self.chain or '_'}:{self.number}{self.insertion}"

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, (tuple, list)) and len(value) in (2, 3):
            return cls(str(value[0]), int(value[1]), str(value[2]) if len(value) == 3 else "")
        raise TypeError("use ResidueKey or (chain, residue_number[, insertion_code])")


@dataclass(frozen=True)
class StructureCache:
    keys: tuple[ResidueKey, ...]
    chain_ids: tuple[str, ...]
    native_index: np.ndarray       # [N]
    chain_index: np.ndarray        # [N]
    group_mask: np.ndarray         # [N,9], termini independent of side chains
    frozen: np.ndarray             # [N], fixed-covalent identities
    env_neighbors: np.ndarray      # [N,Ke], safe padding index zero
    env_mask: np.ndarray           # [N,Ke]
    volume: np.ndarray             # [N,Ke,9,A], variable sidechain volume (A=20 unless compact)
    mass: np.ndarray               # [N,Ke,9,20], variable heavy-atom count
    hbond: np.ndarray              # [N,Ke,9,A], neutral + backbone NH terms
    local_hbond: np.ndarray        # [N,9,20], own backbone donor identity dependence
    bb_volume: np.ndarray          # [N,9]
    bb_mass: np.ndarray            # [N,9]
    bb_hbond: np.ndarray           # [N,9], fixed backbone CO contribution
    reorganization: np.ndarray     # [N,9], multiply by burial inside JIT
    neighbors: np.ndarray          # [N,Kc], includes same-residue terminal edges
    pair_mask: np.ndarray          # [N,Kc,9,9]
    coulomb_geometry: np.ndarray   # [N,Kc,9,9], 244.12/R * taper; divide by eps
    hb_donor: np.ndarray           # [N,Kc,9,9], directed i-donor -> j-acceptor
    hb_reverse: np.ndarray         # [N,Kc,9,9], j-donor -> i-acceptor
    metadata: dict
    # Compact identity-restricted storage: identity_columns[j,c] is the alphabet
    # index held in env column c for SOURCE residue j (padded columns repeat an
    # allowed identity and hold exact zeros). None = all 20 identities in order.
    identity_columns: np.ndarray | None = None  # [N,A] int

    @property
    def n_residues(self):
        return len(self.keys)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in vars(self).values() if isinstance(x, np.ndarray))

    def validate(self):
        n = self.n_residues
        if not n or len(set(self.keys)) != n:
            raise ValueError("residue keys must be nonempty and unique")
        if len(set(self.chain_ids)) != len(self.chain_ids) or not self.chain_ids:
            raise ValueError("chain IDs must be nonempty and unique")
        if self.env_neighbors.ndim != 2 or self.neighbors.ndim != 2:
            raise ValueError("neighbor arrays must be rank two")
        ke, kc = self.env_neighbors.shape[1], self.neighbors.shape[1]
        if min(ke, kc) < 1:
            raise ValueError("use at least one padded neighbor slot")
        shapes = {"native_index": (n,), "chain_index": (n,), "frozen": (n,),
                  "group_mask": (n,9), "env_neighbors": (n,ke), "env_mask": (n,ke),
                  "neighbors": (n,kc), "local_hbond": (n,9,20)}
        if self.identity_columns is None:
            columns = 20
        else:
            ic = np.asarray(self.identity_columns)
            if ic.ndim != 2 or ic.shape[0] != n or not 1 <= ic.shape[1] <= 20:
                raise ValueError("identity_columns must have shape [N,A] with 1 <= A <= 20")
            if not np.issubdtype(ic.dtype, np.integer) or np.any((ic < 0) | (ic >= 20)):
                raise ValueError("identity_columns must hold alphabet indices")
            columns = ic.shape[1]
        for name in ("volume", "mass", "hbond"):
            shapes[name] = (n,ke,9,columns)
        for name in ("bb_volume", "bb_mass", "bb_hbond", "reorganization"):
            shapes[name] = (n,9)
        for name in ("pair_mask", "coulomb_geometry", "hb_donor", "hb_reverse"):
            shapes[name] = (n,kc,9,9)
        for name, shape in shapes.items():
            x = np.asarray(getattr(self,name))
            if x.shape != shape or not np.isfinite(x).all():
                raise ValueError(f"{name}: expected finite {shape}, got {x.shape}")
        for name in ("group_mask", "frozen", "env_mask", "pair_mask"):
            if np.asarray(getattr(self,name)).dtype != np.dtype(bool):
                raise TypeError(f"{name} must have boolean dtype")
        for name in ("env_neighbors", "neighbors", "native_index", "chain_index"):
            if not np.issubdtype(getattr(self,name).dtype, np.integer):
                raise TypeError(f"{name} must have integer dtype")
        if np.any((self.native_index < 0) | (self.native_index >= 20)):
            raise ValueError("native identity out of range")
        if np.any((self.chain_index < 0) | (self.chain_index >= len(self.chain_ids))):
            raise ValueError("chain index out of range")
        for i, key in enumerate(self.keys):
            if self.chain_ids[self.chain_index[i]] != key.chain:
                raise ValueError("chain annotation does not match residue key")
        for name in ("env_neighbors", "neighbors"):
            if np.any((getattr(self,name)<0) | (getattr(self,name)>=n)):
                raise ValueError("even padded neighbor indices must be safe")
        env_edges = {}
        for i in range(n):
            js = self.env_neighbors[i,self.env_mask[i]].tolist()
            if len(set(js)) != len(js) or i in js:
                raise ValueError("environment graph must be unique and non-self")
            env_edges.update({(i,j): True for j in js})
        if any((j,i) not in env_edges for i,j in env_edges):
            raise ValueError("environment graph must be reciprocal")
        directed = {}
        for i in range(n):
            for k, j in enumerate(self.neighbors[i]):
                if self.pair_mask[i,k].any():
                    if (i,int(j)) in directed:
                        raise ValueError("duplicate pair edge")
                    directed[i,int(j)] = k
                    block = self.pair_mask[i,k]
                    allowed = self.group_mask[i,:,None] & self.group_mask[j,None,:]
                    if i == j:
                        allowed &= ~np.eye(9,dtype=bool)
                        allowed[:7,:7] = False
                    if np.any(block & ~allowed):
                        raise ValueError("invalid self, alternative-identity, or inactive-site edge")
        for (i,j), k in directed.items():
            if (j,i) not in directed:
                raise ValueError("pair graph must be reciprocal")
            kr = directed[j,i]
            if not np.array_equal(self.pair_mask[i,k], self.pair_mask[j,kr].T):
                raise ValueError("asymmetric pair mask")
            for a,b in (("coulomb_geometry","coulomb_geometry"),("hb_donor","hb_reverse")):
                if not np.allclose(getattr(self,a)[i,k],getattr(self,b)[j,kr].T,atol=2e-5):
                    raise ValueError(f"nonreciprocal {a}")
        for name in ("volume", "mass", "coulomb_geometry", "hb_donor", "hb_reverse"):
            if np.any(getattr(self,name)<0):
                raise ValueError(f"{name} must be nonnegative")
        return self

    def expanded_env(self, name):
        """Environment tensor ``name`` in the full [N,Ke,9,20] identity layout."""
        x = getattr(self, name)
        if self.identity_columns is None:
            return x
        out = np.zeros(x.shape[:3]+(20,), x.dtype)
        cols = self.identity_columns[self.env_neighbors]           # [N,Ke,A]
        for c in range(x.shape[-1]):
            # Padded columns duplicate an identity but hold zeros: add, never overwrite.
            idx = np.broadcast_to(cols[:, :, None, c], x.shape[:3])
            np.put_along_axis(out, idx[..., None],
                              np.take_along_axis(out, idx[..., None], -1)+x[..., c:c+1], -1)
        return out

    def select(self, residues=None):
        if residues is None:
            return np.arange(self.n_residues, dtype=np.int32)
        result = []
        lookup = {k:i for i,k in enumerate(self.keys)}
        for value in residues:
            if isinstance(value, (int, np.integer)):
                index = int(value)
                if not 0 <= index < self.n_residues:
                    raise IndexError(index)
            else:
                key = ResidueKey.from_value(value)
                if key not in lookup:
                    raise KeyError(f"unknown residue {key}")
                index = lookup[key]
            result.append(index)
        if not result or len(set(result)) != len(result):
            raise ValueError("selection must be nonempty and have no duplicates")
        return np.asarray(result, dtype=np.int32)

    def fingerprint(self):
        h = hashlib.sha256()
        for f in fields(self):
            x = getattr(self,f.name)
            if isinstance(x,np.ndarray):
                h.update(f.name.encode()); h.update(str(x.dtype).encode())
                h.update(str(x.shape).encode()); h.update(x.tobytes())
        h.update(json.dumps(self.chain_ids).encode())
        h.update(json.dumps([vars(k) for k in self.keys],sort_keys=True).encode())
        h.update(json.dumps({k:v for k,v in self.metadata.items() if k != "precompute_seconds"},sort_keys=True).encode())
        return h.hexdigest()

    def save(self, path):
        self.validate()
        arrays = {k:v for k,v in vars(self).items() if isinstance(v,np.ndarray)}
        info = {"schema":1, "alphabet":ALPHABET, "groups":GROUPS,
                "keys":[vars(k) for k in self.keys], "chain_ids":self.chain_ids,
                "metadata":self.metadata, "fingerprint":self.fingerprint()}
        # Passing a file object avoids silently changing a user-specified filename.
        with open(path,"wb") as f:
            np.savez_compressed(f, **arrays, info=np.asarray(json.dumps(info)))

    @classmethod
    def load(cls, path):
        with np.load(Path(path),allow_pickle=False) as f:
            info = json.loads(str(f["info"]))
            if info["schema"] != 1 or info["alphabet"] != ALPHABET or tuple(info["groups"]) != GROUPS:
                raise ValueError("incompatible cache schema")
            result = cls(keys=tuple(ResidueKey(**x) for x in info["keys"]),
                         chain_ids=tuple(info["chain_ids"]), metadata=info["metadata"],
                         **{k:f[k].copy() for k in f.files if k != "info"})
        result.validate()
        if result.fingerprint() != info["fingerprint"]:
            raise ValueError("cache fingerprint mismatch")
        return result
