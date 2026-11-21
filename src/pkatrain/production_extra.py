"""Extra PINDER training records next to store-v2 (quartet-rank-v1b, 2026-11-16, pool-v5: the Ab/Ag paratope screen).

A pool built after store-v2 (training/pool-v5) needs records that store-v2 and its side tables do not hold. They are
built with the unchanged store-v2 builders into an extra directory with the same layout (abdata-v5 build):

EXTRA is a runtime-shaped root; under X = EXTRA/training/<version>/pinder:
  X/store-v2/          production_graphs store (records of the new complexes only) + X/manifest-v1.json, X/ids.json
  X/backbone-v1/       production_inputs.BackboneTable                (orientation, dihedrals, disulfide gate)
  X/sidechains-v1/     production_sidechains.SidechainTable           (disulfide gate / --drop-bonded-cys)
  X/contacts-v1/       production_aux.ContactTable                    (aux burial head)
  X/residue-rsa-v1/    production_residue.ResidueTable                (residue burial / edges / within / masks / quartets)
plus an oracle store for the hotspot mask (--mask-store-extra, store-hotspot-v2 schema).

--pinder-manifest FILE replaces <pinder output>/manifest-v1.json: the pool's manifest (store-v2 records with the pool's
min_fraction, records the pool dropped removed, the extra records appended; validation records unchanged; the store-v2
bucket policy and capacities). --pinder-extra EXTRA then gives the TRAINING PINDER source union views: every lookup by
complex id goes to the store-v2 table when the id is there, otherwise to the extra table (ids are disjoint, asserted).
Validation sources and every other dataset are untouched. Without the flags nothing here is imported or called.

Supported: the asym50 / quartet recipes (aux burial head, residue tables, gate, hotspot masks, quartets). Refused:
side-chain inputs, packed side chains, alt geometry, pH, swap strata, rsa-bound Siamese aux, site masking (their tables
have no extra counterpart here).
"""
from __future__ import annotations

import json
from pathlib import Path

from pkabench.runtime import digest

TABLES = ("store-v2", "backbone-v1", "sidechains-v1", "contacts-v1", "residue-rsa-v1")  # under output(EXTRA, "pinder")


class _Union:
    """Dispatch by complex id: base when the id is in base, else extra."""

    def __init__(self, base, extra, base_ids, extra_ids, name):
        overlap = set(base_ids) & set(extra_ids)
        if overlap: raise AssertionError((name, "extra ids already in the base table", sorted(overlap)[:5]))
        self.base, self.extra, self.name = base, extra, name; self._base_ids = frozenset(base_ids); self._extra_ids = frozenset(extra_ids)
        self.verification_sha256 = f"{getattr(base, 'verification_sha256', None)}+extra({getattr(extra, 'verification_sha256', None)})"

    def _pick(self, cid):
        if cid in self._base_ids: return self.base
        if cid in self._extra_ids: return self.extra
        raise KeyError((self.name, cid))


class UnionCall(_Union):  # BackboneTable, SidechainTable, ContactTable
    def __call__(self, cid): return self._pick(cid)(cid)


class UnionRows(_Union):  # ResidueTable
    def row(self, cid): return self._pick(cid).row(cid)


class UnionStore(_Union):  # ProductionStore
    def __init__(self, base, extra):
        super().__init__(base, extra, base.by_id, extra.by_id, "store-v2"); self.path = base.path

    def raw(self, cid): return self._pick(cid).raw(cid)

    def close(self): self.base.close(); self.extra.close()


class UnionHotspot(_Union):  # HotspotStore: record(cid) -> None for a complex in neither store (random fallback, counted)
    def __init__(self, base, extra):
        super().__init__(base, extra, base.by_id, extra.by_id, "hotspot"); self.path = f"{base.path}+{extra.path}"

    def __len__(self): return len(self.base) + len(self.extra)

    def record(self, cid):
        if cid in self._base_ids: return self.base.record(cid)
        return self.extra.record(cid)


def check_manifest(manifest, base, extra):
    """The pool manifest must keep store-v2's store, policy and capacities and draw extra records from EXTRA/store-v2 only."""
    from .production_graphs import ProductionStore
    if manifest.get("store") != base["store"] or manifest.get("store_verification_sha256") != base["store_verification_sha256"]:
        raise AssertionError("pool manifest must name store-v2 as its store")
    for key in ("bucket_policy", "capacities", "version", "dataset"):
        if manifest.get(key) != base.get(key): raise AssertionError(("pool manifest differs from manifest-v1.json in", key))
    from .production_graphs import output
    xstore = output(extra, "pinder") / "store-v2"
    if manifest.get("extra_store") != str(xstore): raise AssertionError("pool manifest extra_store != the EXTRA store-v2")
    if manifest.get("extra_store_verification_sha256") != digest(xstore / "verification.json"): raise AssertionError("extra store changed")
    base_val = [r for r in base["records"] if r["split"] == "val"]; val = [r for r in manifest["records"] if r["split"] == "val"]
    if json.dumps(val, sort_keys=True) != json.dumps(base_val, sort_keys=True): raise AssertionError("pool manifest changes the validation records")
    store = ProductionStore(xstore); base_ids = {r["id"] for r in base["records"]}
    try:
        for r in manifest["records"]:
            if r["id"] in base_ids: continue
            if r["split"] != "train" or r["id"] not in store.by_id: raise AssertionError((r["id"], "record neither in store-v2 nor a training record of EXTRA/store-v2"))
    finally: store.close()


def refuse(**options):
    bad = [k for k, v in options.items() if v]
    if bad: raise ValueError(f"--pinder-extra does not support {bad}")


def attach(source, root, extra, hotspot_extra=None):
    """Turn the training PINDER source's store and side tables into base/extra unions, in place. EXTRA is a runtime-shaped
    root (the abdata-v5 builders' mini runtime): its tables live in output(EXTRA, 'pinder') / <table>, so every table class
    opens its extra copy with its own constructor on root EXTRA. Returns provenance."""
    from .production_graphs import ProductionStore, output
    from .production_inputs import BackboneTable
    from .production_sidechains import SidechainTable
    from .production_aux import ContactTable
    from .production_residue import HotspotStore, ResidueTable
    extra = Path(extra); folder = output(extra, "pinder"); base_dir = output(root, "pinder"); used = {}

    def opened(obj, name): used[name] = obj.verification_sha256; return obj

    store = ProductionStore(folder / "store-v2"); used["store-v2"] = digest(folder / "store-v2" / "verification.json")
    source.store = UnionStore(source.store, store)
    if source.sidechains is not None:
        side = source.sidechains
        if not isinstance(side, BackboneTable): raise ValueError("--pinder-extra supports the backbone table only in the node-feature slot (no --sidechains)")
        x = opened(BackboneTable(extra, "pinder"), "backbone-v1"); source.sidechains = UnionCall(side, x, side.index, x.index, "backbone-v1")
    if getattr(source, "contacts", None) is not None:
        c = source.contacts
        if c.verification_sha256 != digest(base_dir / "contacts-v1" / "verification.json"):
            raise ValueError("--pinder-extra supports the contacts-v1 aux table only (not rsa-bound-v1)")
        x = opened(ContactTable(extra, "contacts-v1"), "contacts-v1"); source.contacts = UnionCall(c, x, c.index, x.index, "contacts-v1")
    if source.disulfides is not None:
        d = source.disulfides; xs = opened(SidechainTable(extra, "pinder"), "sidechains-v1"); xb = BackboneTable(extra, "pinder")
        d.sidechains = UnionCall(d.sidechains, xs, d.sidechains.index, xs.index, "sidechains-v1")
        d.backbone = UnionCall(d.backbone, xb, d.backbone.index, xb.index, "backbone-v1")
        d.verification_sha256 = "disulfide(" + d.sidechains.verification_sha256 + "+" + d.backbone.verification_sha256 + ")"
    if source.residues is not None:
        r = source.residues; x = opened(ResidueTable(extra), "residue-rsa-v1"); source.residues = UnionRows(r, x, r.index, x.index, "residue-rsa-v1")
    if source.hotspot is not None and source.hotspot[0] is not None:
        if hotspot_extra is None: raise ValueError("--mask-store with --pinder-extra needs --mask-store-extra")
        h = HotspotStore(hotspot_extra); used["hotspot"] = h.verification_sha256
        source.hotspot = (UnionHotspot(source.hotspot[0], h),) + tuple(source.hotspot[1:])
    elif hotspot_extra is not None: raise ValueError("--mask-store-extra needs --mask-oracle with --mask-store")
    for name in ("strata", "alt"):
        if getattr(source, name, None) is not None: raise ValueError(f"--pinder-extra does not support the {name} hook")
    return {"extra": str(extra), "tables_verification_sha256": used, **({"hotspot_extra": str(hotspot_extra)} if hotspot_extra else {})}
