"""jaxpb.records unit checks (numpy only, seconds): size classes, bucket tables, and (when M0 golden shards are given)
shard round trip and validation.

pytest: the golden-shard check reads JAXPB_GOLDENS (a glob or os.pathsep-separated list of records shards, e.g. the M0
m0_goldens_*.npz) and is skipped when it is unset or matches nothing.
usage: python tests/test_jaxpb_records.py <shard.npz> [more shards...]"""
import glob
import json
import os
import sys
import tempfile

import numpy as np
import pytest

from jaxpb import records as RC


def golden_shards():
    spec = os.environ.get("JAXPB_GOLDENS", "")
    return sorted(p for part in spec.split(os.pathsep) if part for p in glob.glob(part))


def check_tables():
    for c, n in enumerate(RC.NA_C):
        assert RC.size_class(n) == c and (c == 4 or RC.size_class(n + 1) == c + 1)
    for mp in RC.MP_BUCKETS:
        assert RC.np_bucket(mp) == mp // 2 and RC.pp_bucket(mp) == mp // 16 and RC.mp_bucket(mp) == mp


def check_shards(shards):
    out = {"states": 0, "taus": 0}
    for sh in shards:
        recs = RC.load_states(sh)
        for r in recs:
            r.validate()
            assert np.isfinite(r.tau_pkmod[~r.tau_isref]).all(), r.state_id
            assert r.ref["sitpotM"].shape == (r.n_taus, r.n_atoms)
            me = json.loads(r.ref["mc_energies"])
            assert sum(me["npossible_states"]) == r.n_taus and len(me["all_sites"]) == r.n_sites
            assert [f"{c}_{n}_{s}" for c, n, s in zip(r.site_chain, r.site_resname, r.site_resnum)] == me["all_sites"], r.state_id
            out["states"] += 1; out["taus"] += r.n_taus
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "rt.npz")
            RC.save_states(p, recs[:2])
            back = RC.load_states(p)
            for a, b in zip(recs[:2], back):
                for k in RC.ARRAY_FIELDS:
                    x, y = getattr(a, k), getattr(b, k); assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), k
                assert np.array_equal(a.ref["sitpotM"], b.ref["sitpotM"]) and a.ref["mc_energies"] == b.ref["mc_energies"]
    return out


def test_size_classes_and_buckets():
    check_tables()


def test_golden_shards_round_trip():
    shards = golden_shards()
    if not shards:
        pytest.skip("JAXPB_GOLDENS unset or matches no records shard")
    assert check_shards(shards)["states"] > 0


if __name__ == "__main__":
    check_tables()
    print(json.dumps({"pass": True, **check_shards(sys.argv[1:])}))
