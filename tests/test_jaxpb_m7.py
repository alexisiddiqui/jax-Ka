"""M7: driver summary memory is bounded; only the delphi PB mode exists."""
import dataclasses
import pytest
from jaxpb import api as A


def test_only_delphi_mode():
    A.Config().check()
    with pytest.raises(NotImplementedError):
        dataclasses.replace(A.Config(), pb_mode="converged").check()


def test_bounded_history():
    history = {"x": [], "history_totals": {}}
    log = []
    for _ in range(3000):
        A._history(history, "x", dict(states=1, seconds=0.5), log.append)
    assert len(history["x"]) == A.SUMMARY_HISTORY_LIMIT
    assert history["history_totals"]["x"]["count"] == 3000
    assert history["history_totals"]["x"]["sums"]["states"] == 3000
    assert len(log) == 3000
