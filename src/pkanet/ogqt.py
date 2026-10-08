"""Canonical orientation Graph Query Transformer (oGQT) API.

oGQT is the default backbone-only GQT architecture.  The implementation remains
in :mod:`pkanet.site_model` so historical experiment hashes and checkpoints do
not change; this module provides stable names for new training and inference
code.
"""
from .site_model import (
    initialize_site,
    predict_site_pkpdb_indexed,
    predict_site_shift_indexed,
    predict_site_with_trace,
)


initialize = initialize_site
predict_pkpdb = predict_site_pkpdb_indexed
predict_shift = predict_site_shift_indexed
predict_with_trace = predict_site_with_trace

__all__ = ("initialize", "predict_pkpdb", "predict_shift", "predict_with_trace")
