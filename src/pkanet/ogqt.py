"""Canonical orientation Graph Query Transformer (oGQT) API.

oGQT is the default backbone-only GQT architecture.  The implementation remains
in :mod:`pkanet.site_model` so historical experiment hashes and checkpoints do
not change; this module provides stable names for new training and inference
code.
"""
from .site_model import (
    initialize_site,
    initialize_site_auxiliary,
    predict_site_pkpdb_indexed,
    predict_site_multi_indexed,
    predict_site_shift_indexed,
    predict_site_with_trace,
    site_embeddings_indexed,
)


initialize = initialize_site
predict_pkpdb = predict_site_pkpdb_indexed
predict_shift = predict_site_shift_indexed
predict_with_trace = predict_site_with_trace
initialize_auxiliary = initialize_site_auxiliary
predict_multi = predict_site_multi_indexed
site_embeddings = site_embeddings_indexed

__all__ = ("initialize", "initialize_auxiliary", "predict_pkpdb", "predict_shift",
           "predict_multi", "predict_with_trace", "site_embeddings")
