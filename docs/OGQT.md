# oGQT

The orientation Graph Query Transformer (**oGQT**) is the default backbone-only
GQT architecture for new experiments. It retains the residue graph encoder and
adds explicit titratable-site tokens, 20 A site-to-site attention, local-frame
direction features, and relative backbone orientation. ARG is retained as an
unlabelled context token.

The canonical code interface is `pkanet.ogqt`. Historical experiments retain
their original `site-orientation` arm names and `pkanet.site_model` imports so
their manifests and code hashes remain reproducible.

## Default training objective

Use unweighted signed pKPDB-shift MSE when overall validation pKa accuracy is
the primary objective. Across seeds 17, 29, and 43, unweighted oGQT achieved
overall MAE `0.5760 +/- 0.0094`, compared with `0.5838 +/- 0.0044` for the
residue-query GQT. Its large-shift MAE was also lower: `1.3222 +/- 0.0329`
versus `1.4975 +/- 0.0529`.

Use the name **oGQT-mild** for the same architecture trained with the registered
inverse-square-root shift-bin weights. It improves the predicted-versus-target
shift slope from `0.6904` to `0.7455` and lowers large-shift MAE to
`1.2362 +/- 0.0350`, at an overall MAE cost of `0.0221`. The fully balanced
loss is not the default because its overall MAE worsens to `0.6230`.

The frozen three-seed evidence and residue-type breakdown are in
`experiments/3_paired_complex/40_gqt_site_weighting.md`.
