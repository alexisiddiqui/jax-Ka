# Experimental Set 2 follow-up — 2026-10-05

No new quantitative labels were admitted. The next scientific work is still exact
construct matching and primary measurement curation, independently of experiment 02.

The [barnase–barstar primary abstract](https://pubmed.ncbi.nlm.nih.gov/8494892/)
reports Kd 1.3×10⁻¹⁴ M at pH 8 and 1.6×10⁻¹¹ M at pH 5 with 100 mM NaCl.
It separately reports 2.4×10⁻¹¹ M with 500 mM NaCl. These cannot yet form a
matched pH curve: the abstract does not establish all shared conditions or
uncertainties. The free-barnase His102 pKa of 6.4 is not a measured bound/free
pKa pair. Full methods/curves and the barstar construct must still be verified.

A public XML request for the [protein G–Fc paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2673305/)
failed with HTTP 500. Existing construct and curve-extraction blockers remain;
no figure values were guessed. FcRn remains on hold under the earlier screen.

Machine-readable observations and unresolved fields are in
[curation/set2_primary_followup.json](curation/set2_primary_followup.json).
Raw primary metadata and download receipts are under
`_runtime/jax-Ka/pkabench/audits/set2-primary-v2` (compute job 733186).
Frozen experimental-family reservations are unchanged. Existing model fitting can
proceed on the structural teacher labels, but it cannot substitute for this
independent experimental evaluation.
