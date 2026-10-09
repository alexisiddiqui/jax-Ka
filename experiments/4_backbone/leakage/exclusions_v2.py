"""PINDER held-out exclusions v2 = v1 with the prefilter's antibody path for Ab/Ag dimers (ab_rule.py): the antibody
chain is checked by CDRs (>= 70% vs experimental antibody CDRs) instead of whole-chain experimental_30 / fragment_90;
the antigen chain keeps the whole-chain rules; reserved PDB IDs unchanged. Writes pinder_heldout_exclusions_v2.tsv,
exclusions_v2.json and factorial_cohort_flagged_v2.tsv."""
import json, hashlib, pandas as pd
R = "/home/coulson/oc/lina4225/_runtime/jax-Ka/pkabench/"
v1 = pd.read_csv("pinder_heldout_exclusions_v1.tsv", sep="\t", dtype=str).fillna("")
rel = set(pd.read_csv("ab_rule_released.tsv", sep="\t").id)
v2 = v1[~v1.id.isin(rel)].merge(pd.read_csv("ab_rule_decisions.tsv", sep="\t", dtype=str).fillna(""), on="id", how="left").fillna(""); v2.to_csv("pinder_heldout_exclusions_v2.tsv", sep="\t", index=False)
coh = json.load(open(R + "training/ogqt-pinder-factorial-v1/cohort.json")); bad = v2.set_index("id")
fc = pd.DataFrame([dict(id=c["id"], cohort_split=c["split"], rules=bad.loc[c["id"], "rules"]) for c in coh["records"] if c["id"] in bad.index])
fc.to_csv("factorial_cohort_flagged_v2.tsv", sep="\t", index=False)
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
meta = dict(version="pinder-heldout-exclusions-v2", sha256=h("pinder_heldout_exclusions_v2.tsv"), supersedes="pinder-heldout-exclusions-v1",
            v1_sha256=h("pinder_heldout_exclusions_v1.tsv"), complexes_excluded=len(v2), clusters_touched=int(v2.cluster_id.nunique()),
            released_from_v1_by_antibody_path=len(rel), abag_excluded=int((v2.kind == "Ab/Ag").sum()),
            antibody_path="Ab/Ag dimers: antigen chain whole-chain experimental_30 / fragment_90; antibody chain concatenated CDRs >= 0.70 identity "
                          "(same chain type) vs CDRs of experimental antibody references (1igc, 1axt H/L), CDRs transferred by alignment to "
                          "SAbDab-annotated V domains as in pinder-prefilter-v1; reserved_pdb_id unchanged; other dimers as v1",
            factorial_cohort_flagged=fc.cohort_split.value_counts().to_dict() if len(fc) else {})
json.dump(meta, open("exclusions_v2.json", "w"), indent=1); print(json.dumps(meta, indent=1))
