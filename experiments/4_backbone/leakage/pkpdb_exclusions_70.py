"""pKPDB held-out exclusions at the PINDER leakage rule (adopted for pKPDB 2026-10-08): an entry is excluded when any
polypeptide(L) entity matches a held-out chain at >= 70% identity and >= 80% coverage of both (pkpdb_vs_heldout.tsv),
or its PDB ID is itself held out. Writes pkpdb_heldout_exclusions_70.tsv (pdb, reason, best identity, held-out target)."""
import pandas as pd
pk = pd.read_csv("pkpdb_entities.tsv", sep="\t", names=["pdb", "entity", "seq"], dtype=str)
import hashlib; pk["h"] = pk.seq.map(lambda q: "q" + hashlib.sha1(q.encode()).hexdigest()[:16])
hits = pd.read_csv("pkpdb_vs_heldout.tsv", sep="\t", names=["h", "t", "f"]).sort_values("f", ascending=False).drop_duplicates("h")
m = pk.merge(hits, on="h").sort_values("f", ascending=False).drop_duplicates("pdb")
ex = m[["pdb", "f", "t"]].rename(columns={"f": "identity", "t": "heldout_chain"}); ex.insert(1, "reason", "sequence_70")
ex.sort_values("pdb").to_csv("pkpdb_heldout_exclusions_70.tsv", sep="\t", index=False)
print("excluded entries", len(ex), "of", pk.pdb.nunique(), "| identity bins", pd.cut(ex.identity, [0.7, 0.8, 0.9, 0.95, 1.0001], right=False).value_counts().sort_index().to_dict())
