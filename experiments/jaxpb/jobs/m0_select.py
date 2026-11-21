"""M0: choose ~100 validation states (33 complete AB+A+B triples) from the hybrid-mc-v1 requests: val split only, pilot
complexes excluded, duplicate complexes (same residues + microstate counts + role) collapsed, 11 Ab/Ag + 22 general spread
evenly over AB microstates. usage: python m0_select.py <requests_dir> <out.json>"""
import json, glob, sys
PILOT = {"d0d5c619bd23b13d", "24d02660df3ac695", "d6004cf2ee4008d2"}
rows = []
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(f))
    if d["split"] != "val" or d["complex_id"] in PILOT:
        continue
    st = d["states"]
    if not all(st[s].get("source") for s in ("AB", "A", "B")):
        continue
    rows.append(dict(cid=d["complex_id"], role=d["role"], nres=d["n_residues"],
                     ms={s: st[s]["microstates"] for s in ("AB", "A", "B")}, src={s: st[s]["source"] for s in ("AB", "A", "B")}))
seen = set(); uniq = []
for r in sorted(rows, key=lambda r: (r["ms"]["AB"], r["cid"])):
    k = (r["role"], r["nres"], r["ms"]["AB"], r["ms"]["A"], r["ms"]["B"])
    if k in seen:
        continue
    seen.add(k); uniq.append(r)


def spread(lst, n):
    if len(lst) <= n:
        return lst
    return [lst[round(i * (len(lst) - 1) / (n - 1))] for i in range(n)]


pick = spread([r for r in uniq if r["role"] == "antibody_antigen"], 11) + spread([r for r in uniq if r["role"] == "general"], 22)
states = [dict(state_id=f"{r['cid']}_{s}", cid=r["cid"], state=s, role=r["role"], split="val", source=r["src"][s],
               microstates=r["ms"][s]) for r in pick for s in ("AB", "A", "B")]
json.dump(states, open(sys.argv[2], "w"), indent=1)
print(len(pick), "complexes", len(states), "states; microstates total", sum(s["microstates"] for s in states))
print(sorted(r["ms"]["AB"] for r in pick))
