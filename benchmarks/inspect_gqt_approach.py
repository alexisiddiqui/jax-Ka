"""Resolve far-separation identity effects against residual cross-partner edges."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from pkabench.runtime import atomic_json,require_compute


def main():
    p=argparse.ArgumentParser();p.add_argument("out",type=Path);a=p.parse_args();require_compute(threads=4,allow_comp1400=True)
    traj=pq.read_table(a.out/"site_trajectory.parquet",filters=[("separation_A","=",40)]).to_pylist()
    att=pq.read_table(a.out/"attention_summary.parquet",filters=[("separation_A","=",40),("edge_class","=","opposite_partner")]).to_pylist()
    opportunity=defaultdict(float);mass=defaultdict(float)
    for r in att:
        key=(r["complex_id"],r["chain"],r["resnum"],r["icode"],r["group"])
        opportunity[key]=max(opportunity[key],r["opportunity"]);mass[key]=max(mass[key],r["mass"])
    anomalous=[]
    for r in traj:
        effect=abs(r["delta_mask_opposite_identity"])
        if effect<.05:continue
        key=(r["complex_id"],r["chain"],r["resnum"],r["icode"],r["group"])
        anomalous.append(dict(complex_id=r["complex_id"],chain=r["chain"],resnum=r["resnum"],group=r["group"],
            min_partner_ca_A=r["min_partner_ca_A"],effect=effect,edge_effect=abs(r["delta_remove_opposite_edges"]),
            opportunity=opportunity.get(key),attention_mass=mass.get(key)))
    atomic_json(a.out/"far_identity_audit.json",dict(sites=len(traj),anomalous=len(anomalous),
        anomalous_with_cross_opportunity=sum((r["opportunity"] or 0)>0 for r in anomalous),
        anomalous_fully_separated=sum(r["min_partner_ca_A"]>20 for r in anomalous),
        max_edge_effect=max((r["edge_effect"] for r in anomalous),default=0),examples=sorted(anomalous,key=lambda r:r["effect"],reverse=True)[:30]))


if __name__=="__main__":main()
