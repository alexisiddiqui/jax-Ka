"""Prepare 6SC6 protein chains ABC and a 5-Angstrom A-B contact mask."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from biotite.structure.io.pdb import PDBFile
from jaxpropka.topology import load_topology
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory',type=Path,help='Directory containing original 6SC6.pdb; outputs are written here')
root=parser.parse_args().directory
top=load_topology(root/'6SC6.pdb',chains=('A','B','C'),gap_policy='cap',freeze_disulfides=True,ignore_nonprotein=True)
f=PDBFile();f.set_structure(top.atoms);f.write(root/'6SC6_ABC.pdb')
residue=np.repeat(np.arange(top.n_residues),np.diff(top.starts));chain=np.array([k.chain for k in top.keys]);mask=np.zeros(top.n_residues,bool);rows=[]
for a,b in [('A','B'),('B','A')]:
 ia=np.flatnonzero(top.atoms.chain_id==a);ib=np.flatnonzero(top.atoms.chain_id==b)
 distances,_=cKDTree(top.atoms.coord[ib]).query(top.atoms.coord[ia])
 for r in np.flatnonzero(chain==a):
  d=float(distances[residue[ia]==r].min());mask[r]=d<=5.
  if mask[r]:rows.append(dict(index=int(r),chain=a,number=int(top.keys[r].number),insertion=top.keys[r].insertion,min_partner_distance=d))
np.save(root/'interface.npy',mask)
report=dict(source='https://files.rcsb.org/download/6SC6.pdb',source_sha256=hashlib.sha256((root/'6SC6.pdb').read_bytes()).hexdigest(),chains=['A','B','C'],residues=top.n_residues,interface_count=int(mask.sum()),interface=rows,topology_metadata=top.metadata)
(root/'interface.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
