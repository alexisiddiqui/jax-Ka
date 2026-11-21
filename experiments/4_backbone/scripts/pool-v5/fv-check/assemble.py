"""Assemble Fv (own antibody chain + partner chain) and antigen-alone PDBs for the PINDER val Ab/Ag Fv check, plus
contact pairs (min heavy-atom distance < 4 + 3 ln 2 A, i.e. score_decoy target > 0.5) keyed by output residue order."""
import gemmi, json, math, numpy as np, collections
from pathlib import Path
CUT = 4 + 3 * math.log(2)
d = json.load(open('fvset.json')); E = Path('entries'); out = Path('fvcheck'); (out / 'pdb').mkdir(parents=True, exist_ok=True)
def load(eid):
    """{file chain: [residue dict(num, ins, name, atoms [(atom, element, xyz)])]} and file->original residue map."""
    t = gemmi.cif.read(str(E / eid / 'AB.cif.gz')).sole_block().find('_atom_site.', ['label_atom_id', 'type_symbol', 'auth_comp_id', 'auth_asym_id',
                                                                                   'auth_seq_id', 'pdbx_PDB_ins_code', 'Cartn_x', 'Cartn_y', 'Cartn_z'])
    chains = collections.OrderedDict()
    for row in t:
        at, el, rn, ch, num, ins, x, y, z = [row[i] for i in range(9)]
        if el == 'H': continue
        ins = '' if ins in ('.', '?') else ins; res = chains.setdefault(ch, [])
        if not res or (res[-1]['num'], res[-1]['ins']) != (int(num), ins): res.append(dict(num=int(num), ins=ins, name=rn, atoms=[]))
        res[-1]['atoms'].append((at, el, np.array([float(x), float(y), float(z)])))
    mp = json.load(open(E / eid / 'map_AB.json')); orig = {(m[0], int(m[1])): (m[2], int(m[3]), m[4]) for m in mp}
    return chains, orig
def keys(name, residues, orig): return [(name, r['num'], r['ins']) for r in residues]  # mmCIF auth ids = original PDB ids
def ca(residues): return np.array([next((a[2] for a in r['atoms'] if a[0] == 'CA'), np.full(3, np.nan)) for r in residues])
def write(chains, path):
    lines = []; serial = 1
    for cname, residues in chains:
        for i, r in enumerate(residues, 1):
            for at, el, xyz in r['atoms']:
                lines.append(f"ATOM  {serial:5d} {at:<4s} {r['name']:>3s} {cname}{i:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {el:>2s}")
                serial += 1
        lines.append('TER')
    Path(path).write_text('\n'.join(lines) + '\nEND\n')
def contacts(a, b):
    xa = [np.array([t[2] for t in r['atoms']]) for r in a]; xb = [np.array([t[2] for t in r['atoms']]) for r in b]
    ca_a, ca_b = ca(a), ca(b); near = np.linalg.norm(ca_a[:, None] - ca_b[None], axis=-1) < CUT + 20
    return [(i, j) for i, j in zip(*np.nonzero(near)) if np.min(np.linalg.norm(xa[i][:, None] - xb[j][None], axis=-1)) < CUT]
def ucode(eid): return [p.split('__')[1].rsplit('_', 1) for p in eid.split('--')]
def roles(eid, ch, orig):
    """(antibody file chain, antigen file chain) by the original chain id in map_AB (file order need not follow the id)."""
    idch = {c: up for c, up in ucode(eid)}; ab = ag = None  # cif 'A' = id A1, 'D-3' = id D3
    for name, res in ch.items():
        if idch[name.replace('-', '') if '-' in name else name + '1'] == 'UNDEFINED': ab = name
        else: ag = name
    assert ab and ag and ab != ag, (eid, list(ch))
    return ab, ag
rows = []; stats = collections.Counter()
for x in d:
    eid = x['id']; abi = [k for k, (_, up) in enumerate(ucode(eid)) if up == 'UNDEFINED'][0]
    ch, orig = load(eid); abn, agn = roles(eid, ch, orig); ab, ag = ch[abn], ch[agn]
    agname = f'ag_{eid}'; write([('C', ag)], out / 'pdb' / f'{agname}.pdb')
    rec = dict(id=eid, kind=x['kind'], antigen=agname, n_ag=len(ag), n_own=len(ab), own_contacts=[list(map(int, p)) for p in contacts(ab, ag)])
    if x['kind'].startswith('Fab') and x.get('partner_ag'):
        pid = x['partner_ag'][0]; pabi = [k for k, (_, up) in enumerate(ucode(pid)) if up == 'UNDEFINED'][0]
        pch, porig = load(pid); pabn, pagn = roles(pid, pch, porig); pab, pag = pch[pabn], pch[pagn]
        k1 = dict(zip(keys(agn, ag, orig), ca(ag))); k2 = dict(zip(keys(pagn, pag, porig), ca(pag)))
        common = [k for k in k1 if k in k2]; dev = float(np.sqrt(np.nanmean([np.sum((k1[k] - k2[k]) ** 2) for k in common]))) if common else float('nan')
        mind = float(np.nanmin(np.linalg.norm(ca(ab)[:, None] - ca(pab)[None], axis=-1)))
        agidx = {k: i for i, k in enumerate(keys(agn, ag, orig))}; pagk = keys(pagn, pag, porig)
        pc = [[int(i), agidx[pagk[j]]] for i, j in contacts(pab, pag) if pagk[j] in agidx]
        fvname = f'fv_{eid}'; write([('A', ab), ('B', pab)], out / 'pdb' / f'{fvname}.pdb')
        rec.update(fv=fvname, partner_entry=pid, n_partner=len(pab), partner_contacts=pc, antigen_rmsd=dev, antigen_common=len(common), vh_vl_min_ca=mind)
        stats['fv'] += 1
    rows.append(rec)
json.dump(rows, open(out / 'cases.json', 'w'))
print(stats, 'antigens', len(rows))
for r in rows:
    if 'fv' in r: print(f"{r['id'][:44]:44s} {r['kind']:10s} agRMSD {r['antigen_rmsd']:.3f} n {r['antigen_common']:4d}/{r['n_ag']:4d} VH-VL minCA {r['vh_vl_min_ca']:5.1f} own {len(r['own_contacts']):3d} partner {len(r['partner_contacts']):3d}")
