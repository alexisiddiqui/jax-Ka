"""Extend the Fv check: for Fab cases without a usable partner-antigen PINDER entry, take the cognate VH/VL partner from
the RCSB entry (first model): the chain (not own, not antigen; 90-300 residues) with the most CA-CA < 8 A pairs to the
own chain, superposed into the PINDER frame by Kabsch on own-chain + antigen CA matched by (chain, number, icode)."""
import gemmi, json, numpy as np
exec(open('assemble.py').read().split('rows = []; stats')[0])   # load / write / contacts / roles / keys / ca / CUT
cases = json.load(open('fvcheck/cases.json')); byid = {c['id']: c for c in cases}
def gem(pdb):
    st = gemmi.read_structure(f'rcsb/{pdb}.cif.gz'); st.remove_hydrogens(); st.remove_ligands_and_waters(); m = st[0]; out = {}
    for ch in m:
        res = []
        for r in ch:
            if r.het_flag == 'H': continue
            res.append(dict(num=r.seqid.num, ins=r.seqid.icode.strip(), name=r.name, atoms=[(a.name, a.element.name.upper() if len(a.element.name)==1 else a.element.name, np.array(a.pos.tolist())) for a in r if a.altloc in ('\0', 'A')]))
        if res: out.setdefault(ch.name, res)
    return out
def kabsch(P, Q):  # R, t with R P + t ~ Q
    pc, qc = P.mean(0), Q.mean(0); H = (P - pc).T @ (Q - qc); U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T; return R, qc - R @ pc

import difflib, itertools
def seq(res): return ''.join(gemmi.find_tabulated_residue(r['name']).one_letter_code.upper() if gemmi.find_tabulated_residue(r['name']) else 'X' for r in res)
def match(a, b):
    """index pairs (i in a, j in b) from difflib matching blocks on one-letter sequences."""
    sm = difflib.SequenceMatcher(None, seq(a), seq(b), autojunk=False); out = []
    for blk in sm.get_matching_blocks(): out += [(blk.a + k, blk.b + k) for k in range(blk.size)]
    return out
def candidates(res, R_):
    """RCSB chains whose sequence covers >= 90% of res."""
    out = []
    for name, rr in R_.items():
        m = match(res, rr)
        if len(m) >= 0.9 * len(res): out.append((name, m))
    return out
log = []
for eid in json.load(open('rcsb_cases.json')):
    c = byid[eid]; ch, orig = load(eid); abn, agn = roles(eid, ch, orig); ab, ag = ch[abn], ch[agn]
    R_ = gem(eid[:4]); cab, cag = candidates(ab, R_), candidates(ag, R_)
    if not cab or not cag: log.append((eid, f'no sequence match (ab {len(cab)}, ag {len(cag)})')); continue
    best_fit = None
    for (nab, mab), (nag, mag) in itertools.product(cab, cag):
        if nab == nag: continue
        P = np.vstack([ca(R_[nab])[[j for _, j in mab]], ca(R_[nag])[[j for _, j in mag]]])
        Q = np.vstack([ca(ab)[[i for i, _ in mab]], ca(ag)[[i for i, _ in mag]]]); ok = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
        if ok.sum() < 10: continue
        Rm, t = kabsch(P[ok], Q[ok]); rmsd = float(np.sqrt(((P[ok] @ Rm.T + t - Q[ok]) ** 2).sum(1).mean()))
        if best_fit is None or rmsd < best_fit[0]: best_fit = (rmsd, Rm, t, nab, nag, int(ok.sum()))
    if best_fit is None: log.append((eid, 'no superposition')); continue
    rmsd, Rm, t, nab, nag, nok = best_fit; ok = np.ones(nok, bool)
    best, bn = None, 0; own_ca = ca(R_[nab])
    for name, res in R_.items():
        if name in (nab, nag) or not 90 <= len(res) <= 300: continue
        n = int((np.linalg.norm(own_ca[:, None] - ca(res)[None], axis=-1) < 8).sum())
        if n > bn: best, bn = name, n
    if best is None or bn < 15: log.append((eid, f'no partner (best {bn})')); continue
    pab = [dict(r, atoms=[(a, e, Rm @ x + t) for a, e, x in r['atoms']]) for r in R_[best]]
    pc = [[int(i), int(j)] for i, j in contacts(pab, ag)]
    mind = float(np.nanmin(np.linalg.norm(ca(ab)[:, None] - ca(pab)[None], axis=-1)))
    fvname = f'fv_{eid}'; write([('A', ab), ('B', pab)], out / 'pdb' / f'{fvname}.pdb')
    c.update(fv=fvname, partner_entry=f'rcsb:{eid[:4]}:{best}', n_partner=len(pab), partner_contacts=pc, antigen_rmsd=rmsd,
             antigen_common=int(ok.sum()), vh_vl_min_ca=mind, partner_source='rcsb')
    log.append((eid, f'partner {best} pairs {bn} superpose RMSD {rmsd:.3f} n {int(ok.sum())} minCA {mind:.1f} partner contacts {len(pc)}'))
json.dump(cases, open('fvcheck/cases.json', 'w'))
for e, m in log: print(f'{e[:44]:44s} {m}')
fv = [c for c in cases if 'fv' in c and c['vh_vl_min_ca'] < 8]; print('usable Fv cases', len(fv))
