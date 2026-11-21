import csv, collections, json, sys
exec(open('val.py').read().split("c=collections.Counter()")[0].replace("print('val rows'","#"))
ent=set(open(sys.argv[1]).read().split())
def parts(i):
    return [tuple(p.split('__')[1].rsplit('_',1)) for p in i.split('--')]
by_pdb=collections.defaultdict(set)
for e in ent: by_pdb[e[:4]].add(e)
out=[]; c=collections.Counter()
for r in val:
    if r['stratum']!='Ab/Ag': continue
    sq,_=ab_seq(r)
    if sq is None: continue
    doms=A.get(seq2key.get(sq),[]); t=''.join(sorted({d['chain_type'] for d in doms})); sp=doms[0]['species'] if doms else ''
    p=parts(r['id']); ab=[ch for ch,u in p if u=='UNDEFINED'][0]; ag=[ch for ch,u in p if u!='UNDEFINED'][0]
    if sp=='alpaca' and t=='H': out.append(dict(id=r['id'],kind='nanobody',ab=ab,ag=ag)); c['nanobody']+=1; continue
    if t not in ('H','K','L'): c['other']+=1; continue
    others=by_pdb[r['pdb_id']]-{r['id']}
    pag=[o for o in others if ag in [ch for ch,_ in parts(o)] and sum(u=='UNDEFINED' for _,u in parts(o))==1]
    phl=[o for o in others if ab in [ch for ch,_ in parts(o)] and all(u=='UNDEFINED' for _,u in parts(o))]
    # partner chain: prefer one that is in an H-L entry with our chain AND in a partner-Ag entry
    cand=collections.Counter()
    for o in pag: cand.update(ch for ch,u in parts(o) if u=='UNDEFINED' and ch!=ab)
    hl_partners={ch for o in phl for ch,u in parts(o) if ch!=ab}
    both=[ch for ch in cand if ch in hl_partners]
    pick = both[0] if both else (list(cand)[0] if cand else (sorted(hl_partners)[0] if hl_partners else None))
    src = 'partner-Ag + H-L' if both else 'partner-Ag' if cand else 'H-L only' if hl_partners else 'none'
    c[src]+=1
    out.append(dict(id=r['id'],kind='Fab-'+('heavy' if t=='H' else 'light'),ab=ab,ag=ag,partner=pick,src=src,
                    partner_ag=[o for o in pag if pick in [ch for ch,_ in parts(o)]], hl=[o for o in phl if pick in [ch for ch,_ in parts(o)]]))
print(c)
json.dump(out, open(sys.argv[2],'w'), indent=0)
