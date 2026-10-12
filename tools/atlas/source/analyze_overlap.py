"""Aggregate bidirectional homology searches against frozen dataset memberships."""
import collections, csv, json, time
from pathlib import Path

BASE = Path(__file__).resolve().parent

def pair_compatible(a, b, target_chains):
    """Require two different chain occurrences, including two copies in homomers."""
    return any(x in a and y in b for i, x in enumerate(target_chains)
               for j, y in enumerate(target_chains) if i != j)

def main():
    atlas = json.loads((BASE/'atlas-high-confidence.json').read_text())
    datasets = json.loads((BASE/'reference_membership.json').read_text())
    def union(name, names):
        datasets[name] = {d+':'+c: ss for d in names for c, ss in datasets[d].items()}
    union('current_training_union', ['PINDER_train','pKPDB_train'])
    union('current_complex_training_union', ['PINDER_train','pKPDB_complex_train'])
    union('all_training_union', ['PINDER_train','pKPDB_train','benchmark_train'])
    union('current_validation_union', ['PINDER_val','pKPDB_val','benchmark_active_val142'])
    union('all_validation_and_test_union', ['PINDER_val','pKPDB_val','benchmark_val','benchmark_test'])
    aq = {s for r in atlas for s in r['chains']}
    rt = {s for d in datasets.values() for ss in d.values() for s in ss}
    hits = collections.defaultdict(dict)
    quality = {}
    for filename, reverse in [('atlas_vs_references.tsv',False),('references_vs_atlas.tsv',True)]:
        counts = collections.Counter(); kept = 0
        with (BASE/filename).open() as f:
            for line in f:
                q,t,ident,qcov,tcov,*_ = line.split()
                counts[q] += 1
                if float(qcov) < .8 or float(tcov) < .8: continue
                if reverse: q,t = t,q
                assert q in aq and t in rt
                ident = float(ident)
                hits[q][t] = max(ident,hits[q].get(t,0)); kept += 1
        quality[filename] = dict(rows=sum(counts.values()),coverage_qualified_rows=kept,
                                 max_hits_per_query=max(counts.values(), default=0),
                                 queries_at_candidate_limit=sum(v>=10000 for v in counts.values()))
    # Search heuristics can miss short chains; equality is deterministic and exhaustive.
    for q in aq & rt: hits[q][q] = 1.0
    summary=[];screening={r['id']:{} for r in atlas}
    detail_datasets={'current_training_union','current_complex_training_union',
                     'current_validation_union','all_validation_and_test_union',
                     'reference_experimental'}
    for threshold in ['exact',.95,.9,.7,.5,.3]:
        matches = {q: {t for t,v in hs.items() if (q==t if threshold=='exact' else v>=threshold)}
                   for q,hs in hits.items()}
        reverse_matches = collections.defaultdict(set)
        for q,ts in matches.items():
            for t in ts: reverse_matches[t].add(q)
        for name, records in datasets.items():
            inverse = collections.defaultdict(set)
            for cid,ss in records.items():
                for s in ss: inverse[s].add(cid)
            ref_sequences=set(inverse)
            local={q:ts & ref_sequences for q,ts in matches.items()}
            local={q:ts for q,ts in local.items() if ts}
            covered_refs=ref_sequences & set(reverse_matches)
            any_n=both_n=pair_n=0
            do_pair=threshold in ['exact',.7] and name!='pKPDB_accepted_source_entities' and not name.startswith('reference_')
            if do_pair:
                complex_matches={q:set().union(*(inverse[t] for t in ts)) for q,ts in local.items()}
            for r in atlas:
                a,b=r['chains']; ma=local.get(a,set()); mb=local.get(b,set())
                any_hit=bool(ma or mb); both=bool(ma and mb); pair=False
                if both and do_pair:
                    pair=any(pair_compatible(ma,mb,records[c]) for c in complex_matches[a]&complex_matches[b])
                any_n+=any_hit;both_n+=both;pair_n+=pair
                if name in detail_datasets and (threshold in ['exact',.7] or name=='reference_experimental' and threshold==.3):
                    screening[r['id']][name+'_'+str(threshold)]=dict(any_chain=any_hit,both_chains=both,same_complex=pair if do_pair else None)
            row=dict(dataset=name,identity=threshold,reference_records=len(records),reference_unique_chains=len(ref_sequences),
                     atlas_unique_chains_matched=len(local),atlas_complexes_any_chain=any_n,atlas_complexes_both_chains=both_n,
                     atlas_complexes_same_reference_complex=pair_n if do_pair else None,
                     reference_chains_covered=len(covered_refs),
                     reference_records_any_chain=sum(bool(set(ss)&covered_refs) for ss in records.values()),
                     reference_records_all_chains=sum(set(ss)<=covered_refs for ss in records.values()))
            summary.append(row)
        print('Finished identity',threshold,flush=True)
    (BASE/'overlap-summary.json').write_text(json.dumps(dict(atlas_complexes=len(atlas),atlas_unique_chains=len(aq),
           reference_unique_chains=len(rt),search_quality=quality,rows=summary),indent=2))
    with (BASE/'overlap-summary.tsv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]),delimiter='\t');w.writeheader();w.writerows(summary)
    (BASE/'atlas-overlap-flags.json').write_text(json.dumps(screening))
    print('Analysis complete',flush=True)

if __name__=='__main__': main()
