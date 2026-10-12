import collections,csv,hashlib,json
from pathlib import Path
B=Path(__file__).resolve().parent
atlas=json.loads((B/'atlas-high-confidence.json').read_text())
flags=json.loads((B/'atlas-overlap-flags.json').read_text())
totals=collections.Counter()
out=B/'atlas-screening.tsv'
fields=['id','chain_A','chain_B','residues','exact_training_pair','training_any_chain_70',
        'current_validation_any_chain_70','validation_test_any_chain_70',
        'experimental_any_chain_30','preliminary_holdout_flag','over_1500_residues']
with out.open('w') as f:
 w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader()
 for r in atlas:
  x=flags[r['id']]
  train=x['current_training_union_0.7']['any_chain']
  val=x['current_validation_union_0.7']['any_chain']
  held=x['all_validation_and_test_union_0.7']['any_chain']
  exp=x['reference_experimental_0.3']['any_chain']
  exact=x['current_training_union_exact']['same_complex']
  long=sum(r['lengths'])>1500
  row=dict(zip(fields,[r['id'],*r['chains'],sum(r['lengths']),int(exact),int(train),int(val),int(held),int(exp),int(held or exp),int(long)]))
  w.writerow(row)
  for k,v in row.items():
   if k not in ['id','chain_A','chain_B','residues']:totals[k]+=v
  totals['no_detected_training_hit_70']+=not train
  totals['passes_preliminary_holdout_screen']+=not (held or exp)
  totals['passes_preliminary_screen_and_length']+=not (held or exp or long)
  totals['passes_preliminary_screen_and_no_training_hit_70']+=not (held or exp or train)
(B/'screening-summary.json').write_text(json.dumps(totals,indent=2))
files=['atlas-high-confidence.json','atlas.fasta','reference_membership.json','references.fasta',
       'atlas_vs_references.tsv','references_vs_atlas.tsv','overlap-summary.json',
       'atlas-screening.tsv','extract_references.py','analyze_overlap.py','audit.sbatch']
hashes={f:hashlib.file_digest((B/f).open('rb'),'sha256').hexdigest() for f in files}
(B/'analysis-provenance.json').write_text(json.dumps(dict(sha256=hashes,mmseqs_version=(B/'mmseqs-version.txt').read_text().strip(),
 identity_definition='Identical aligned residues divided by alignment length; exact means full normalized sequence equality.',
 coverage='At least 0.8 of each chain, MMseqs cov-mode 0; union of both search directions.',
 limitation='Whole-chain similarity screen; not project-specific antibody/CDR or one-sided fragment clearance. No claims of structural interface equivalence.'),indent=2))
print(json.dumps(totals,indent=2),flush=True)
