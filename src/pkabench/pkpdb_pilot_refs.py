"""Frozen held-out chains and conservative experimental reference inventory."""
import gzip
import io
import json
import re
from pathlib import Path
from collections import defaultdict
from .runtime import atomic_json,digest,config_hash


def read(path):return json.loads(Path(path).read_text())


def cif(path):
    from biotite.structure.io import pdbx
    with gzip.open(path,'rt') as stream:return pdbx.CIFFile.read(io.StringIO(stream.read()))


def sequences(file):
    poly=file.block.get('entity_poly');asym=file.block.get('struct_asym')
    if poly is None or asym is None:raise ValueError('missing polymer sequence metadata')
    entities={};other=[]
    for e,t,s in zip(poly['entity_id'].as_array(str),poly['type'].as_array(str),poly['pdbx_seq_one_letter_code_can'].as_array(str)):
        if 'polypeptide' not in str(t):other.append(str(t));continue
        seq=''.join(str(s).split()).upper()
        if not seq or not re.fullmatch('[A-Z]+',seq):raise ValueError('unresolved canonical protein sequence')
        entities[str(e)]=seq
    rows=[dict(chain=str(c),sequence=entities[str(e)]) for c,e in zip(asym['id'].as_array(str),asym['entity_id'].as_array(str)) if str(e) in entities]
    if not rows:raise ValueError('no protein chains')
    return rows,other


def reference_inventory(root,out):
    import pyarrow.parquet as pq
    from .download import fetch
    freeze=root/'universe/structural-freeze-v1';manifest=read(freeze/'manifest.json')
    assignments=freeze/'assignments.parquet';assert digest(assignments)==manifest['artifacts_sha256']['assignments.parquet']
    held={r['complex_id'] for r in pq.read_table(assignments).to_pylist() if r['split'] in ('val','test')}
    candidatepath=root/'universe/combined-split-v1/index.json'
    candidates=read(candidatepath)['candidates'];records=[];pdb_ids=set();sources={str(assignments):digest(assignments),str(candidatepath):digest(candidatepath)}
    seen=set()
    for r in candidates:
        if r['complex_id'] not in held:continue
        seen.add(r['complex_id']);pdb_ids.add(r['pdb_id'].lower())
        for c in r['chains']:
            records.append(dict(sequence=c['sequence'],kind='benchmark',source=r['complex_id'],chain=c['chain']))
    assert seen==held,'Missing held-out chain sequences'
    reference_files=[root/'universe/combined-split-v1/usable-proposal-v2/reference-sequences.json',
        root/'audits/set2-screen-v1/usable-proposal-v2/reference-sequences.json',
        root/'audits/experimental-inventory-v1/official-delta/usable-proposal-v2/reference-sequences.json']
    experimental_pdb=set()
    for path in reference_files:
        value=read(path);sources[str(path)]=digest(path)
        for r in value.get('resolved',[])+value.get('parent_proxies',[]):
            if r.get('sequence'):records.append(dict(sequence=r['sequence'],kind='experimental',source=str(path),reference=r.get('id')))
            if re.fullmatch('[0-9][A-Za-z0-9]{3}',r.get('pdb_id','')):experimental_pdb.add(r['pdb_id'].lower())
        for r in value.get('failed',[]):
            if re.fullmatch('[0-9][A-Za-z0-9]{3}',r.get('pdb_id','')):experimental_pdb.add(r['pdb_id'].lower())
    pkad=root/'experimental/pkadr-full-v1/records.json';sources[str(pkad)]=digest(pkad)
    for r in read(pkad):
        for field in ('PDB','Alternative_PDBs'):
            experimental_pdb.update(x.lower() for x in re.findall(r'\b[0-9][A-Za-z0-9]{3}\b',r['raw'].get(field,'')))
    # Reserve all listed experimental entries, including records not yet admitted for fitting.
    unresolved=[]
    for pdb in sorted(experimental_pdb):
        path=root/'pretraining/pkpdb-v1/structures'/pdb[1:3]/f'{pdb}.cif.gz'
        try:
            if not path.exists():
                path=out/'reference-cifs'/f'{pdb}.cif.gz'
                if not path.exists():atomic_json(path.with_suffix('.receipt.json'),fetch(f'https://files.rcsb.org/download/{pdb}.cif.gz',path,limit=512*1024**2))
            seqs,_=sequences(cif(path));sources[str(path)]=digest(path)
            records.extend(dict(sequence=r['sequence'],kind='experimental',source=pdb,chain=r['chain']) for r in seqs)
        except Exception as exc:unresolved.append(dict(pdb_id=pdb,error=repr(exc)))
    grouped=defaultdict(list)
    for r in records:grouped['r'+config_hash(r['sequence'])[:24]].append(r)
    refs={key:dict(sequence=rr[0]['sequence'],kinds=sorted({r['kind'] for r in rr}),provenance=rr) for key,rr in grouped.items()}
    atomic_json(out/'references.json',dict(references=refs,reserved_pdb_ids=sorted(pdb_ids|experimental_pdb),
        unresolved=unresolved,heldout_complexes=len(held),sources=sources,
        scope='All chains of frozen val/test, including antibodies; all available experimental reference sequences and full PKAD-R listed PDB entries'))
    (out/'references.fasta').write_text(''.join(f'>{key}\n{r["sequence"]}\n' for key,r in sorted(refs.items())))
    return read(out/'references.json')


def disallowed_hit(identity,qcov,tcov,kinds):
    """90% held-out gate covers fragments; retain stricter experimental reservation."""
    return (('benchmark' in kinds and identity>=.9 and max(qcov,tcov)>=.8) or
            ('experimental' in kinds and identity>=.3 and min(qcov,tcov)>=.8))
