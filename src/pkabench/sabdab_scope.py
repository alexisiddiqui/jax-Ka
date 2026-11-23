"""Published antibody-unit boundaries, independent of observed sequence gaps."""
import json

from .runtime import config_hash

POLICY = 'sabdab-published-imgt-antibody-scope-v1'


def crop_record(row, metadata_pin=None):
    """Keep the source's numbering/keep interval rather than infer a crop."""
    domains=[]
    for family,prefix,field in [('H','VH','Hchain'),('L','VL','Lchain')]:
        author=row.get(field,'')
        if author in ('','+'):continue
        numbering=row.get(prefix+'_numbering_list')
        interval=row.get(prefix+'_imgt_keep_interval')
        sequence=row.get(prefix+'_numerable_seq')
        if not numbering or not interval or not sequence:
            raise ValueError('Missing published antibody crop metadata: '+prefix)
        numbering=json.loads(numbering) if isinstance(numbering,str) else numbering
        interval=json.loads(interval) if isinstance(interval,str) else interval
        if len(interval)!=2 or not numbering:
            raise ValueError('Invalid published IMGT crop interval')
        if ''.join(r[0] for r in numbering)!=sequence:
            raise ValueError('Published numbered sequence differs from domain sequence')
        domains.append(dict(family=family,author_chain=author,numbering=numbering,
            keep_interval=interval,numerable_sequence=sequence))
    if not domains:raise ValueError('No declared antibody domains')
    record=dict(policy=POLICY,instance=row['INSTANCE'],domains=domains,
        metadata_pin=metadata_pin,source_row_sha256=config_hash(row))
    record['sha256']=config_hash(record)
    return record


def _key(value):
    number,insertion=value
    return int(number),str(insertion).strip()


def retained_intervals(record, residue_map, letters, side_by_chain, instance):
    """Prove each declared boundary against IMGT author keys and source positions.

    Bounds come from the published keep interval. In particular, a missing
    declared boundary cannot silently become the first/last observed residue.
    """
    if record['policy']!=POLICY or record['instance']!=instance:
        raise ValueError('Published crop identity/policy differs from prepared input')
    if record['sha256']!=config_hash({k:v for k,v in record.items() if k!='sha256'}):
        raise ValueError('Published crop metadata changed')
    if len(residue_map)!=len(letters):raise ValueError('Crop residue/sequence mapping differs')
    observed={};antibody=set()
    for index,(r,letter) in enumerate(zip(residue_map,letters)):
        if side_by_chain[r['chain']]!='A':continue
        key=(r['source_author_chain'],int(r['source_author_number']),r['source_author_insertion'].strip())
        if key in observed:raise ValueError('Ambiguous antibody IMGT identity')
        observed[key]=(r,letter,index);antibody.add(index)
    domains=[];covered=set();intervals={}
    for domain in record['domains']:
        numbered=[(_key(label),letter) for letter,label in domain['numbering']]
        keys=[k for k,_ in numbered]
        if len(keys)!=len(set(keys)):raise ValueError('Duplicate published IMGT key')
        low,high=map(_key,domain['keep_interval'])
        if low not in keys or high not in keys or keys.index(low)>keys.index(high):
            raise ValueError('Published crop boundaries are absent or reversed')
        selected=numbered[keys.index(low):keys.index(high)+1]
        mapped=[]
        for (number,insertion),letter in selected:
            identity=(domain['author_chain'],number,insertion)
            if identity not in observed:
                raise ValueError('Published retained antibody residue is missing: '+repr(identity))
            mapping,actual,index=observed[identity]
            if actual!=letter:raise ValueError('Published retained antibody sequence differs from coordinates')
            if index in covered:raise ValueError('Overlapping declared antibody domains')
            covered.add(index);mapped.append(mapping)
        chains={r['chain'] for r in mapped}
        if len(chains)!=1:raise ValueError('One published domain spans multiple physical chains')
        chain=next(iter(chains));positions=[r['source_label_position'] for r in mapped]
        if positions!=sorted(set(positions)):
            raise ValueError('Published domain/source polymer order differs')
        interval=(positions[0],positions[-1]);intervals.setdefault(chain,[]).append(interval)
        domains.append(dict(family=domain['family'],author_chain=domain['author_chain'],chain=chain,
            source_start=interval[0],source_end=interval[1],published_imgt_interval=domain['keep_interval'],
            observed_residues=len(mapped)))
    if covered!=antibody:raise ValueError('Published crop does not cover exactly the prepared antibody')
    for chain,ranges in intervals.items():
        ranges.sort()
        if any(right[0]<=left[1] for left,right in zip(ranges,ranges[1:])):
            raise ValueError('Overlapping published antibody source intervals')
    return intervals,dict(policy=POLICY,status='published_boundaries_verified',
        record_sha256=record['sha256'],source_metadata_pin=record.get('metadata_pin'),domains=domains)


def omitted_runs(positions, length):
    """Contiguous one-based source positions omitted from the declared unit."""
    runs=[]
    for position in range(1,length+1):
        if position in positions:continue
        if not runs or position!=runs[-1][-1]+1:runs.append([position])
        else:runs[-1].append(position)
    return runs
