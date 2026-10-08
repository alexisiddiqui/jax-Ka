"""Full FoldBench feasibility audit; does not create production labels or splits."""
from collections import Counter, defaultdict
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdbx
from scipy.spatial import cKDTree

from .runtime import require_compute, atomic_json, digest, config_hash
from .prep import prepare_pair, Rejection, CANONICAL
from .audit import component_inventory

NEUTRAL_ADDITIVES = {'GOL', 'EDO', 'PEG', 'PGE', 'PG4', '1PE', 'MPD', 'DMS'}
BUFFER_ADDITIVES = {'SO4', 'PO4', 'ACT', 'FMT', 'TRS', 'MES', 'HEP', 'BME'}
SCENARIOS = ('strict', 'neutral_additives', 'all_additives', 'pair_additives', 'pair_only_upper_bound')
AA = dict(zip(('ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'), 'ARNDCQEGHILKMFPSTWYV'))
AA.update(MSE='M', SEC='U', PYL='O')


def index_dataset(manifest, archive, root, smoke):
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    old = {r['complex_id'] for r in json.loads(Path(smoke).read_text())['candidates']}
    rows = []; seen = set()
    for i, row in enumerate(csv.DictReader(Path(manifest).open())):
        if row['interface_chain_type_1'] != 'protein' or row['interface_chain_type_2'] != 'protein': continue
        identity = (row['pdb_id'], *sorted((row['interface_chain_id_1'], row['interface_chain_id_2'])))
        if identity in seen: continue
        seen.add(identity); cid = config_hash(identity)[:16]
        rows.append({'complex_id': cid, 'pdb_id': row['pdb_id'], 'manifest_index': i,
            'partner_A_chains': [row['interface_chain_id_1']], 'partner_B_chains': [row['interface_chain_id_2']],
            'in_original_smoke': cid in old})
    wanted = {r['pdb_id'] + '.cif' for r in rows}
    sources = root/'sources'; sources.mkdir()
    found = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or name not in wanted: continue
            if name in found: raise ValueError(f'ambiguous archive basename {name}')
            path = sources/name; path.write_bytes(tar.extractfile(member).read()); found[name] = digest(path)
    for row in rows:
        row['source_sha256'] = found.get(row['pdb_id']+'.cif')
    result = {'manifest': str(manifest), 'manifest_sha256': digest(manifest), 'archive': str(archive),
        'archive_sha256': digest(archive), 'candidates': rows, 'unique_pairs': len(rows),
        'unique_assemblies': len(wanted), 'missing_sources': sorted(wanted-found.keys()),
        'original_smoke': sum(r['in_original_smoke'] for r in rows),
        'scope': 'All unique protein-protein pairs in local FoldBench manifest, not PDB-wide or SAbDab',
        'job': os.environ['SLURM_JOB_ID']}
    atomic_json(root/'index.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'candidates'}, indent=2))


def install_mmseqs(root):
    root = Path(root); target = root/'tools'; target.mkdir(parents=True, exist_ok=True)
    receipt = target/'mmseqs.json'
    if receipt.exists(): return
    with urllib.request.urlopen('https://api.github.com/repos/soedinglab/MMseqs2/releases/latest', timeout=60) as response:
        release = json.load(response)
    asset = next(a for a in release['assets'] if a['name'].endswith('linux-sse41.tar.gz'))
    with urllib.request.urlopen(asset['browser_download_url'], timeout=120) as response:
        raw = response.read(150*1024*1024+1)
    if len(raw)>150*1024*1024: raise ValueError('archive exceeds size limit')
    archive = target/asset['name']; archive.write_bytes(raw)
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        for member in tar:
            rel = Path(member.name)
            if not member.isfile() or rel.is_absolute() or '..' in rel.parts: continue
            dest = target/rel; dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(member).read()); dest.chmod(member.mode & 0o755)
    binary = target/'mmseqs/bin/mmseqs'
    version = subprocess.check_output([str(binary), 'version'], text=True).strip()
    atomic_json(receipt, {'version':version, 'release':release['tag_name'], 'url':asset['browser_download_url'],
        'sha256':digest(archive), 'binary':str(binary), 'binary_sha256':digest(binary)})
    print(version)


def chain_metadata(cif, atoms, selected):
    asym = cif.block.get('struct_asym'); poly = cif.block.get('entity_poly'); entity = cif.block.get('entity')
    chain_entity = {} if asym is None else dict(zip(asym['id'].as_array(str), asym['entity_id'].as_array(str)))
    sequences = {}; descriptions = {}
    if poly is not None and 'pdbx_seq_one_letter_code_can' in poly:
        sequences = {str(e): ''.join(str(s).split()) for e,s in zip(poly['entity_id'].as_array(str), poly['pdbx_seq_one_letter_code_can'].as_array(str))}
    if entity is not None and 'pdbx_description' in entity:
        descriptions = dict(zip(entity['id'].as_array(str), entity['pdbx_description'].as_array(str)))
    result = []
    for chain in selected:
        residues = atoms[atoms.chain_id == chain]
        starts = struc.get_residue_starts(residues)
        observed = ''.join(AA.get(str(residues.res_name[i]), 'X') for i in starts)
        sequence = sequences.get(chain_entity.get(chain), '')
        origin = 'entity_poly_canonical'
        if not sequence or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in sequence):
            sequence = observed; origin = 'observed_residues'
        result.append({'chain':chain, 'sequence':sequence, 'sequence_source':origin,
            'observed_residues':len(starts), 'description':str(descriptions.get(chain_entity.get(chain), ''))})
    return result


def filtered_cif(source, keep):
    filtered = pdbx.CIFFile.read(source); cat = filtered.block['atom_site']; new = pdbx.CIFCategory()
    for name in cat: new[name] = pdbx.CIFColumn(cat[name].as_array(str)[keep])
    filtered.block['atom_site'] = new
    return filtered


def inspect_candidate(root, row):
    source = root/'sources'/f"{row['pdb_id']}.cif"
    if not source.exists(): raise FileNotFoundError(source)
    cif = pdbx.CIFFile.read(source)
    atoms = pdbx.get_structure(cif, model=1, altloc='occupancy', use_author_fields=False, include_bonds=True)
    selected = row['partner_A_chains'] + row['partner_B_chains']
    chains = chain_metadata(cif, atoms, selected)
    components = component_inventory(atoms, cif, selected)
    cat = cif.block['atom_site']; raw_chains = cat['label_asym_id'].as_array(str); raw_names = cat['label_comp_id'].as_array(str)
    ccd = cif.block.get('chem_comp'); charges = {}
    if ccd is not None and 'pdbx_formal_charge' in ccd:
        for name, charge in zip(ccd['id'].as_array(str), ccd['pdbx_formal_charge'].as_array(str)):
            try: charges[str(name)] = int(charge)
            except ValueError: pass
    protected = set(); connections = cif.block.get('struct_conn')
    if connections is not None:
        for i, kind in enumerate(connections['conn_type_id'].as_array(str)):
            if str(kind).lower().startswith(('covale','metalc')):
                for p in ('ptnr1', 'ptnr2'):
                    c, n = f'{p}_label_asym_id', f'{p}_label_comp_id'
                    if c in connections and n in connections:
                        protected.add((str(connections[c].as_array(str)[i]), str(connections[n].as_array(str)[i])))
    trees = {}
    for chain in selected:
        coords = atoms.coord[(atoms.chain_id == chain) & ~np.isin(np.char.upper(atoms.element), ['H','D'])]
        if len(coords): trees[chain] = cKDTree(coords)
    removals = {mode:np.zeros(len(raw_chains),bool) for mode in ('neutral_additives','all_additives')}
    for comp in components:
        coords = atoms.coord[comp['start']:comp['end']]
        distances = {c:float(t.query(coords)[0].min()) for c,t in trees.items()}
        connected = (comp['chain'],comp['name']) in protected
        clash_or_bond = comp['min_pair_distance'] is not None and comp['min_pair_distance'] < 1.9
        neutral = comp['name'] in NEUTRAL_ADDITIVES and charges.get(comp['name'],0) == 0
        buffer = comp['name'] in BUFFER_ADDITIVES
        candidate = comp['code']=='ligand' and not connected and not clash_or_bond
        comp.update(partner_distances=distances, bridges_partners_4A=len(distances)==2 and max(distances.values())<=4,
            formal_charge=charges.get(comp['name']), declared_connection=connected, bond_or_clash_proximity=clash_or_bond,
            neutral_additive_candidate=candidate and neutral, buffer_additive_candidate=candidate and buffer)
        mask = (raw_chains == comp['chain']) & (raw_names == comp['name'])
        if candidate and neutral: removals['neutral_additives'] |= mask
        if candidate and (neutral or buffer): removals['all_additives'] |= mask
    masks = {'strict':np.ones(len(raw_chains),bool), 'neutral_additives':~removals['neutral_additives'],
        'all_additives':~removals['all_additives'], 'pair_additives':~removals['all_additives'],
        'pair_only_upper_bound':np.isin(raw_chains,selected)}
    results = {}; cache = {}
    for mode in SCENARIOS:
        relaxed = mode.startswith('pair_'); keep = masks[mode]
        key = (keep.tobytes(), relaxed)
        if key in cache:
            results[mode] = dict(cache[key]); continue
        work = root/'work'/row['complex_id']/mode; work.mkdir(parents=True, exist_ok=True)
        try:
            structure, sites = prepare_pair(filtered_cif(source,keep), row, work,
                str(Path(os.environ['PKABENCH_RUNTIME'])/'envs/pypka/bin/pdb2pqr30'), audit_allow_subcomplex=relaxed)
            result = {'status':'accepted', 'n_residues':structure['n_residues'], 'n_sites':len(sites),
                'homomeric':structure['homomeric'], 'annotation':json.loads(structure['provenance'])['annotation']}
        except Rejection as exc:
            result = {'status':'rejected', 'code':exc.code, 'detail':str(exc)}
        except Exception as exc:
            import traceback
            result = {'status':'pipeline_error','detail':str(exc),'traceback':traceback.format_exc()}
        cache[key] = result; results[mode] = result
    descriptions = ' '.join(c['description'] for c in chains).lower()
    method = cif.block.get('exptl')
    return {**row, 'chains':chains, 'components':components, 'scenarios':results,
        'pair_residues':sum(c['observed_residues'] for c in chains),
        'antibody_description_hint':any(word in descriptions for word in ('antibody','immunoglobulin','nanobody','heavy chain','light chain','scfv')),
        'method':None if method is None else method['method'].as_array(str).tolist(),
        'implementation_sha256':digest(Path(__file__)), 'job':os.environ['SLURM_JOB_ID'], 'node':os.environ['SLURMD_NODENAME']}


def scan_dataset(root, shard, shards):
    root = Path(root); index = json.loads((root/'index.json').read_text()); results = root/'rows'; results.mkdir(exist_ok=True)
    assigned = index['candidates'][shard::shards]
    for i,row in enumerate(assigned):
        target = results/f"{row['complex_id']}.json"
        if target.exists(): continue
        try: result = inspect_candidate(root,row)
        except Exception as exc:
            import traceback
            result = {**row,'pipeline_error':str(exc),'traceback':traceback.format_exc()}
        atomic_json(target,result)
        atomic_json(root/'progress'/f'{shard}.json', {'completed':i+1,'assigned':len(assigned),'job':os.environ['SLURM_JOB_ID']})
    atomic_json(root/'shards'/f'{shard}.json', {'shard':shard,'shards':shards,'assigned':len(assigned),'job':os.environ['SLURM_JOB_ID']})


class UnionFind:
    def __init__(self, keys): self.parent = {k:k for k in keys}
    def find(self, key):
        if self.parent[key] != key: self.parent[key] = self.find(self.parent[key])
        return self.parent[key]
    def join(self, a, b):
        a,b = self.find(a),self.find(b)
        if a != b: self.parent[max(a,b)] = min(a,b)


def load_rows(root):
    index = json.loads((root/'index.json').read_text())
    paths = [root/'rows'/f"{r['complex_id']}.json" for r in index['candidates']]
    missing = [str(p) for p in paths if not p.exists()]
    if missing: raise ValueError(f'{len(missing)} missing candidate rows')
    return [json.loads(p.read_text()) for p in paths]


def diversity_summary(rows, pairs, links, cutoff):
    seqs = {s for p in pairs.values() for s in p}; uf = UnionFind(seqs)
    for a,b,identity in links:
        if identity >= cutoff: uf.join(a,b)
    clusters = {s:uf.find(s) for s in seqs}
    graph = UnionFind(set(clusters.values()))
    for a,b in pairs.values(): graph.join(clusters[a],clusters[b])
    components = {cid:graph.find(clusters[p[0]]) for cid,p in pairs.items()}
    universe_counts = Counter(components.values())
    stats = {}
    for mode in SCENARIOS:
        retained = [r for r in rows if r.get('scenarios',{}).get(mode,{}).get('status')=='accepted']
        ids = [r['complex_id'] for r in retained if r['complex_id'] in pairs]
        counts = Counter(components[cid] for cid in ids)
        cluster_pairs = {tuple(sorted(clusters[s] for s in pairs[cid])) for cid in ids}
        stats[mode] = {'retained_pairs':len(retained), 'with_sequences':len(ids),
            'independent_components':len(counts), 'unique_cluster_pairs':len(cluster_pairs),
            'largest_component_pairs':max(counts.values(),default=0),
            'largest_component_fraction':max(counts.values(),default=0)/len(ids) if ids else None,
            'component_sizes':sorted(counts.values(),reverse=True),
            'components':dict(sorted(counts.items()))}
    return {'identity':cutoff, 'chain_clusters':len(set(clusters.values())),
        'universe_components':len(universe_counts), 'universe_largest_component':max(universe_counts.values(),default=0),
        'scenarios':stats}


def sequence_audit(root):
    root = Path(root); rows = load_rows(root); work = root/'sequence'; work.mkdir(exist_ok=True)
    seqs = {}; pairs = {}; origins = Counter()
    for row in rows:
        chains = row.get('chains',[])
        if len(chains)!=2 or any(not c['sequence'] for c in chains): continue
        ids = []
        for chain in chains:
            seq = chain['sequence']; key = 's'+config_hash(seq)[:20]
            seqs[key] = seq; ids.append(key); origins[chain['sequence_source']]+=1
        pairs[row['complex_id']] = ids
    fasta = work/'chains.fasta'
    fasta.write_text(''.join(f'>{key}\n{seq}\n' for key,seq in sorted(seqs.items())))
    tool = json.loads((root/'tools/mmseqs.json').read_text()); hits = work/'hits.tsv'
    command = [tool['binary'], 'easy-search', str(fasta), str(fasta), str(hits), str(Path(os.environ['TMPDIR'])/'mmseqs'),
        '--threads','1','--min-seq-id','0.3','-c','0.8','--cov-mode','0','--alignment-mode','3',
        '--max-seqs','10000','-s','7.5','--split-memory-limit','2G',
        '--format-output','query,target,fident,qcov,tcov']
    with (work/'mmseqs.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
    links = []
    for line in hits.read_text().splitlines():
        a,b,identity,qcov,tcov = line.split('\t')
        if min(float(qcov),float(tcov))>=.8: links.append((a,b,float(identity)))
    result = {'tool':tool,'command':command,'sequence_sources':dict(origins),'unique_sequences':len(seqs),
        'pairs_with_sequences':len(pairs),'total_pairs':len(rows),'fasta_sha256':digest(fasta),'hits_sha256':digest(hits),
        'method':'MMseqs2 all-v-all alignments; identity edges use >=80% coverage of both sequences. Chain clusters and complex split units are connected components.',
        'limitations':'Conservative feasibility audit, not a frozen split. Uses all local candidate pairs before prep, including rejected bridges. No antibody CDR replacement; observed-sequence fallbacks are recorded. Search is heuristic.',
        'thresholds':[diversity_summary(rows,pairs,links,c) for c in (.3,.5,.9)],
        'job':os.environ['SLURM_JOB_ID'],'node':os.environ['SLURMD_NODENAME']}
    atomic_json(work/'diversity.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='thresholds'},indent=2))


def summarize(root, shards):
    root = Path(root); rows = load_rows(root)
    absent = [i for i in range(shards) if not (root/'shards'/f'{i}.json').exists()]
    if absent: raise ValueError(f'unfinished shards: {absent}')
    sequence = json.loads((root/'sequence/diversity.json').read_text())
    histograms = {}; strata = {}; sample = {}
    for mode in SCENARIOS:
        counts = Counter(); group_counts = defaultdict(Counter)
        for row in rows:
            result = row.get('scenarios',{}).get(mode,{'status':'pipeline_error'})
            status = result.get('code', result['status']); counts[status]+=1
            n = row.get('pair_residues',0); size = '<200' if n<200 else '200-499' if n<500 else '500-999' if n<1000 else '>=1000'
            group_counts[f'size:{size}'][status]+=1
            group_counts[f"antibody_hint:{row.get('antibody_description_hint',False)}"][status]+=1
            group_counts[f"original_smoke:{row['in_original_smoke']}"][status]+=1
        histograms[mode] = dict(counts); strata[mode] = {k:dict(v) for k,v in group_counts.items()}
        sample[mode] = [r['complex_id'] for r in rows if r.get('scenarios',{}).get(mode,{}).get('status')=='accepted']
    compounds = defaultdict(lambda:{'instances':0,'pairs':set(),'near_4A_pairs':set(),'bridging_pairs':set(),'connected_pairs':set()})
    for row in rows:
        for c in row.get('components',[]):
            stat = compounds[c['name']]; stat['instances']+=1; stat['pairs'].add(row['complex_id'])
            if c.get('min_pair_distance') is not None and c['min_pair_distance']<=4: stat['near_4A_pairs'].add(row['complex_id'])
            if c.get('bridges_partners_4A'): stat['bridging_pairs'].add(row['complex_id'])
            if c.get('declared_connection'): stat['connected_pairs'].add(row['complex_id'])
    compounds = {k:{field:len(v) if isinstance(v,set) else v for field,v in stat.items()} for k,stat in compounds.items()}
    rescued = {mode:[{'complex_id':r['complex_id'],'pdb_id':r['pdb_id'],'components':r['components']}
        for r in rows if r['complex_id'] in set(sample[mode])-set(sample['strict'])] for mode in SCENARIOS[1:]}
    result = {'total_pairs':len(rows),'remaining_pairs':sum(not r['in_original_smoke'] for r in rows),
        'histograms':histograms,'strata':strata,'component_inventory':compounds,'accepted_ids':sample,'rescued':rescued,
        'diversity':sequence,'pipeline_errors':[r['complex_id'] for r in rows if 'pipeline_error' in r or any(v['status']=='pipeline_error' for v in r.get('scenarios',{}).values())],
        'scenarios':{'strict':'Unchanged current binary-assembly and chemistry/geometry/completion policy',
            'neutral_additives':'Remove listed neutral additive candidates unless declared covalent/metal-connected or <1.9 A from the selected pair; retain binary rule',
            'all_additives':'Also remove listed buffer/salt candidates; retain binary rule; diagnostic, not validated safe',
            'pair_additives':'Same additive removal while allowing selected pair from a larger assembly; other components still checked',
            'pair_only_upper_bound':'Keep selected chains only, then all remaining prep checks; unsafe upper-bound scenario, not production'},
        'neutral_additives':sorted(NEUTRAL_ADDITIVES),'buffer_additives':sorted(BUFFER_ADDITIVES),
        'limits':'Additive identity/proximity is not evidence of dispensability. No teacher predictions, ligand parameterisation, chemical remapping, CDR annotation or production split performed.',
        'job':os.environ['SLURM_JOB_ID'],'node':os.environ['SLURMD_NODENAME']}
    atomic_json(root/'report.json',result)
    compact = {'total_pairs':len(rows), 'remaining_pairs':result['remaining_pairs'], 'pipeline_errors':result['pipeline_errors'],
        'retention':[], 'size_bias':[], 'rescued_components':{},
        'scope':f'{len(rows)} general protein-protein FoldBench interfaces; separate antibody-antigen/peptide/ligand tasks and PDB-wide universe were not screened'}
    for mode in SCENARIOS:
        accepted = histograms[mode].get('accepted',0)
        table_row = {'scenario':mode,'accepted_pairs':accepted,'fraction':accepted/len(rows),
            'accepted_outside_smoke':strata[mode].get('original_smoke:False',{}).get('accepted',0)}
        for threshold in sequence['thresholds']:
            table_row[f"components_{int(threshold['identity']*100)}pct"] = threshold['scenarios'][mode]['independent_components']
        compact['retention'].append(table_row)
        for size, counts in strata[mode].items():
            if size.startswith('size:'):
                compact['size_bias'].append({'scenario':mode,'size':size[5:],'total':sum(counts.values()),
                    'accepted':counts.get('accepted',0),'fraction':counts.get('accepted',0)/sum(counts.values())})
        if mode in rescued:
            compact['rescued_components'][mode] = [{'pdb_id':r['pdb_id'],'components':sorted({c['name'] for c in r['components']}),
                'bridging_additives':sorted({c['name'] for c in r['components'] if c.get('bridges_partners_4A') and (c.get('neutral_additive_candidate') or c.get('buffer_additive_candidate'))})} for r in rescued[mode]]
    atomic_json(root/'summary.json',compact)
    for name, data in (('retention',compact['retention']),('size_bias',compact['size_bias'])):
        with (root/f'{name}.csv').open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    print(json.dumps({'total_pairs':len(rows),'histograms':histograms,'pipeline_errors':result['pipeline_errors'],
        'diversity':[{'identity':d['identity'],'scenarios':{m:{k:v for k,v in s.items() if k not in ('components','component_sizes')} for m,s in d['scenarios'].items()}} for d in sequence['thresholds']]},indent=2))


def defect_audit(root):
    """Separate assembly multiplicity from unobserved residues/atoms; no label imputation."""
    from jaxpropka.geometry import _template
    root = Path(root); rows = load_rows(root); results = []; chain_hist = Counter(); multi_size = Counter()
    for row in rows:
        strict = row['scenarios']['strict']; source = root/'sources'/f"{row['pdb_id']}.cif"
        cif = pdbx.CIFFile.read(source)
        atoms = pdbx.get_structure(cif, model=1, altloc='occupancy', use_author_fields=False)
        selected = [c['chain'] for c in row['chains']]
        all_poly = atoms[np.isin(atoms.res_name,list(CANONICAL))]
        if strict.get('code') == 'multi_partner':
            detail = json.loads(strict['detail']); chain_hist[len(detail['protein_chains'])]+=1
            total = len(struc.get_residue_starts(all_poly))
            multi_size['<=1500' if total<=1500 else '>1500']+=1
        record = {'complex_id':row['complex_id'],'pdb_id':row['pdb_id'],'strict_status':strict.get('code',strict['status']),
            'missing_atom_residues':[],'unobserved_runs':[]}
        for chain in row['chains']:
            chain_id = chain['chain']; residues = atoms[atoms.chain_id==chain_id]
            partner = atoms[(atoms.chain_id==next(c for c in selected if c!=chain_id)) & ~np.isin(np.char.upper(atoms.element),['H','D'])]
            tree = cKDTree(partner.coord)
            starts = struc.get_residue_starts(residues,add_exclusive_stop=True); observed = {}; distances = {}
            for s,e in zip(starts[:-1],starts[1:]):
                residue = residues[s:e]; number = int(residue.res_id[0]); name = str(residue.res_name[0])
                if number<=0: continue
                observed[number] = name; distance = float(tree.query(residue.coord)[0].min()); distances[number]=distance
                if name not in CANONICAL: continue
                template,_,_ = _template(name); missing = sorted(set(template)-{'OXT'}-set(map(str,residue.atom_name)))
                if missing: record['missing_atom_residues'].append({'chain':chain_id,'label_seq_id':number,'resname':name,
                    'missing':missing,'missing_backbone':bool(set(missing)&{'N','CA','C','O'}),
                    'observed_min_partner_distance':distance,'observed_interface_zone':distance<=10})
            if not observed: continue
            absent = sorted(set(range(1,len(chain['sequence'])+1))-observed.keys())
            runs = []
            for number in absent:
                if not runs or number != runs[-1][-1]+1: runs.append([number])
                else: runs[-1].append(number)
            for run in runs:
                kind = 'N_terminal' if run[-1]<min(observed) else 'C_terminal' if run[0]>max(observed) else 'internal'
                flanks = [i for i in (run[0]-1,run[-1]+1) if i in observed]
                record['unobserved_runs'].append({'chain':chain_id,'start':run[0],'end':run[-1],'length':len(run),
                    'kind':kind,'flank_min_partner_distance':min((distances[i] for i in flanks),default=None),
                    'flank_in_interface_zone':any(distances[i]<=10 for i in flanks)})
        results.append(record)
    strict_missing = [r for r in results if r['strict_status'] in ('missing_backbone','interface_gap','interface_missing_sidechain','missing_titratable_sidechain')]
    report = {'multi_partner_chain_count_histogram':dict(sorted(chain_hist.items())), 'multi_partner_assembly_residue_size':dict(multi_size),
        'strict_first_rejection_missing_structure':strict_missing,
        'accepted_with_unobserved_terminal_residues':sum(r['strict_status']=='accepted' and any(g['kind']!='internal' for g in r['unobserved_runs']) for r in results),
        'accepted_with_distal_missing_atom_residues':sum(r['strict_status']=='accepted' and any(not g['observed_interface_zone'] for g in r['missing_atom_residues']) for r in results),
        'limits':'Unobserved positions are identified by deposited canonical sequence versus label sequence IDs. A missing segment has no observed location; flank distances do not prove the entire missing segment is distal or buried. No missing-site targets are imputed.',
        'candidates':results,'job':os.environ['SLURM_JOB_ID'],'node':os.environ['SLURMD_NODENAME']}
    atomic_json(root/'defects.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='candidates'},indent=2))


def main(args):
    require_compute()
    if args.stage == 'index': index_dataset(args.manifest, args.archive, args.root, args.smoke)
    elif args.stage == 'install-mmseqs': install_mmseqs(args.root)
    elif args.stage == 'scan': scan_dataset(args.root, args.shard, args.shards)
    elif args.stage == 'sequence': sequence_audit(args.root)
    elif args.stage == 'summarize': summarize(args.root, args.shards)
    elif args.stage == 'defects': defect_audit(args.root)
