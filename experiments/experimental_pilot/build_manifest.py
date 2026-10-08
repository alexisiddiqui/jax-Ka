"""Build a provenance-preserving experimental candidate manifest on a compute node.

This is a curation preflight, not a training-data exporter. Structure mapping and
experimental construct matching are separate admission gates.
"""
import csv
import json
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path

from pkabench.runtime import require_compute, atomic_json, digest

require_compute()
out = Path(os.environ['PKABENCH_RUNTIME']) / 'experimental/pilot-v1'
src = out / 'sources'
tables = {x['id']: x for x in json.loads((out / 'fcrn_tables.json').read_text())}
observations = []

def add(identifier, observable, value, error, ph, construct, source, locator, **extra):
    observations.append(dict(
        observation_id=identifier, observable=observable, value=value,
        uncertainty=error, unit='M', pH=ph, construct=construct,
        source_url=source, source_locator=locator,
        measurement_verified=True, training_eligible=False,
        mask_status='not_applied_construct_match_pending',
        reservation_group='igg_connected', split='evaluation_candidate', **extra))

def pair(cell):
    # Table 3's first avidity error uses '+'; caption explicitly defines ± fit error.
    vals = re.findall(r'\d+(?:\.\d+)?', cell)
    assert len(vals) == 2, cell
    return tuple(map(float, vals))

fcrn_source = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11164218/'
fcrn_common = dict(
    source_sha256=digest(src / 'PMC11164218-epmc.raw'),
    uncertainty_type='global_fit_error', temperature_K=298.15,
    conditions='10 mM phosphate, 140 mM NaCl, 50 uM EDTA, 50 uM EGTA, 0.05% Tween20',
    assay='switchSENSE; immobilized hscFcRn; medium ligand density',
    structure_candidate='4N0U', structure_status='proxy_only',
    hold_reason='Full mAb1 assay; Fab sequence unresolved; 4N0U Fc fragment and receptor fusion do not match assay construct')
kinetic_checks = []
for row in tables['t0003']['rows'][2:]:
    ph = float(row[0])
    kon = pair(row[1])[0] * 1e6
    koff = pair(row[2])[0] * 1e-2
    kd, err = pair(row[3])
    discrepancy = abs(koff / kon / (kd * 1e-9) - 1)
    assert discrepancy < 0.01, (ph, discrepancy)
    kinetic_checks.append(dict(pH=ph, relative_Kd_rounding_error=discrepancy))
    for observable, cell in [('affinity_Kd', row[3]), ('avidity_Kd', row[5])]:
        if cell == 'NA':
            continue
        value, error = pair(cell)
        add(f'fcrn2024-YTE-{ph:.1f}-{observable}', observable,
            value*1e-9, error*1e-9, ph, 'mAb1 full IgG1 YTE + hscFcRn',
            fcrn_source, f'Table 3, pH {ph:.1f}, {observable}',
            raw_cell=cell, binary_linkage_eligible_observable=(observable == 'affinity_Kd'),
            **fcrn_common)

# The YTE row of Table 2 repeats Table 3 pH 6.0; retain only the WT row.
wt = tables['t0002']['rows'][2]
assert wt[0] == 'hIgG1 Fc WT'
for observable, cell in [('affinity_Kd', wt[3]), ('avidity_Kd', wt[5])]:
    value, error = pair(cell)
    add(f'fcrn2024-WT-6.0-{observable}', observable, value*1e-9, error*1e-9,
        6.0, 'mAb1 full IgG1 WT + hscFcRn', fcrn_source,
        f'Table 2, WT, {observable}', raw_cell=cell,
        binary_linkage_eligible_observable=(observable == 'affinity_Kd'), **fcrn_common)

for construct, value, error in [('GB01', 4.9e-7, .2e-7), ('GB09', 2.9e-7, .6e-7), ('GB0919', 4.3e-8, .8e-8)]:
    add(f'proteinG2009-{construct}-7.4', 'affinity_Kd', value, error, 7.4,
        f'{construct} + trastuzumab Fc', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC2673305/',
        'Results: Kinetic Analysis of Histidine-introduced Mutants; Figure 5a paragraph',
        source_sha256=digest(src / 'PMC2673305-ncbi.raw'),
        uncertainty_type='reported_plus_minus_definition_unresolved', temperature_K=None,
        conditions='10 mM HEPES, 150 mM NaCl, 0.05% Tween20',
        assay='SPR; immobilized trastuzumab Fc; 1:1 fit',
        structure_candidate='1FCC', structure_status='proxy_only',
        hold_reason='1FCC contains protein G C2 and MO61 Fc; assay uses B1-derived variants and trastuzumab Fc; assay temperature unresolved',
        binary_linkage_eligible_observable=True)

raw = json.loads((src / 'PKAD-R-250211.json').read_text())
inventory = []
for row in raw:
    label = str(row['Expt_pKa']).strip()
    exact = re.fullmatch(r'-?\d+(?:\.\d+)?', label)
    kind = 'point' if exact else ('censored' if '<' in label or '>' in label else 'approximate_or_other')
    inventory.append(dict(
        observation_id='pkadr-' + str(row['Index']), observable='residue_pKa',
        value=float(label) if exact else None, raw_label=label, label_kind=kind,
        protein=row['Protein_Name'], pdb_id=row['PDB'], chain=row['Chain'],
        residue_id=str(row['ResID_in_PDB']), residue_name=row['ResName'],
        primary_reference=row['Reference'], source_row=row,
        source_sha256=digest(src / 'PKAD-R-250211.json'),
        source_url='https://compbio.clemson.edu/pkad-r/',
        measurement_verified=False, training_eligible=False,
        hold_reason='Curated secondary record; primary conditions, construct, masks and leakage audit pending'))
selected = [r for r in inventory if 'barnase' in r['protein'].lower() or 'protein g,' in r['protein'].lower()]

# Author-chain/number/residue identity check only. This does not assert exact
# experimental construct agreement or complete side chains/absence of gaps.
from biotite.structure.io.pdbx import CIFFile
mapping = []
for pdb in sorted({r['pdb_id'] for r in selected}):
    assert re.fullmatch(r'[A-Za-z0-9]{4}', pdb), pdb
    path = src / f'{pdb}-cif.raw'
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(f'https://files.rcsb.org/download/{pdb}.cif', timeout=60).read())
    block = CIFFile.read(path).block
    atoms = block['atom_site']
    columns = {k: atoms[k].as_array(str) for k in
               ('auth_asym_id', 'auth_seq_id', 'auth_comp_id', 'label_atom_id', 'pdbx_PDB_model_num', 'pdbx_PDB_ins_code')}
    residues = {}
    for i in range(atoms.row_count):
        if columns['pdbx_PDB_model_num'][i] != '1':
            continue
        key = (columns['auth_asym_id'][i], columns['auth_seq_id'][i])
        residues.setdefault(key, []).append((columns['auth_comp_id'][i], columns['label_atom_id'][i], columns['pdbx_PDB_ins_code'][i]))
    for record in [r for r in selected if r['pdb_id'] == pdb]:
        found = residues.get((record['chain'], record['residue_id']), [])
        names = sorted({a[0] for a in found})
        insertions = sorted({a[2] for a in found})
        terminal = record['residue_name'] in ('C-term', 'N-term')
        good = None if terminal else names == [record['residue_name']] and all(x in ('.', '?') for x in insertions)
        mapping.append(dict(observation_id=record['observation_id'], pdb_id=pdb,
            source_url=f'https://files.rcsb.org/download/{pdb}.cif', source_sha256=digest(path),
            observed_residue_names=names, insertion_codes=insertions,
            residue_identity_matches=good, terminal_site=terminal,
            backbone_N_CA_C_present=all(a in {v[1] for v in found} for a in ('N','CA','C')),
            quality_mask_status='not_yet_applied', construct_match_status='not_verified'))
        record.update(reservation_group='barnase_barstar' if 'barnase' in record['protein'].lower() else 'igg_connected',
                      split='evaluation_candidate', residue_identity_matches=good)

assert len(observations) == 22
assert len({x['observation_id'] for x in observations + inventory}) == len(observations) + len(inventory)
assert all(not x['training_eligible'] for x in observations + inventory)
assert all(x['value'] is None for x in inventory if x['label_kind'] != 'point')
assert sum(x['observable'] == 'affinity_Kd' for x in observations) == 13
atomic_json(out / 'verified_measurements.json', observations)
atomic_json(out / 'pkadr_candidate_inventory.json', inventory)
atomic_json(out / 'lead_residue_candidates.json', selected)
atomic_json(out / 'residue_mapping_preflight.json', mapping)
atomic_json(out / 'validation.json', dict(kinetic_checks=kinetic_checks, unique_ids=True,
    censored_values_not_point_labels=True, training_gate_closed=True,
    duplicate_rows_omitted=['Table 2 YTE affinity and avidity repeat Table 3 pH 6.0'],
    missing_rows_not_zero=['Table 3 pH 7.4 avidity NA']))
atomic_json(out / 'reservation_proposal.json', dict(status='proposal; existing synthetic split unchanged',
    blocks={'barnase_barstar': ['barnase', 'barstar'], 'igg_connected': ['protein G', 'Fc', 'FcRn', 'IgG']},
    policy='Reserve all current lead families as evaluation candidates; no row-level random split; run sequence/component leakage audit before any experimental training'))
counts = dict(verified_affinity=13, avidity_context_only=9,
    pkadr_inventory=len(inventory), pkadr_label_kinds=dict(Counter(x['label_kind'] for x in inventory)),
    lead_residue_candidates=len(selected), lead_label_kinds=dict(Counter(x['label_kind'] for x in selected)),
    lead_families=dict(Counter(x['protein'] for x in selected)),
    residue_identity_matches=sum(x['residue_identity_matches'] is True for x in mapping),
    terminal_mapping_pending=sum(x['terminal_site'] for x in mapping),
    training_ready=0, measured_paired_residue_shifts=0)
atomic_json(out / 'counts.json', counts)
with (out / 'verified_measurements.csv').open('w') as f:
    fields = ['observation_id','observable','value','uncertainty','unit','pH','construct','uncertainty_type','training_eligible','hold_reason','source_url','source_locator']
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    writer.writeheader(); writer.writerows(observations)
lines = ['# Experimental pilot v1 — curation preflight', '',
    'Verified numeric measurements exist; none is admitted to training yet. This release distinguishes measurement evidence from structure eligibility.', '',
    '| Branch | Records | Admission status |', '|---|---:|---|',
    '| FcRn–IgG affinity | 10 | Verified; construct mapping held |',
    '| Protein G–Fc affinity | 3 | Verified; construct/temperature held |',
    '| FcRn avidity | 9 | Context only; not binary affinity labels |',
    f'| PKAD-R candidate inventory | {len(inventory)} | Secondary records; not primary-verified |',
    f'| Barnase/protein G residue-pKa candidates (subset) | {len(selected)} | Primary conditions/construct/masks pending |',
    '| Measured bound-minus-free residue-pKa pairs | 0 | Not established |',
    '| Training-ready observations | 0 | Admission gates remain closed |', '',
    '## FcRn YTE affinity series', '', '| pH | Kd (nM) | Global-fit error (nM) |', '|---:|---:|---:|']
for r in observations:
    if r['observation_id'].startswith('fcrn2024-YTE') and r['observable'] == 'affinity_Kd':
        lines.append(f"| {r['pH']:.1f} | {r['value']*1e9:g} | {r['uncertainty']*1e9:g} |")
lines += ['', 'Errors are fitting errors, not replicate SDs or confidence intervals. Table 2 YTE duplicates are omitted; missing avidity is not zero.', '',
    '[FcRn primary source, Tables 2–3 and methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11164218/). The assay uses full mAb1 IgG and a single-chain FcRn fusion. 4N0U is a structural proxy, not an exact construct match; Fab contributions cannot be assumed absent.', '',
    '[Protein G primary source, Figure 5a results](https://pmc.ncbi.nlm.nih.gov/articles/PMC2673305/): GB01 490 ± 20 nM, GB09 290 ± 60 nM, GB0919 43 ± 8 nM at pH 7.4. The numerical pH series in Figure 5b has not been digitized. 1FCC has C2/MO61, while the assay uses B1 variants/trastuzumab Fc.', '',
    '[Barnase primary abstract](https://pubmed.ncbi.nlm.nih.gov/8494892/) remains a lead: an apparent binding ionization is not a measured bound/free residue-pKa pair. Full conditions and the 1BRS barstar mutations need reconciliation.', '',
    '## Residue-pKa candidates', '', '| Protein | Candidate rows |', '|---|---:|']
for name, count in counts['lead_families'].items():
    lines.append(f'| {name} | {count} |')
lines += ['', f"Author-chain/residue-name mapping passes for all {counts['residue_identity_matches']} nonterminal lead records. One additional record is the barnase C-terminus at Arg110; terminal chemistry requires separate validation, rather than comparing the label C-term to the residue name ARG. This checks numbering and identity only, not experimental construct equivalence or side-chain completeness. Of the 42 records, 35 are point values, five are censored and two are approximate.", '',
    '[PKAD-R source](https://compbio.clemson.edu/pkad-r/). Original uncertainty, temperature, salt, method, warnings, alternatives and primary references remain in source_row. Censored and approximate values are retained verbatim and are not exported as ordinary point labels. Barnase salt ranges and isotope/state differences need primary-source resolution.', '',
    '## Admission and split rules', '',
    'Current leads form two connected candidate evaluation blocks: barnase/barstar and IgG-linked protein G/FcRn. Shared Fc prevents treating the latter as independent families. The reservation file is a proposal; existing synthetic train/validation/test assignments were not changed. No experimental training has started.', '',
    'Next: verify primary residue-pKa tables/conditions, exact constructs and sequence overlap; apply existing preparation and site masks after mapping; expand to independent protein families. Do not use the held evaluation families for tuning.', '',
    'Existing train/evaluation radii remain buffers 15/20 Å, eligible ligands 15/25 Å, neutral glycans 20/25 Å and exposed uncoordinated metals 25/25 Å. Buried/coordinated metals remain excluded. Use calibrated anchor/length rules for missing residues. These site-level rules do not validate a whole-complex affinity measurement after stripping species.', '',
    '## Validation and provenance', '',
    'All nine affinity Kd values agree with koff/kon to <1% rounding error. IDs are unique, duplicate table entries are omitted, and all training gates are closed. Raw sources, hashes, machine-readable JSON, CSV, mapping checks and counts accompany this report. No pKa calculation or model fitting was performed.']
(out / 'report.md').write_text('\n'.join(lines) + '\n')
atomic_json(out / 'release_manifest.json', dict(schema_version=1, slurm_job_id=os.environ['SLURM_JOB_ID'],
    generator_sha256=digest(Path(__file__)), artifacts={p.name: digest(p) for p in sorted(out.glob('*')) if p.is_file() and p.name != 'release_manifest.json'}))
print(json.dumps(counts, indent=2), flush=True)
