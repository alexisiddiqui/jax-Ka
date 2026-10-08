"""Observed-coordinate provenance and paired supervision masks; never impute labels."""
import numpy as np
import biotite.structure as struc
from biotite.structure.io import pdbx
from .annotate import SITE_ATOMS
from .prep import CANONICAL, read_cif
from jaxpropka.geometry import _template

RADII = (10, 15, 20)
TAIL_RADII = {10:5, 15:8, 20:10}


def classify_exposure(cif, partners, evidence):
    """Attachment exposure is a proxy, never an observation of a missing tail."""
    from scipy.spatial import cKDTree
    atoms=pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
    atoms=atoms[np.isin(atoms.chain_id,partners['A']+partners['B']) & np.isin(atoms.res_name,list(CANONICAL)) & ~np.isin(np.char.upper(atoms.element),['H','D'])]
    bound=np.asarray(struc.sasa(atoms,probe_radius=1.4,point_number=1000),float)
    free=np.zeros(len(atoms)); trees={}
    for p,chains in partners.items():
        mask=np.isin(atoms.chain_id,chains)
        free[mask]=struc.sasa(atoms[mask],probe_radius=1.4,point_number=1000)
        trees[p]=cKDTree(atoms.coord[mask])
    starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
    residues={(str(atoms.chain_id[s]),int(atoms.res_id[s])):(s,e) for s,e in zip(starts[:-1],starts[1:])}
    seqs={r['chain']:r for r in evidence['sequences']}
    incomplete={(r['key'][0],r['label_seq_id']) for r in evidence['atoms'] if r['missing_atoms']}
    uncertain={tuple(d['key']) for d in evidence['defects'] if d['kind']=='alternate_conformation'}
    uncertain_label={(r['key'][0],r['label_seq_id']) for r in evidence['atoms'] if tuple(r['key']) in uncertain}
    for defect in evidence['defects']:
        defect['exposure_class']='unknown'; defect['smaller_tail_radius_candidate']=False
        if defect['kind'] not in ('terminal_gap','internal_gap','missing_atoms'): continue
        chain=defect.get('chain',defect.get('key',[''])[0])
        partner=next(p for p,ch in partners.items() if chain in ch); other='B' if partner=='A' else 'A'
        if defect['kind']=='missing_atoms': positions=[defect['label_seq_id']]
        elif defect['kind']=='internal_gap': positions=[defect['start']-1,defect['end']+1]
        elif defect['start']==1: positions=list(range(defect['end']+1,defect['end']+4))
        else: positions=list(range(defect['start']-3,defect['start']))
        anchors=[]
        for position in positions:
            if (chain,position) not in residues: continue
            s,e=residues[(chain,position)]; residue=atoms[s:e]
            isolated=float(np.nansum(struc.sasa(residue,probe_radius=1.4,point_number=1000)))
            b=float(np.nansum(bound[s:e]))/isolated if isolated else 0.
            f=float(np.nansum(free[s:e]))/isolated if isolated else 0.
            anchors.append({'label_seq_id':position,'bound_exposure_fraction':b,'free_exposure_fraction':f,
                'min_partner_distance':float(trees[other].query(residue.coord)[0].min()),
                'complete':(chain,position) not in incomplete,'alternate_uncertainty':(chain,position) in uncertain_label})
        defect['exposure_evidence']=anchors
        if anchors:
            fractions=[min(a['bound_exposure_fraction'],a['free_exposure_fraction']) for a in anchors]
            defect['exposure_class']='exposed_anchor' if min(fractions)>=.4 else 'buried_anchor' if max(fractions)<=.1 else 'intermediate_anchor'
        if defect['kind']=='terminal_gap':
            sequence=seqs[chain]['sequence'][defect['start']-1:defect['end']]
            defect['missing_sequence']=sequence
            defect['missing_ionisable_residues']=sum(aa in 'DEHKRCY' for aa in sequence)
            defect['smaller_tail_radius_candidate']=bool(defect['length']<=5 and len(anchors)==3
                and seqs[chain]['sequence_source']=='entity_poly_canonical' and all(aa in 'ARNDCQEGHILKMFPSTWYV' for aa in sequence)
                and not defect['missing_ionisable_residues'] and defect['exposure_class']=='exposed_anchor'
                and all(a['complete'] and not a['alternate_uncertainty'] and a['min_partner_distance']>20 for a in anchors))
    evidence['exposure_policy']={'proxy':'observed flank SASA / same isolated-residue SASA, bound and free; not missing-residue burial',
        'candidate':'terminal gap <=5 residues, 3 complete exposed flanks, >20 A from partner, no missing ionisable residue, no flank alternate uncertainty',
        'exposed_threshold':.4,'buried_threshold':.1,'tail_radii':TAIL_RADII,'validated':False,
        'limits':'Absent neighbours can inflate anchor exposure. The contour envelope is retained; shortening the radius does not localise long tails.'}
    return evidence


def inventory(cif, partners):
    from .dataset_audit import chain_metadata
    label = pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=False)
    author = pdbx.get_structure(cif,model=1,altloc='occupancy',use_author_fields=True)
    if len(label)!=len(author) or not np.array_equal(label.coord,author.coord): raise ValueError('ambiguous atom mapping')
    selected = list(partners['A'])+list(partners['B']); rows=[]; defects=[]; boundaries=[]
    sequences=chain_metadata(cif,label,selected)
    for chain in sequences:
        mask=label.chain_id==chain['chain']; atoms=label[mask]; auth=author[mask]
        starts=struc.get_residue_starts(atoms,add_exclusive_stop=True); observed={}
        for s,e in zip(starts[:-1],starts[1:]):
            residue=atoms[s:e]; name=str(residue.res_name[0]); position=int(residue.res_id[0])
            if name not in CANONICAL or position<=0: continue
            key=[chain['chain'],int(auth.res_id[s]),str(auth.ins_code[s]).strip()]
            centre=residue.coord[residue.atom_name=='CA']
            if not len(centre): centre=residue.coord[:1]
            centre=centre[0].astype(float).tolist(); observed[position]=(centre,key)
            template,_,_=_template(name); expected=sorted(set(template)-{'OXT'})
            missing=sorted(set(expected)-set(map(str,residue.atom_name)))
            rows.append({'key':key,'label_seq_id':position,'resname':name,'expected_atoms':expected,
                'input_atom_mask':[a in set(residue.atom_name) for a in expected],'missing_atoms':missing})
            if missing:
                defects.append({'kind':'missing_atoms','key':key,'label_seq_id':position,'missing_atoms':missing,
                    'centres':[centre],'extent':8.0,'envelope':'provisional 8 A side-chain envelope from CA/remaining atom'})
        if not observed: continue
        absent=sorted(set(range(1,len(chain['sequence'])+1))-observed.keys()); runs=[]
        for number in absent:
            if not runs or number!=runs[-1][-1]+1: runs.append([number])
            else: runs[-1].append(number)
        for run in runs:
            flanks=[i for i in (run[0]-1,run[-1]+1) if i in observed]
            kind='internal_gap' if len(flanks)==2 else 'terminal_gap'
            defects.append({'kind':kind,'chain':chain['chain'],'start':run[0],'end':run[-1],
                'length':len(run),'centres':[observed[i][0] for i in flanks],'extent':8.,
                'envelope':'CA contour-length reach plus provisional 8 A heavy-atom extent; missing coordinates are not inferred'})
            for i in flanks: boundaries.append(observed[i][1])
    return {'atoms':rows,'sequences':sequences,'defects':defects,'artificial_terminal_keys':boundaries}


def clearance(points, defect):
    """Lower bound on distance to an uncertain region; smaller means more conservative."""
    points=np.asarray(points,float); centres=np.asarray(defect.get('centres',[]),float)
    if not len(points) or not len(centres): return 0.0
    distances=np.linalg.norm(points[:,None,:]-centres[None,:,:],axis=-1)
    if defect['kind']=='internal_gap' and len(centres)==2:
        n=defect['length']
        # Every possible CA lies in the intersection of two contour-reach balls.
        # The max of distances to those balls is a lower bound to their intersection.
        return float(max(0.,min(np.maximum(np.maximum(distances[:,0]-3.8*k,distances[:,1]-3.8*(n+1-k)),0).min() for k in range(1,n+1))-defect.get('extent',0.)))
    extent=3.8*defect['length']+defect.get('extent',0.) if defect['kind']=='terminal_gap' else defect.get('extent',0.)
    return float(max(0.,distances.min()-extent))


def annotate_masks(sites, atoms, evidence, default_radius=15):
    if default_radius not in RADII: raise ValueError('unvalidated radius')
    incomplete={tuple(r['key']) for r in evidence['atoms'] if r['missing_atoms']}
    ambiguous={tuple(d['key']) for d in evidence['defects'] if d['kind']=='alternate_conformation'}
    boundaries={tuple(k) for k in evidence['artificial_terminal_keys']}
    indexed={}
    starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
    for s,e in zip(starts[:-1],starts[1:]):
        r=atoms[s:e]; indexed[(str(r.chain_id[0]),int(r.res_id[0]),str(r.ins_code[0]).strip())]=r
    for site in sites:
        key=(site['chain'],site['resnum'],site['icode']); residue=indexed[key]
        coords=residue.coord[np.isin(residue.atom_name,SITE_ATOMS[site['group']])]
        distance=min((clearance(coords,d) for d in evidence['defects']),default=None)
        artificial=site['is_break_terminus'] or (key in boundaries and site['group'] in ('NTERM','CTERM'))
        usable=key not in incomplete and key not in ambiguous and not artificial and site['functional_atoms_complete']
        site.update(coordinates_observed=key not in incomplete,defect_clearance=distance,
            is_break_terminus=bool(artificial))
        for radius in RADII: site[f'supervision_mask_{radius}']=bool(usable and (distance is None or distance>radius))
        for radius in RADII:
            site[f'supervision_mask_adaptive_{radius}']=bool(usable and all(clearance(coords,d)>(TAIL_RADII[radius] if d.get('smaller_tail_radius_candidate') else radius) for d in evidence['defects']))
        site['alternate_conformation_uncertain']=key in ambiguous
        site['supervision_mask']=site[f'supervision_mask_adaptive_{default_radius}']
    return sites
