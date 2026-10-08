"""Scientific invariants and execution safeguards for the shared smoke pipeline."""
import json
from pathlib import Path
import numpy as np
import pytest
from pkabench.runtime import require_compute
from pkabench.schema import write_table, read_table, PH, GROUPS
from pkabench.prep import export_pdb, topology, write_cif, read_cif
from pkabench.annotate import annotate

DATA = Path(__file__).parent / "data"


def test_binary_selection_rejects_subcomplex():
    from biotite.structure.io import pdbx
    from pkabench.prep import validate_partner_selection, Rejection
    from jaxpropka.topology import load_topology
    atoms = load_topology(DATA / 'two_chains.pdb').atoms
    chains = list(dict.fromkeys(map(str, atoms.chain_id)))
    row = {'partner_A_chains': [chains[0]], 'partner_B_chains': [chains[1]]}
    cif = pdbx.CIFFile(); pdbx.set_structure(cif, atoms)
    validate_partner_selection(atoms, cif, row)
    extra = atoms[atoms.chain_id == chains[0]].copy(); extra.chain_id[:] = 'Z'
    with pytest.raises(Rejection) as exc:
        validate_partner_selection(atoms + extra, cif, row)
    assert exc.value.code == 'multi_partner'


def test_teacher_site_universe_is_explicit():
    from pkabench.adapters.base import TEACHER
    from pkabench.gates import compare_deposit
    assert TEACHER['ser_thr_titration'] is False
    assert compare_deposit({'pypka_params': {'ser_thr_titration': True}}, {'ser_thr_titration': False})['status'] == 'fail'
    assert compare_deposit({'mc_params': {'temp': 298}}, {'temp': 298.15})['status'] == 'fail'


def test_component_audit_distinguishes_remote_from_nearby():
    from biotite.structure import AtomArray
    from biotite.structure.io import pdbx
    from pkabench.audit import component_inventory
    atoms = AtomArray(3)
    atoms.chain_id[:] = ['A', 'L', 'M']; atoms.res_id[:] = [1, 2, 3]
    atoms.res_name[:] = ['ALA', 'SO4', 'MG']; atoms.element[:] = ['C', 'S', 'MG']
    atoms.atom_name[:] = ['CA', 'S', 'MG']; atoms.coord[:] = [[0, 0, 0], [5, 0, 0], [50, 0, 0]]
    cif = pdbx.CIFFile(); pdbx.set_structure(cif, atoms)
    entries = component_inventory(atoms, cif, ['A'])
    assert [(e['code'], e['min_pair_distance']) for e in entries] == [('ligand', 5.), ('metal', 50.)]


def test_compute_guard(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm"): require_compute()
    monkeypatch.setenv("SLURM_JOB_ID", "1"); monkeypatch.setenv("SLURMD_NODENAME", "comp1400")
    with pytest.raises(RuntimeError, match="excluded"): require_compute()
    monkeypatch.setenv("SLURMD_NODENAME", "comp0600"); monkeypatch.setenv("SLURM_MEM_PER_CPU", "1024")
    with pytest.raises(RuntimeError, match="2G"): require_compute()
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "2048"); require_compute()


def test_three_chain_two_partner_geometry():
    from jaxpropka.topology import load_topology
    t = load_topology(DATA/"two_chains_far.pdb")
    atoms = t.atoms.copy(); extra = atoms[atoms.chain_id == atoms.chain_id[0]].copy()
    extra.chain_id[:] = "Z"; extra.coord[:, 1] += 100
    t = topology(atoms + extra)
    chains = list(dict.fromkeys(atoms.chain_id))
    rows, delta, summary = annotate(t, {"A": [chains[0], "Z"], "B": [chains[1]]})
    assert abs(summary["half_sum_buried_area"]) < 1e-4
    assert {r["partner"] for r in rows} == {"A", "B"}
    assert sum(r["residue_delta_sasa"] for r in rows) == pytest.approx(delta.sum())


def test_canonical_export_preserves_identity(tmp_path):
    from jaxpropka.topology import load_topology
    atoms = load_topology(DATA/"two_chains.pdb").atoms
    atoms.chain_id[atoms.chain_id == atoms.chain_id[0]] = "A-2"
    atoms.ins_code[0:4] = "X"
    mapping = export_pdb(atoms, tmp_path/"input.pdb")
    assert any(k[0] == "A-2" for k in mapping.values())
    write_cif(tmp_path/"AB.cif", atoms)
    restored = read_cif(tmp_path/"AB.cif")
    np.testing.assert_allclose(restored.coord, atoms.coord)
    np.testing.assert_array_equal(restored.chain_id, atoms.chain_id)


def test_schema_rejects_invalid_curves(tmp_path):
    row = dict(complex_id="x", chain="A", resnum=1, icode="", group="ASP", state="AB", method="test", pka=4., status="ok", curve=[.5]*72, curve_source="native")
    with pytest.raises(ValueError, match="curve"): write_table(tmp_path/"p.parquet", "predictions", [row])
    row["curve"] = [.5]*73
    write_table(tmp_path/"p.parquet", "predictions", [row])
    assert len(read_table(tmp_path/"p.parquet")[0]["curve"]) == 73
    with pytest.raises(ValueError, match="duplicate"): write_table(tmp_path/"p.parquet", "predictions", [row,row])
    row["curve"]=None; row["pka"]=None; row["status"]="failed"
    write_table(tmp_path/"p.parquet", "predictions", [row])
    assert read_table(tmp_path/"p.parquet")[0]["curve"] is None


def test_linkage_sign_and_reference():
    from pkabench.linkage import integrate
    result=integrate(np.ones(73))
    assert result[36] == 0
    assert np.all(np.diff(result)>0)
    np.testing.assert_allclose(result,1.364*(PH-7))


def test_metric_degeneracy_and_null():
    from pkabench.score import metrics
    assert metrics([],[])["skill"] is None
    assert metrics([0,0],[0,0])["skill"] is None
    assert metrics([1,-1],[0,0])["skill"] == 0
    assert metrics([1,-1],[1,-1])["skill"] == 1


def test_queue_accounting_counts_pending_elements():
    from pkabench.jobs import queued_cores
    assert queued_cores("4\n1\n1\n") == 6
    assert queued_cores("\n") == 0
    with pytest.raises(ValueError): queued_cores("1-400")


def test_incomplete_linkage_is_not_silent():
    from pkabench.linkage import linkage
    site=dict(complex_id="x",chain="A",resnum=1,icode="",group="ASP",partner="A",is_break_terminus=False)
    assert linkage([], [site])["status"] == "incomplete_charge_coverage"


def test_partner_pairing_and_break_exclusion():
    from pkabench.score import paired
    site=dict(complex_id="x",chain="B",resnum=1,icode="",group="ASP",partner="B",is_break_terminus=False,min_partner_distance=2.)
    rows=[{**site,"method":"test","state":s,"pka":v,"status":"ok"} for s,v in (("AB",6.),("A",99.),("B",4.))]
    assert list(paired(rows,[site],"test").values())[0][0] == 2
    site["is_break_terminus"]=True
    assert not paired(rows,[site],"test")


def test_component_policy_rejects_before_filtering():
    from biotite.structure import AtomArray
    from biotite.structure.io import pdbx
    from pkabench.prep import clean_components, Rejection
    atoms=AtomArray(1); atoms.chain_id[:]='X'; atoms.res_id[:]=1
    atoms.res_name[:]='NA'; atoms.element[:]='NA'; atoms.atom_name[:]='NA'
    cif=pdbx.CIFFile(); pdbx.set_structure(cif,atoms)
    assert len(clean_components(atoms,cif)) == 0
    atoms.res_name[:]='MG'; atoms.element[:]='MG'; atoms.atom_name[:]='MG'
    with pytest.raises(Rejection) as error: clean_components(atoms,cif)
    assert error.value.code == 'metal'
    atoms.res_name[:]='SO4'; atoms.element[:]='S'; atoms.atom_name[:]='S'
    with pytest.raises(Rejection) as error: clean_components(atoms,cif)
    assert error.value.code == 'ligand'


def test_missing_backbone_has_structured_code():
    from jaxpropka.topology import load_topology
    from pkabench.prep import Rejection
    atoms=load_topology(DATA/'peptide.pdb').atoms
    with pytest.raises(Rejection) as error: topology(atoms[np.arange(len(atoms))!=0])
    assert error.value.code == 'missing_backbone'


def test_worker_timeout_kills_process_group(tmp_path):
    import subprocess
    import sys
    from pkabench.adapters.base import execute
    with pytest.raises(subprocess.TimeoutExpired):
        execute([sys.executable,'-c','import time; time.sleep(10)'],tmp_path,.1)
    assert (tmp_path/'stderr.log').exists()


def test_method_failure_retains_all_expected_states(tmp_path):
    from pkabench.adapters.base import Adapter
    from jaxpropka.topology import load_topology
    atoms=load_topology(DATA/'two_chains.pdb').atoms
    paths={}
    for state in ('AB','A','B'):
        paths[state]=tmp_path/f'{state}.cif'; write_cif(paths[state],atoms)
    site=dict(complex_id='x',chain='A',resnum=1,icode='',group='ASP',partner='A')
    adapter=Adapter('pkai',[site],tmp_path/'absent-runtime')
    rows=adapter.run(paths,tmp_path/'work')
    assert len(rows)==2
    assert {r['state'] for r in rows} == {'AB','A'}
    assert all(r['status']=='failed' for r in rows)


def test_null_has_exact_zero_shift_and_linkage(tmp_path):
    from pkabench.adapters.base import Adapter
    from pkabench.linkage import linkage
    site=dict(complex_id='x',chain='A',resnum=1,icode='',group='ASP',partner='A',is_break_terminus=False)
    adapter=Adapter('null',[site],tmp_path)
    rows=adapter.run({s:tmp_path/'unused' for s in ('AB','A','B')},tmp_path/'work')
    assert rows[0]['pka']==rows[1]['pka']
    result=linkage(rows,[site])
    assert result['status']=='ok'
    assert max(abs(x) for x in result['delta_q'])==0


def test_job_resume_merge_and_input_tamper(tmp_path,monkeypatch):
    from pkabench.jobs import run_job,merge
    from pkabench.runtime import atomic_json
    from jaxpropka.topology import load_topology
    runtime=tmp_path/'runtime'; (runtime/'manifests').mkdir(parents=True)
    (runtime/'manifests/runner.requirements.lock').write_text('fixture-runtime\n')
    monkeypatch.setenv('PKABENCH_RUNTIME',str(runtime))
    campaign=tmp_path/'campaign'; work=campaign/'structures/x'; work.mkdir(parents=True)
    atomic_json(campaign/'manifest.json',{'candidates':[{'complex_id':'x'}]})
    atoms=load_topology(DATA/'two_chains.pdb').atoms
    for state in ('AB','A','B'): write_cif(work/f'{state}.cif',atoms)
    site=dict(complex_id='x',chain='A',resnum=1,icode='',group='ASP',partner='A',is_break_terminus=False)
    write_table(work/'sites.parquet','sites',[site])
    write_table(campaign/'structures.parquet','structures',[{'complex_id':'x'}])
    run_job(campaign,'x','null')
    receipt=campaign/'jobs/null/x.json'; first=receipt.read_bytes()
    run_job(campaign,'x','null')
    assert receipt.read_bytes()==first
    merge(campaign,['null'])
    assert len(read_table(campaign/'predictions.parquet'))==2
    with pytest.raises(ValueError,match='duplicate methods'): merge(campaign,['null','null'])
    (work/'AB.cif').write_text((work/'AB.cif').read_text()+'\n# changed input\n')
    with pytest.raises(ValueError,match='hash mismatch'): merge(campaign,['null'])
    with pytest.raises(ValueError,match='incompatible hashes'): run_job(campaign,'x','null')


def test_deposit_defaults_cannot_masquerade_as_evidence():
    from pkabench.gates import compare_deposit
    expected={'epsin':15.,'epssol':80.,'pbc_dimensions':0,'keep_ions':False}
    assert compare_deposit({'epsin':15},expected)['status']=='unresolved'
    assert compare_deposit({'epsin':20},expected)['status']=='fail'
    actual={'pypka_params':{'keep_ions':'false'},'delphi_params':{'epsin':15,'epssol':80,'pbc_dim':0}}
    assert compare_deposit(actual,expected)['status']=='pass'
    actual['epsin']=20
    with pytest.raises(ValueError,match='conflicting'): compare_deposit(actual,expected)


def test_hpc_scripts_parse_and_reject_login_execution():
    import os
    import subprocess
    workspace=Path(__file__).resolve().parents[2]
    install=workspace/'_HPC/install/jax-Ka/pkabench'
    submission=workspace/'_HPC/submission/jax-Ka/pkabench'
    if not install.exists(): pytest.skip('workspace HPC wrappers not installed')
    for path in [*install.glob('*.sh'),*submission.glob('*.sh'),*submission.glob('*.sbatch')]:
        subprocess.run(['bash','-n',str(path)],check=True)
    env=os.environ.copy(); env.pop('SLURM_JOB_ID',None)
    result=subprocess.run(['bash',str(install/'common.sh')],env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'Slurm compute allocation' in result.stderr
    env['SLURM_JOB_ID']='test'; env['SLURMD_NODENAME']='comp1400'
    result=subprocess.run(['bash',str(install/'common.sh')],env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'comp1400 is excluded' in result.stderr
