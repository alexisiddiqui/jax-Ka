"""Checkpoint-backed batch inference, feature parity and compilation reuse."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pkanet.features import featurise, prepare_file, load_features
from pkanet.inference import Predictor, load_model, pad_graph, capacities
from pkanet.ogqt import predict_with_trace
from pkanet.model import PKPDB_PK_MOD
from pkanet.cli import main

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'models/ogqt-production-50k'
DATA = ROOT/'tests/data'


def test_site_feature_rules_match_training_source():
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n,ast.FunctionDef)}
    actual = functions(ROOT/'src/pkanet/features.py')
    reference = functions(ROOT/'src/pkatrain/site_graph_data.py')
    for name in ('frames','candidate_sites','build_site_graph'):
        assert actual[name] == reference[name]


def test_checkpoint_integrity(tmp_path):
    params,info=load_model(MODEL)
    assert sum(x.size for x in jax.tree.leaves(params)) == 67770
    shutil.copytree(MODEL,tmp_path/'model')
    path=tmp_path/'model/params.npz'
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='SHA256'): load_model(path.parent)


def test_features_and_content_cache(tmp_path):
    path,meta=prepare_file(DATA/'two_chains.pdb',tmp_path)
    stamp=path.stat().st_mtime_ns
    assert prepare_file(DATA/'two_chains.pdb',tmp_path)[0] == path
    assert path.stat().st_mtime_ns == stamp
    graph,_=load_features(path)
    assert graph['nodes'].shape[1] == 24
    assert np.all(graph['nodes'][:,22]==0)
    assert not np.any(graph['query_group']==6)
    assert len(meta['sites']) == len(graph['query_site'])
    changed,_=prepare_file(DATA/'two_chains.pdb',tmp_path,gap_policy='error')
    assert changed != path


def test_mixed_sizes_batch_parity_reuse_and_partial_batch(tmp_path):
    records=[prepare_file(DATA/name,tmp_path) for name in ('peptide.pdb','two_chains.pdb','two_chains_far.pdb')]
    assert len({tuple(m['dimensions']) for _,m in records})>1
    predictor=Predictor(MODEL,batch_size=2)
    output=dict((i,(s,p)) for i,s,p in predictor.predict(records))
    expected_shapes=len({capacities(m['dimensions']) for _,m in records})
    assert predictor.last_stats['compiled_shapes']==expected_shapes
    assert predictor.last_stats['structures']==3
    for i,(path,_) in enumerate(records):
        graph,_=load_features(path)
        graph=jax.tree.map(jnp.asarray,graph)
        expected=predict_with_trace(predictor.params,graph)['predicted_shift']
        np.testing.assert_allclose(output[i][0],expected,atol=3e-5,rtol=3e-5)
        np.testing.assert_allclose(output[i][1],expected+PKPDB_PK_MOD[graph['query_group']],atol=3e-5,rtol=3e-5)
    again=list(predictor.predict(records))
    assert predictor.last_stats['compiled_shapes']==0
    for i,s,p in again:
        np.testing.assert_array_equal(p,output[i][1])


def test_overflow_and_memory_limits(tmp_path):
    record=prepare_file(DATA/'two_chains.pdb',tmp_path)
    graph,_=load_features(record[0])
    with pytest.raises(ValueError,match='overflow'): pad_graph(graph,(1,1,1,1,1))
    predictor=Predictor(MODEL,max_input_bytes=1)
    with pytest.raises(ValueError,match='input bytes'): list(predictor.predict([record]))
    with pytest.raises(ValueError,match='GPU'): Predictor(MODEL,backend='triton',device='cpu')


def test_cli_checkpoint_prediction_input_order(tmp_path):
    inputs=[str(DATA/'two_chains.pdb'),str(DATA/'peptide.pdb')]
    out=tmp_path/'predictions.jsonl'
    main(['predict',*inputs,'--model',str(MODEL),'--cache',str(tmp_path/'cache'),
          '--output',str(out),'--batch-size','2'])
    rows=[json.loads(line) for line in out.read_text().splitlines()]
    assert [r['source'] for r in rows] == inputs
    assert [r['input_index'] for r in rows]==[0,1]
    assert all(np.isfinite(site['pka']) for row in rows for site in row['sites'])
    summary=json.loads(out.with_suffix('.jsonl.summary.json').read_text())
    assert summary['structures']==2
    assert summary['model_sha256']==hashlib.sha256((MODEL/'params.npz').read_bytes()).hexdigest()


def test_cli_prepare_list_and_empty_predictor(tmp_path):
    listing=tmp_path/'inputs.txt';listing.write_text(str(DATA/'peptide.pdb')+'\n')
    out=tmp_path/'manifest.json'
    main(['prepare','--input-list',str(listing),'--cache',str(tmp_path/'cache'),'--output',str(out)])
    assert len(json.loads(out.read_text())['records'])==1
    p=Predictor(MODEL)
    assert list(p.predict([]))==[]
    assert p.last_stats['structures']==0


def test_cuda_triton_parity(tmp_path):
    import os
    try:
        devices=jax.devices('gpu')
    except RuntimeError:
        devices=[]
    if not devices:
        if os.environ.get('OGQT_REQUIRE_GPU')=='1': pytest.fail('GPU validation requires a GPU')
        pytest.skip('requires JAX GPU and Triton')
    # Do not silently skip missing Triton on a GPU validation run.
    records=[prepare_file(DATA/name,tmp_path) for name in ('peptide.pdb','two_chains.pdb','two_chains_far.pdb')]
    native=Predictor(MODEL,device='gpu',backend='jax',batch_size=2)
    indexed=Predictor(MODEL,device='gpu',backend='triton',batch_size=2)
    reference={i:p for i,_,p in native.predict(records)}
    for i,_,p in indexed.predict(records):
        np.testing.assert_allclose(p,reference[i],atol=2e-4,rtol=2e-4)
    list(indexed.predict(records))
    assert indexed.last_stats['compiled_shapes']==0


def test_known_disulfide_cysteine_is_context_only(monkeypatch):
    from dataclasses import replace
    import pkanet.features as features
    from jaxpropka.parameters import ALPHABET
    top=features.load_topology(DATA/'two_chains.pdb',freeze_disulfides=True,gap_policy='cap')
    identities=top.native_index.copy();identities[0]=ALPHABET.index('C')
    disulfide=top.disulfide.copy();disulfide[0]=True
    top=replace(top,native_index=identities,disulfide=disulfide)
    monkeypatch.setattr(features,'load_topology',lambda *a,**kw:top)
    graph,_=features.featurise(DATA/'two_chains.pdb')
    assert np.any((graph['site_residue']==0)&(graph['site_type']==3))
    assert not np.any((graph['query_residue']==0)&(graph['query_group']==3))
