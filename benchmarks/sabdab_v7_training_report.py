"""Pin completed native candidate evidence without activating a training release."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from pkabench.runtime import atomic_json, digest


def read(path):
    return json.loads(Path(path).read_text())


def run(args):
    root = args.root
    target = root / 'stage-receipt.json'
    if target.exists():
        raise FileExistsError(target)
    candidate = root / 'candidate-manifests-v2/candidate.json'
    index = read(candidate)
    semantic = read(root / 'semantic-verification.json')
    loader = read(root / 'loader-verification.json')
    smoke = read(root / 'cache-smoke-v2/summary.json')
    graph = read(root / 'graphs-v2/report.json')
    pools = read(root / 'pools-v2/summary.json')
    ownership = read(root / 'ownership/summary.json')
    patches = read(root / 'native-patches-v2/summary.json')
    assert semantic['status'] == loader['status'] == 'passed'
    assert semantic['candidate_index_sha256'] == loader['candidate_index_sha256'] == smoke['candidate_index_sha256'] == digest(candidate)
    assert graph['source_pools_sha256'] == pools['membership_sha256'] == semantic['membership_sha256']
    assert graph['records'] == semantic['sabdab_graphs_checked'] == ownership['actions']['include_native']
    assert graph['sites'] == semantic['sabdab_teacher_queries_checked']
    assert ownership['unresolved'] == 0
    assert semantic['native_admission_sha256'] == digest(root / 'native-patches-v2/summary.json')
    assert not any(c['geometry_differences'] for c in smoke['patch_controls'])
    assert index['release_active'] is False and index['masif'] is False
    for entry in index['sources']:
        assert digest(entry['manifest']) == entry['manifest_sha256']
    test_log = args.tests
    assert '20 passed' in test_log.read_text() and 'failed' not in test_log.read_text()
    code_paths = [
        'benchmarks/sabdab_v7_ownership.py', 'benchmarks/sabdab_v7_training_pools.py',
        'benchmarks/sabdab_v7_graph_build.py', 'benchmarks/sabdab_v7_native_patches.py',
        'benchmarks/sabdab_v7_native_routes.py', 'benchmarks/sabdab_v7_cache_smoke_prepare.py',
        'benchmarks/sabdab_v7_training_verify.py', 'benchmarks/sabdab_v7_training_report.py',
        'src/pkatrain/native_paired_graph.py', 'src/pkatrain/production_native.py',
        'src/pkatrain/production_loading.py', 'src/pkatrain/production_alt.py',
        'src/pkatrain/production_graphs.py', 'tests/test_native_cache_routes.py',
        'tests/test_native_paired_graph.py', 'tests/test_source_loading.py']
    # Checks ran against a frozen snapshot, rather than mutable workspace code.
    for name in ['src/pkatrain/production_loading.py', 'src/pkatrain/production_alt.py',
                 'src/pkatrain/production_native.py', 'src/pkatrain/native_paired_graph.py',
                 'benchmarks/sabdab_v7_training_verify.py', 'tests/test_native_cache_routes.py',
                 'tests/test_source_loading.py', 'tests/test_native_paired_graph.py']:
        assert digest(args.repo / name) == digest(root / 'loader-code' / name)
    evidence = [
        candidate, root / 'ownership/summary.json', root / 'ownership/policy.json',
        root / 'ownership/membership.parquet', root / 'ownership/pair-checks.parquet',
        root / 'pools-v2/summary.json', root / 'pools-v2/membership.parquet',
        root / 'native-patches-v2/summary.json', root / 'native-routes-v2/summary.json',
        root / 'graphs-v2/report.json', root / 'graph-canary-v2/report.json',
        root / 'semantic-verification.json', root / 'loader-verification.json',
        root / 'cache-smoke-v2/summary.json', test_log]
    store_paths = [Path(graph['store']), Path(patches['patch_store'])]
    stores = []
    for path in store_paths:
        metadata = read(path / 'metadata.json')
        stores.append(dict(path=str(path), compressed_bytes=metadata['compressed_bytes'],
            reader_files=len(list(path.iterdir())), zstd_level=metadata['zstd_level'],
            verification_sha256=digest(path / 'verification.json')))
        evidence.extend([path / 'metadata.json', path / 'verification.json'])
    report = dict(status='native_candidate_build_and_loader_checks_complete',
        created_utc=datetime.now(timezone.utc).isoformat(), release_active=False,
        native_candidate_verified=True, production_recipe_integrated=False,
        counts=pools['counts'], catalogue_records=semantic['all_catalogue_memberships_checked'],
        sabdab_queries=graph['sites'], ownership_actions=ownership['actions'],
        unresolved_copy_reviews=0, native_pinder_patches=patches['counts'],
        graph_stores=stores, incremental_compressed_graph_bytes=sum(s['compressed_bytes'] for s in stores),
        real_cache_probes={k: v['sample_records'] for k, v in smoke['datasets'].items()},
        full_native_payload_copied_or_rehashed=False, targeted_tests_passed=20,
        augmentation_policy='unassessed donor receipts fall back to native; no new apo labels',
        activation_gates=index['activation_gates'],
        optional_view_gate='qualify native-context and donor-role receipts before enabling alternative geometry',
        training_recipe_gate='explicit source mixing and required auxiliary-table compatibility remain to be implemented',
        masif=False, model_training_performed=False, gpu_jobs=0,
        release_transferred=False, committed=False,
        superseded=['graphs', 'native-patches', 'native-routes', 'cache-smoke'],
        evidence_pins={str(p): digest(p) for p in evidence},
        code_pins={name: digest(args.repo / name) for name in code_paths})
    atomic_json(target, report)
    print(json.dumps({k: report[k] for k in ['status', 'counts', 'incremental_compressed_graph_bytes',
                                           'native_candidate_verified', 'production_recipe_integrated', 'release_active']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'repo', 'tests']:
        parser.add_argument('--' + name, required=True, type=Path)
    run(parser.parse_args())
