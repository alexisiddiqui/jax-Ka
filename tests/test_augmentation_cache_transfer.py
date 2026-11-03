import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

script=Path(__file__).with_name('augmentation_cache_transfer.py')
if not script.exists():script=Path(__file__).parents[1]/'benchmarks/augmentation_cache_transfer.py'
spec=importlib.util.spec_from_file_location('cache_transfer',script)
transfer=importlib.util.module_from_spec(spec);spec.loader.exec_module(transfer)


def fixture(tmp_path):
    stage=tmp_path/'release.incoming';stage.mkdir();(stage/'features').mkdir()
    data=b'packed immutable bytes'*100
    (stage/'features/records.bin').write_bytes(data)
    manifest=dict(format='augmentation-feature-cache-v1',verified=True,
        files={'features/records.bin':hashlib.sha256(data).hexdigest()})
    expected=tmp_path/'expected.json';expected.write_text(json.dumps(manifest))
    args=argparse.Namespace(source=stage,destination=tmp_path/'release',expected_manifest=expected,progress=tmp_path/'progress.json')
    return args,manifest


def test_verify_publishes_only_unchanged_allowlist_and_original_manifest(tmp_path):
    args,manifest=fixture(tmp_path);transfer.verify(args)
    assert not args.source.exists()
    assert (args.destination/'manifest.json').read_bytes()==args.expected_manifest.read_bytes()
    assert json.loads(args.progress.read_text())['status']=='complete'
    assert (args.destination/'features/records.bin').is_file()


def test_verify_resumes_after_marker_installed_but_before_directory_rename(tmp_path):
    args,manifest=fixture(tmp_path)
    (args.source/'manifest.json').write_bytes(args.expected_manifest.read_bytes())
    transfer.verify(args)
    assert (args.destination/'manifest.json').read_bytes()==args.expected_manifest.read_bytes()


@pytest.mark.parametrize('failure',['corrupt','extra','escape','symlink'])
def test_bad_or_unlisted_payload_never_publishes(tmp_path,failure):
    args,manifest=fixture(tmp_path)
    if failure=='corrupt':(args.source/'features/records.bin').write_bytes(b'bad')
    elif failure=='extra':(args.source/'worker-checkpoint.json').write_text('{}')
    elif failure=='symlink':
        p=args.source/'features/records.bin';data=p.read_bytes();p.unlink()
        outside=tmp_path/'outside';outside.write_bytes(data);p.symlink_to(outside)
    else:
        manifest['files']['../outside']='0'*64;args.expected_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):transfer.verify(args)
    assert not args.destination.exists() and not (args.source/'manifest.json').exists()
