import biotite.structure as struc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sabdab_v7_native_scope import atom_signature, describe, parquet_rows


def atoms():
    a = struc.AtomArray(6)
    a.chain_id[:] = 'A'; a.res_id[:] = [1, 1, 1, 4, 4, 4]
    a.res_name[:] = ['ALA'] * 3 + ['CYS'] * 3
    a.atom_name[:] = ['N', 'CA', 'C'] * 2
    a.element[:] = ['N', 'C', 'C'] * 2
    a.coord[:] = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0], [11, 0, 0], [12, 0, 0]]
    return a


def test_scope_retains_residue_identity_and_physical_chain_breaks(tmp_path):
    chain = describe(atoms())[0]
    assert chain['sequence'] == 'AC'
    assert chain['backbone_breaks'] == 1
    path = tmp_path / 'scope.parquet'
    pq.write_table(pa.Table.from_pylist([dict(chains=[chain])]), path)
    decoded = pq.read_table(path).to_pylist()[0]['chains'][0]
    assert decoded['residue_keys'] == [{'chain': 'A', 'number': 1, 'insertion': ''}, {'chain': 'A', 'number': 4, 'insertion': ''}]


def test_context_identity_changes_with_coordinate_and_residue_changes():
    a = atoms(); original = atom_signature(a)
    a.coord[0, 0] += .1
    assert atom_signature(a) != original
    a = atoms(); a.res_id[0:3] = 2
    assert atom_signature(a) != original


def test_unassessed_first_record_does_not_erase_later_checked_fields():
    rows = [dict(status='unassessed', input_pins={}),
            dict(status='checked', input_pins={'/input': 'sha'}, chains=describe(atoms()), model_receipt_residues=2)]
    table = pa.Table.from_pylist(parquet_rows(rows)).to_pylist()
    assert table[1]['chains'][0]['sequence'] == 'AC'
    assert table[1]['input_pins'] == [{'path': '/input', 'sha256': 'sha'}]
    assert table[0]['model_receipt_residues'] is None
