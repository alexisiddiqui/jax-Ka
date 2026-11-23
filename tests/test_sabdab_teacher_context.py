from sabdab_v7_teacher_context import normalise_pdb


def line(chain,number,name='CA',residue='ALA',x=1.):
    return f'ATOM  {1:5d} {name:^4s} {residue:3s} {chain}{number:4d}    {x:8.3f}{2.:8.3f}{3.:8.3f}{1.:6.2f}{20.:6.2f}          {"C":>2s}'


def test_remap_keeps_real_and_pseudo_terminal_chemical_identity():
    left=line('A',1)+'\n'+line('A',2,'N','NTR')
    right=line('X',5)+'\n'+line('X',8,'N','NTR')
    a=dict(residues=[['A',1,'H',42,'A']],termini=[['A',2,'NTR','A',1]])
    b=dict(residues=[['X',5,'Q',17,'']],termini=[['X',8,'NTR','X',5]])
    assert normalise_pdb(left,a,{('H',42,'A'):('canonical',1,'')})==normalise_pdb(right,b,{('Q',17,''):('canonical',1,'')})


def test_teacher_coordinate_change_is_not_equivalent():
    mapping=dict(residues=[['A',1,'H',42,'']],termini=[])
    keys={('H',42,''):('canonical',1,'')}
    assert normalise_pdb(line('A',1,x=1.),mapping,keys)!=normalise_pdb(line('A',1,x=1.001),mapping,keys)
