"""Explicit surface preparation/validation, compatible with container Python 3.6.

This is an additional preparation policy over the pinned MaSIF paper runtime.
No nonfinite values or absent chemical features are replaced with zeros.
"""
import re
import numpy as np

SURFACE_POLICY = 'terminal-oxt-mesh-coverage-v2'


def validate_reduce_warning(input_text,output_text,stderr):
    """Validate Reduce 3.23's rc=1 abandoned-clique output, not arbitrary errors.

    Original MaSIF consumes this output. Here the return code is retained as a
    warning only if all heavy atoms and backbone coordinates survive, the PDB
    is complete and the reported newly added H count matches its records.
    """
    if not output_text.startswith('USER  MOD reduce.3.23.130521'):
        raise ValueError('unrecognized Reduce warning output/version')
    if not any(line.strip()=='END' for line in output_text.splitlines()):
        raise ValueError('truncated Reduce PDB output')
    def atoms(text):
        heavy={};hydrogens=0
        for line in text.splitlines():
            if not line.startswith(('ATOM  ','HETATM')):continue
            name=line[12:16].strip();element=line[76:78].strip() if len(line)>=78 else ''
            position=np.array([float(line[k:k+8]) for k in (30,38,46)])
            if not np.isfinite(position).all():raise ValueError('nonfinite Reduce coordinates')
            if element in ('H','D') or not element and name.startswith(('H','D')):
                hydrogens+=1;continue
            key=(line[21:27],line[17:20],name)
            if key in heavy:raise ValueError('duplicate Reduce heavy-atom identity')
            heavy[key]=position
        return heavy,hydrogens
    before,_=atoms(input_text);after,hydrogens=atoms(output_text)
    if not before or set(before)!=set(after):raise ValueError('Reduce warning lost/changed heavy atoms')
    for key,position in before.items():
        if key[-1] in ('N','CA','C','O') and not np.allclose(position,after[key],atol=.002,rtol=0):
            raise ValueError('Reduce warning changed native backbone')
    found=re.search(r'Found\s+(\d+) hydrogens',stderr);added=re.search(r'Added\s+(\d+) hydrogens',stderr)
    removed=re.search(r'Removed\s+(\d+) hydrogens',stderr)
    if not all((found,added,removed)) or int(found.group(1))!=0 or int(removed.group(1))!=0:
        raise ValueError('Reduce warning lacks expected hydrogen summary')
    if hydrogens<=0 or hydrogens!=int(added.group(1)):
        raise ValueError('Reduce hydrogen output/count differs from summary')
    return dict(status='complete_with_partial_optimization_warning',heavy_atoms=len(after),added_hydrogens=hydrogens,
                reduce_version='3.23.130521',returncode=1,backbone_preserved=True)


def complete_terminal_oxygens(atoms, segments):
    """Add only absent OXT at each exported free C terminus; retain input atoms.

    A planar carboxylate uses the C--CA/C--O external bisector and a 1.25 A
    C--OXT bond. Gap fragments are already explicit free chains in this export.
    Missing side-chain atoms remain failures requiring separate repair.
    """
    atoms = [dict(a) for a in atoms]; added = []
    for i, atom in enumerate(atoms):
        if i + 1 < len(atoms) and segments[i + 1] == segments[i]:
            continue
        if 'OXT' in atom:
            continue
        if not all(k in atom for k in ('CA', 'C', 'O')):
            raise ValueError('cannot prepare terminal oxygen without CA/C/O')
        c = np.asarray(atom['C']); a = np.asarray(atom['CA']) - c; b = np.asarray(atom['O']) - c
        if min(np.linalg.norm(a), np.linalg.norm(b)) < .5:
            raise ValueError('degenerate C-terminal geometry')
        direction = -(a / np.linalg.norm(a) + b / np.linalg.norm(b))
        if np.linalg.norm(direction) < .1:
            raise ValueError('collinear C-terminal geometry')
        atom['OXT'] = c + 1.25 * direction / np.linalg.norm(direction); added.append(i)
    return atoms, added


def clean_mesh(vertices, faces):
    """Remove invalid faces after the FINAL vertex merge, then compact vertices."""
    vertices = np.asarray(vertices); faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('invalid mesh vertices')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError('empty or invalid mesh faces')
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('mesh face index outside vertex array')
    triangle = vertices[faces]
    area2 = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1)
    valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2]) & (area2 > 1e-10)
    faces = faces[valid]
    _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(unique)]
    if not len(faces):
        raise ValueError('all mesh faces are degenerate')
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    edges = np.concatenate([faces[:, :2], faces[:, 1:], faces[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices)))
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=count)
    # A three-vertex island has no usable curvature neighbourhood. Remove it
    # only subject to the raw-surface coverage check performed by the caller.
    faces = faces[(sizes[labels[faces]] >= 4).all(axis=1)]
    if not len(faces):
        raise ValueError('mesh has no component with a curvature neighbourhood')
    used, inverse = np.unique(faces, return_inverse=True)
    return vertices[used], inverse.reshape((-1, 3)), dict(
        removed_faces=int(len(valid) - len(faces)), removed_vertices=int(len(vertices) - len(used)))


def validate_coverage(raw_vertices, vertices, atoms):
    """Reject gross shape loss using raw MSMS coverage and source atom extent.

    These are numerical integrity checks, not thermodynamic validity tests.
    A <=1 A remesh normally has sub-A nearest-vertex errors. Every raw surface
    vertex must remain within 3 A, and all source heavy atoms within its bbox
    enlarged by 2 A. Checks apply equally to monomers and multichain partners.
    """
    from scipy.spatial import cKDTree
    distance = cKDTree(vertices).query(raw_vertices)[0]
    outside = ((atoms < vertices.min(axis=0) - 2.) | (atoms > vertices.max(axis=0) + 2.)).any(axis=1)
    metrics = dict(raw_surface_max_nearest_A=float(distance.max()),
                   raw_surface_p99_nearest_A=float(np.percentile(distance, 99)),
                   source_atoms_outside_bbox_plus_2A=int(outside.sum()))
    if distance.max() > 3. or outside.any():
        raise ValueError('mesh coverage loss: ' + str(metrics))
    return metrics
