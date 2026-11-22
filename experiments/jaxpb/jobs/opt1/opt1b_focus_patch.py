"""Prepare the separable focused-probe experiment locally, before freezing a run."""
from pathlib import Path

p = Path('src/jaxpb/maps.py')
s = p.read_text()
s = s.replace('''        if PROBE_HASH:
            psc, pord, ov_p = probe_cells_hashed(probes, lop, ncell_p, cp)''', '''        preg, pval, npreg = region_probes(probes, a, h, n, PB, rp)
        cell_probes = jnp.where(pval[:, None], preg, RC.FAR) if hstat != 1.0 and not brute else probes
        if PROBE_HASH:
            psc, pord, ov_p = probe_cells_hashed(cell_probes, lop, ncell_p, cp)''')
s = s.replace('build_cells(probes, jnp.zeros(probes.shape[0], dt), lop, ncell_p, cp, Kp, batch_size=budget_B)',
              'build_cells(cell_probes, jnp.zeros(cell_probes.shape[0], dt), lop, ncell_p, cp, Kp, batch_size=budget_B)')
s = s.replace('''            ptab = (tab_p,)
        preg, pval, npreg = region_probes(probes, a, h, n, PB, rp)''', '''            ptab = (tab_p,)''')
s = s.replace('''    zp = jnp.zeros(probes.shape[-2], dt)
    npq''', '''    projection_probes = S["preg"] if hstat != 1.0 and not brute else probes
    projection_axis = 0 if hstat != 1.0 and not brute else pa
    zp = jnp.zeros(projection_probes.shape[-2], dt)
    npq''')
s = s.replace('(probes, zp) + S["ptab"]', '(projection_probes, zp) + S["ptab"]')
s = s.replace('(pa, None, 0, 0, 0))', '(projection_axis, None, 0, 0, 0))')
s = s.replace('(pa, None, 0, 0))', '(projection_axis, None, 0, 0))')
s = s.replace('V(stage6, pa, 0, 0, 0, 0, 0, 0)(probes,', 'V(stage6, projection_axis, 0, 0, 0, 0, 0, 0)(projection_probes,')
p.write_text(s)
