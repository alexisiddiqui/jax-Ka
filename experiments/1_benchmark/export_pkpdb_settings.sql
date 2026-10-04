-- Read-only query for the owner of the original pKPDB PostgreSQL database.
-- No connection details are embedded; this has not been run remotely.
-- Each JSON row includes a deposited simulation's settings and identity.
-- The historical fill.py omitted temp/pdb2pqr_h_opt from stored pypka_params:
-- those require accompanying historical version/configuration provenance.
SELECT json_build_object(
    'idcode', trim(p.idcode),
    'pksimid', ps.pksimid,
    'sim_date', ps.sim_date,
    'settid', s.settid,
    'pypka_params', s.pypka_params,
    'delphi_params', s.delphi_params,
    'mc_params', s.mc_params
)
FROM protein AS p
JOIN pk_sim AS ps ON ps.pid = p.pid
JOIN sim_settings AS s ON s.settid = ps.settid
WHERE lower(trim(p.idcode)) IN ('4lzt', '1ubq')
ORDER BY p.idcode, ps.pksimid;

-- Database-wide settings census. Counts are simulation records, not necessarily
-- distinct structures. Include all stored PypKa parameters, not just version,
-- so SER/THR and other version-independent overrides can be inspected.
-- Cast to jsonb because PostgreSQL json has no equality operator for GROUP BY.
-- This read-only query has not been executed against the owner's database.
SELECT
    s.settid,
    s.pypka_params::jsonb ->> 'version' AS pypka_version,
    s.pypka_params::jsonb AS pypka_params,
    s.delphi_params::jsonb AS delphi_params,
    s.mc_params::jsonb AS mc_params,
    count(*) AS simulation_count,
    count(DISTINCT ps.pid) AS structure_count,
    min(ps.sim_date) AS first_sim_date,
    max(ps.sim_date) AS last_sim_date
FROM pk_sim AS ps
JOIN sim_settings AS s ON s.settid = ps.settid
GROUP BY s.settid, s.pypka_params::jsonb,
         s.delphi_params::jsonb, s.mc_params::jsonb
ORDER BY first_sim_date, s.settid;
