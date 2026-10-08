# 01b — Noise floor: how much ΔpKa is just structure choice?

**Timebox:** 1 day
**Hardware:** laptop / one CPU node; no new teacher runs
**Depends on:** cluster definitions from `00_shared.md` only. Off the step-1 critical path;
run after 01 or alongside it.

---

## Question

Mine pKPDB for clusters containing ≥2 *independent apo* structures of the same protein.
Per-site pKa difference between them = crystal-to-crystal variation with zero true signal.
PDB redundancy makes this large (lysozyme alone has hundreds of entries).

Then compute apo/holo differences from the same source, stratified by distance to the
interface.

## Not just SQL

pKPDB gives pKa per entry. Defining the pairs needs more:

- **Apo vs. holo** needs PDB assembly metadata: which partner chains are present.
  "Apo" = the protein's assembly contains no binding partner.
- **Same protein** needs sequence clustering and residue mapping across entries. Use the
  MMseqs2 settings from 00 so clusters mean the same thing everywhere.
- **Same site** needs a sequence alignment between entries; residue numbering differs.

## What it measures

Apo/holo differences include conformational change between independently solved
structures. Rigid-separation ΔpKa (01, 03) excludes it by construction. So this is a
**different quantity**: say so wherever the two are compared, and don't use apo/holo
differences to validate a rigid-separation predictor without that caveat.

Contamination: pKAI and pKAI+ were trained on pKPDB, so these labels are their training
data.

## Decision rule

- Apo/holo at interface sites does **not** separate from the apo/apo null → mined pairs are
  unusable as labels.
- It **does** separate → a large free validation set exists, and the effect size is the
  accuracy bar a model must clear to be useful.

Either way the apo/apo distribution belongs in the paper: it quantifies how much of any
predicted ΔpKa could just be structure choice.

## Outputs

```
results/noise_floor/
  pairs.parquet             # apo/apo and apo/holo site pairs, with cluster + alignment ids
  noisefloor.parquet        # per-site differences, distance-to-interface stratified
  figures/
```

## Success criteria

- ≥1000 apo/apo pairs
- Apo/apo vs. apo/holo distributions with component-bootstrap CIs, per distance shell
