# 05 — Small-molecule and ligand complexes (deferred)

**Status:** scope for a later round, after the protein–protein dataset/model in 03
and backbone-only protein mode in 04. No ligand label generation or training is
part of the first round.

## Motivation and scope

Small molecules, drugs and other bound ligands can provide additional paired
training examples. Keep their source structures and rejection annotations in the
inventory so that exclusion from the initial protein–protein set does not mean
discarding them from the project.

Distinguish crystallisation additives being considered for removal from ligands
whose binding defines the example. Neither a chemical-component name nor proximity
alone establishes that a component is dispensable.

## Paired states

- Start with a bound protein–ligand structure and rigid removal of the ligand:
  protein+ligand versus protein alone in the same protein conformation. Include
  the isolated ligand when modelling total charge or binding linkage.
- Separately inventory experimentally resolved bound/unbound structures of the
  same protein. Their changes include conformational effects and must not be
  pooled with rigid-removal labels without a distinct task label.
- State explicitly whether labels concern protein-site shifts, ligand-site shifts,
  or coupled protein–ligand proton uptake. These require different site coverage.

## Prerequisites before generation

1. Parameterisation and validated reference calculations for ligand charges,
   radii, protonation states and tautomers. Current protein-only PypKa wrappers
   do not establish that arbitrary ligands are supported.
2. Stable atom/site mapping across states, including formal charges, stereochemistry,
   covalent attachments and metal coordination. Keep covalent ligands and metal
   complexes as separately validated cases.
3. Extend the shared representation and schema beyond amino-acid groups. A
   backbone-only protein representation still needs explicit ligand chemistry
   and geometry; ligand removal is not protein side-chain masking.
4. Leakage controls for both protein families and ligand scaffolds/analogues;
   keep every bound/unbound pair in one split.
5. A small diverse pilot with complete site and charge accounting before deciding
   the method roster, dataset size and compute budget.

Report results separately from the initial protein–protein benchmark. This section
is an extension plan, not permission to relax the first-round chemical filters.
