# Side-chain GQT context augmentation

This repeats the cleaned 5k GQT augmentation experiment with native side-chain
heavy-atom geometry. The train/validation split, labels, quality masks, random
seed, optimizer, 20-epoch selection rule, 20 Å C-alpha graph, and approximately
50k parameter budget are held fixed.

The four arms are baseline, 10% hidden dropout, 5% context masking, and both.
Context masks use the same deterministic per-structure/per-epoch draws as the
backbone experiment. Supervised residues are protected. For a masked residue,
the model retains amino-acid identity and node order, but loses side-chain local
coordinates, atom-presence bits, frame validity, and all incoming and outgoing
geometric edges. Validation inputs are complete and unaugmented.

This is a single-seed development ablation against PypKa teacher labels. Any
promising setting must be confirmed with independent seeds before selection.
