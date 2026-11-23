"""Terminus sites for pKAI (pure Python; pkai env has no biotite).

The released pKAI reader never creates terminus sites (protein.py: `self.termini = {}  # TODO ... ignored for now`), but
its model has NTR/CTR residue classes, reference atoms (NTR: N; CTR: OD1/OXT) and environment encodings (an atom N in
a residue named NTR is encoded as a charged amine, NZ_LYS; OXT as a carboxylate oxygen, O_COOH). pKAI keys residues by
(chain, number), so a terminus must be its own residue entry or it would absorb the terminal residue's own site.

add_termini moves, in the PDB text written by pkabench.prep.export_pdb:
  N-terminus: the terminal residue's backbone N, CA, C, O into a residue NTR (PypKa's NTR site, G54A7 NTRtau*.st;
              pKAI reads only N/O/S, so its own environment excludes that carbonyl O, and other sites see N and O as amine),
  C-terminus: the terminal residue's O and OXT (the carboxylate) into a residue CTR,
each with a fresh residue number (>= 9001) in the same chain. Geometry is unchanged. Only termini passed in are
converted (true chain termini; segment ends at chain breaks keep their PDB2PQR caps as before).
Returns the new text and {(chain, new_number): (kind, chain, old_number)}.
"""


def add_termini(text, nterm, cterm, ntr_atoms=("N", "CA", "C", "O")):
    """nterm/cterm: iterables of (pdb_chain, pdb_number) for the residues whose terminus is converted."""
    nterm = set(nterm); cterm = set(cterm); assign = {}; out = []; nxt = {}
    numbers = [int(l[22:26]) for l in text.splitlines() if l.startswith("ATOM")]
    if numbers and max(numbers) >= 9000: raise ValueError("residue numbers collide with terminus numbering")
    def fresh(chain, kind, old):
        k = (kind, chain, old)
        if k not in assign:
            nxt[chain] = nxt.get(chain, 9000) + 1; assign[k] = nxt[chain]
        return assign[k]
    for l in text.splitlines():
        if l.startswith("ATOM"):
            chain, num, name = l[21], int(l[22:26]), l[12:16].strip()
            kind = "NTR" if (chain, num) in nterm and name in ntr_atoms else "CTR" if (chain, num) in cterm and name in ("O", "OXT") else None
            if kind:
                l = l[:17] + f"{kind:<3}" + l[20:22] + f"{fresh(chain, kind, num):>4}" + l[26:]
        out.append(l)
    return "\n".join(out) + "\n", {(c, n): (k, c, o) for (k, c, o), n in assign.items()}
