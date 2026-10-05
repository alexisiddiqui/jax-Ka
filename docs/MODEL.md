# Model definition and approximation boundary

The implementation is an empirical mean-field model using numerical ingredients from PROPKA 3.0/Nov30. It does not reproduce the original pKa-order-dependent, iterative determinant assignment. Reference discrepancies are expected and must be measured. Upstream source locations are listed in `SOURCES.md`.

## Frozen structure and soft sequence

There are N fixed positions, A=20 candidate identities and G=9 channels. Channels 0–6 are ASP, GLU, HIS, CYS, TYR, LYS and ARG. Channels 7–8 are NTERM and CTERM. Let P[i,a] be a row-stochastic sequence probability matrix, after any fixed-covalent identity clamp.

For a side-chain type g, w[i,g] = P[i,aa(g)] times the static group mask. For a terminal site, w is its static presence mask, independent of side-chain identity. Titratable alternatives at the same position never interact with one another. A terminal site can interact with the side chain at its own residue; N- and C-terminal sites can both exist on a single-residue segment.

The state h[i,g] is a **conditional protonated fraction**, not a probability-weighted fraction and not a charged-state fraction. The deprotonated charge q0[g] is -1 for acids and 0 for bases. Therefore the physical outgoing charge is

    Q[i,g] = w[i,g] * (q0[g] + h[i,g]).

Averaging pKas or acid/base character before Henderson–Hasselbalch would produce a different model. Nonionizable identities contribute environmental and H-bond features without becoming titration-state channels.

This is a deterministic soft-feature relaxation, not an exact expectation over discrete sequences: a nonlinear function of mean features differs from the mean of that nonlinear function. Mean-field neighbor responses also do not retain sequence/protonation correlations between sites.

## Geometry preprocessing

Biotite supplies canonical heavy-atom CCD templates and connectivity. Each candidate is rigidly placed in a backbone-local N–CA–C frame. Native heavy side-chain atoms are preserved; alternate identities use one fixed CCD conformation unless explicitly overridden. There is no side-chain optimization, clash removal, rotamer sampling, topology-changing mutation or backbone motion inside JIT.

The center definitions follow the inspected legacy residue definitions, including ARG CZ. A glycine position uses a virtual C-beta anchor for graph construction. Actual interaction centers are type-specific, not universally C-beta. Candidate donor/acceptor coordinates and virtual H directions are reduced into pair kernels. Histidine tautomers and multiply hydrogenated donor orientations are simplified; the old protonator is not ported.

Environment and pair graphs are conservative radius unions over all candidate reaches, with separate maximum padded degrees Ke and Kc. A manually specified K below the required degree raises an error. Padding uses finite zero kernels and a safe index, never NaN coordinates or an out-of-range sentinel. The graphs include cross-chain spatial edges independently of peptide topology.

The preprocessing loop forms atomwise intermediates for one target/source residue edge at a time. It does not create a global N-by-N atom-pair tensor. The runtime contains no atom axis.

## Intrinsic pKa

Per target channel, sequence contractions produce atomistic radial volume V and burial mass M:

    V[i,g] = V_backbone[i,g] + sum_{k,a} P[neighbor(i,k),a] V_side[i,k,g,a]
    M[i,g] = M_backbone[i,g] + sum_{k,a} P[neighbor(i,k),a] M_side[i,k,g,a].

Each source heavy atom contributes its source atom-type volume divided by max(2.75^4, r^4), for r < 20 Å. Burial counts source heavy atoms within r < 15 Å. Same-residue atoms are excluded from these environmental counts, while residues with the same number on a different chain are included. Native disulfide side-chain atoms still contribute environmental volume even though their thiol ionization is disabled.

Default burial and desolvation are

    b = clip((M - 280) / (560 - 280), 0, 1)
    D = Q_formal * (-13) * V * (0.25 + 0.75*b),

where Q_formal is -1 for acidic groups and +1 for basic groups. The inspected Nov30 allowance is zero. Hard geometry cutoffs are constants with respect to P. Burial clipping is piecewise differentiable in P, not globally smooth.

Intrinsic pKa is model pKa plus desolvation, neutral/backbone H-bond terms, and burial-scaled carboxylate backbone reorganization. Backbone donor availability carries the sequence dependence of proline: an internal peptide NH contribution is suppressed by P(Pro). Chain starts and genuine breaks have no preceding peptide NH. Terminal carboxylate acceptors are represented as explicit ionizable sites and are not double-counted as neutral backbone acceptors.

For neutral neighbors, a target-donor contribution raises the intrinsic pKa; a target-acceptor contribution lowers it. The strongest allowed geometric donor–acceptor pair is used. This and the candidate virtual-H treatment are approximations to the original determinant-specific rules and special-case exceptions.

## Electrostatics and protonation-state H bonds

For an active pair, the frozen Coulomb geometry is

    geometry(r) = 244.12 / max(r,4) * clip((10-max(r,4))/6, 0, 1).

Pair burial is clip((M_i + M_j - 560)/560,0,1), and epsilon is 160 - 130*pair_burial. A sequence-dependent burial eligibility gate uses sigmoid((M_i+M_j-280)/gate_width), with default gate_width=20; setting width=0 uses the hard threshold. The inspected COO–TYR eligibility exception is retained. This smooth gate is deliberately not exact legacy behavior.

C is the geometry divided by epsilon, multiplied by eligibility and masks. Its units are kBT ln(10), appropriate for pKa-shift energy differences. It is not an already signed per-residue PROPKA determinant.

The proposed pair-state energy is

    E_ij / (kBT ln(10)) = C_ij q_i q_j
                         - Hd_ij h_i (1-h_j)
                         - Hr_ij (1-h_i) h_j.

Hd and Hr are directed, nonnegative H-bond strengths. Reciprocity is enforced numerically. The conditional protonation field is

    Phi_i = sum_j w_j * [C_ij*q0_j - Hd_ij + (C_ij+Hd_ij+Hr_ij)*h_j].

This is a new mean-field energy model; replacing the original determinant logic with these energies is a scientific approximation. It does not preserve every original H-bond or Coulomb interaction assignment. For fractional sequence identities at a terminal-owning residue, own side-chain/terminal correlations are also treated approximately.

## Fixed-iteration solve and gradients

At each pH:

    target(h) = sigmoid(ln(10) * (intrinsic_pKa - pH - Phi(h)))
    h_next = (1-alpha)*h + alpha*target(h).

The default is 64 parallel updates with alpha=0.35. `lax.fori_loop` has a Python-static trip count. Sequence-dependent intrinsic terms and pair coefficients are computed before the occupancy loop. All N-by-G states are updated together; residues are not visited in a Python loop.

The returned residual is max(abs(target(h)-h)), not the difference between damped iterates. A probability-weighted residual is also returned, but validity uses the full conditional-state residual. A small damped step alone is not evidence of convergence.

Autodiff of charge and curves differentiates the finite-iteration algorithm. A fixed cap, damping and smooth gates do not guarantee a unique equilibrium or convergence for strong interactions. Compare iteration counts and residuals in the actual design regime. There is no implicit differentiation of the equilibrium equations in this version.

The fixed point is a stationary point of the mean-field free energy (`optx_solver.free_energy`), because the weighted pair coupling is symmetric. Under strong coupling more than one stable solution can exist. On a 1,452-complex benchmark sample, about 2% of titrating sites had more than one converged solution at some pH, and where the damped cold-start solver converged it found the lowest free-energy solution at 99.7% of pH points. The damped solver does not select a solution by energy, and continuation from neighbouring pH can follow a different branch.

`optx_solver` provides an opt-in alternative: warm-started optimistix Newton or Levenberg–Marquardt on the active channels, using the dense active-set coupling block, with implicit-adjoint gradients. It is intended for design loops; it can select a different branch from the damped solver and is not the production readout.

Direct midpoint pKa solves h_i(P,pH)=0.5 with a static number of bisection iterations over configured pH bounds. A custom JVP implements

    d(pKa) = -(partial h_i / partial P)[dP] / (partial h_i / partial pH),

where both partials differentiate the **finite-iteration** occupancy function. Forward- and reverse-mode are supported. The derivative is not obtained from the discrete bisection decisions. Bisection history is not retained for its reverse derivative; ordinary unrolled occupancy differentiation still has activation-memory costs.

Validity checks include bracketing, negative non-flat local slope, crossing error and fixed-point residual. They do not establish global uniqueness or choose a globally minimal free-energy branch. Unbracketed/flat roots have safe masked derivatives; a numerically returned root on a nonconverged branch may still have a derivative but must not be used as a valid pKa. Boolean validity is a diagnostic, not a smoothly differentiable loss constraint.

## Fast shared-grid midpoint option

`pka_from_grid(ph)` shares the H full-structure solves across every requested site. It locates the first sampled crossing, linearly interpolates the two surrounding occupancies, and differentiates the interpolated value. It reports bracket width, sampled monotonicity, slope, probability and grid residual. This avoids Q separate sets of full-structure root solves, at the cost of a piecewise-differentiable interpolation approximation.

The derivative can kink at interval changes. Sampled monotonicity is not a proof of monotonicity between samples; the reported interval width is not a measured pKa error. Evaluate grid refinement against direct roots on representative soft and hard sequences. Fixed pH limits can exclude some shifted pKas; excluded outputs are flagged, never extrapolated as confident roots.

## Complexity and intended use

For A=20, G=9, T occupancy iterations, H sampled pHs and Q direct midpoint queries:

| Operation | Approximate cost |
|---|---|
| Per-sequence environment/type contractions | O(N Ke G A + N Kc G²) |
| Single-pH occupancy solve | O(T N Kc G²) |
| H-point titration curves / shared-grid pKas | O(H T N Kc G²) |
| Direct midpoint queries with B bisection steps | O(Q B T N Kc G²) |

All-site direct roots have Q proportional to N and are not the same sparse-linear cost as one charge evaluation. They can be expensive; query selected sites or use the validated grid approximation in a design loop. The root mapper's static batch size bounds its primal occupancy working batch. It does not promise constant total reverse-mode memory.

Ke/Kc can be large for compact structures and long candidate reaches. Dense type blocks are retained in the structural cache for correctness, and memory can still become substantial without an atom axis. `TitrationModel(cache, backend="packed")` opt-in packs every active fixed-geometry type edge for the repeated occupancy field contraction; it does not prune hypothetical sequence identities or change the equations. The default remains `backend="dense"` pending full-panel regression and device-specific validation. Multiple rotamers, sequence-conditioned candidate weights, calibrated H-bond exceptions and a faithful determinant-mode implementation are not provided here.

Use charge/curve objectives where appropriate, enforce required identities explicitly, discretize candidate designs, and rescore them with an external reference on an appropriate repacked structure. This model alone is not a folding energy, structural validity check or experimentally calibrated inverse-folding objective.
