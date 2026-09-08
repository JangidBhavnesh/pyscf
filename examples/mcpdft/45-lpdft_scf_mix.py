#!/usr/bin/env python
"""Self-consistent L-PDFT with unequal weights and mixed-spin solvers."""

from pyscf import fci, gto, mcpdft, scf
from pyscf.mcpdft.lpdft_scf import LPDFSCF


mol = gto.M(
    atom="Li 0 0 0; H 1.5 0 0",
    basis="sto-3g",
    unit="angstrom",
    verbose=4,
)
mf = scf.RHF(mol).run()

# The first block contains one triplet. The second block contains two
# singlets. LPDFSCF permits rotations only within each solver block.
triplet_solver = fci.direct_spin1.FCI(mol)
triplet_solver.spin = 2
triplet_solver = fci.addons.fix_spin(triplet_solver, shift=0.2, ss=2)
triplet_solver.nroots = 1

singlet_solver = fci.direct_spin0.FCI(mol)
singlet_solver.spin = 0
singlet_solver.nroots = 2

weights = [0.1, 0.6, 0.3]
mc = mcpdft.CASSCF(mf, "tPBE", 2, 2, grids_level=1)
mc = mc.multi_state_mix(
    [triplet_solver, singlet_solver], weights, method="lin"
)
mc.optimize_mcscf_()

mc = LPDFSCF(mc).run()

print("L-PDFT energies [triplet, singlet 0, singlet 1]:", mc.e_states)
