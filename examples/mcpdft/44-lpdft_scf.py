#!/usr/bin/env python
"""Fixed-orbital self-consistent L-PDFT with unequal state weights."""

from pyscf import gto, mcpdft, scf
from pyscf.mcpdft.lpdft_scf import LPDFSCF


mol = gto.M(
    atom="Li 0 0 0; H 1.5 0 0",
    basis="sto-3g",
    unit="angstrom",
    verbose=4,
)
mf = scf.RHF(mol).run()

# Construct L-PDFT with unequal weights, but run only the underlying
# SA-CASSCF optimization. LPDFSCF will perform the state-interaction step.
weights = [0.8, 0.2]
mc = mcpdft.CASSCF(mf, "tPBE", 2, 2, grids_level=1)
mc.fix_spin_(ss=0)
mc = mc.multi_state(weights, method="lin")
mc.optimize_mcscf_()

mc = LPDFSCF(mc).run()

print("L-PDFT energies:", mc.e_states)
