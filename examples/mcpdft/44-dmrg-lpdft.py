#!/usr/bin/env python

"""Linearized pair-density functional theory with a DMRG solver.

This example requires the optional ``pyscf-dmrgscf`` extension and a working
Block or Block2 executable. Configure the executable and MPI launcher in
``pyscf.dmrgscf.settings`` before running the example.
"""

from pyscf import dmrgscf, gto, mcpdft, scf


# For example, these settings can also be configured in dmrgscf/settings.py:
# dmrgscf.settings.BLOCKEXE = "/path/to/block2main"
# dmrgscf.settings.MPIPREFIX = "mpirun -np 4"

mol = gto.M(
    atom="Li 0 0 0; H 0 0 3.0",
    basis="sto-3g",
    symmetry=False,
    verbose=4,
)
mf = scf.RHF(mol).run()

mc = mcpdft.CASSCF(mf, "tPBE", 4, 2)
mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=500, tol=1e-8)

# multi_state automatically selects the RDM-based DMRG-LPDFT implementation.
# DMRGCI supplies the diagonal and transition one- and two-particle RDMs used
# to construct the L-PDFT Hamiltonian.
weights = [0.5, 0.5]
lpdft = mc.multi_state(weights, method="lin").run()

print("State-average DMRG-SCF energies:", lpdft.e_mcscf)
print("DMRG-LPDFT energies:            ", lpdft.e_states)
print("DMRG-LPDFT Hamiltonian:\n", lpdft.lpdft_ham)
print("DMRG-LPDFT mixing coefficients:\n", lpdft.si_pdft)

# DMRG wavefunctions are stored externally and are identified by root number.
# Consequently, lpdft.ci retains the DMRG root IDs instead of rotated MPSs.
# Adiabatic MPS returns, nuclear gradients, and dipole moments are not yet
# implemented for DMRG-LPDFT.
print("DMRG reference-state IDs:", lpdft.ci)
