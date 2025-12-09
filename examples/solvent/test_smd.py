#!/usr/bin/env python
'''
An example of using SMD solvent models in the mean-field calculation and CASCI calculations.
'''

from pyscf import gto, scf, solvent, mcscf, lib

# In order to use the implicit solvent model for the CAS calculations,
# we need to create the solvent object for the MC-SCF object.

# Note: there are two ways to include the implicit solvent effect in the calculations:
# 1) Frozen: Use the mean-field density matrix to evaluate the solvent effect and add that to Hamiltonian
# 2) SCRF: Self-consistently include the solvent effect in the CAS calculations

# To check the accuracy of the SMD implementation for the CASCI calculations, I am performing the CAS calculation
# with active space (2e, 4o) such that the CAS energy should be equals to the SCF energy.

def get_mol(basis='6-31G', charge=0, spin=0, verbose=lib.logger.INFO, max_memory=120000):
    mol = gto.Mole()
    mol.atom = '''
        C        0.000000    0.000000             -0.542500
        O        0.000000    0.000000              0.677500
        H        0.000000    0.9353074360871938   -1.082500
        H        0.000000   -0.9353074360871938   -1.082500
            '''
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.max_memory = max_memory
    mol.verbose = verbose
    mol.output = 'smd_casci.log'
    return mol

def compare_mf_cas(mf, mc):
    e_scf = mf.e_tot
    e_cas = mc.e_tot
    print("SCF with SMD: ", e_scf)
    print("CAS with SMD: ", e_cas)
    print("Difference: ", f"{abs(e_scf - e_cas):.10f}")

test1 = False
test2 = False
test3 = True

# Test-1: SCRF Procedure
if test1:
    # Hartree-Fock with SMD (water)
    mol = get_mol()
    mf = scf.RHF(mol).SMD()
    mf.with_solvent.solvent = 'water'
    mf.kernel()

    mc = mcscf.CASCI(mf, 2, 4)
    mc = solvent.SMD(mc)
    e1 = mc.kernel()

    compare_mf_cas(mf, mc)
    del mf, mc

# Test-2: Frozen Density
if test2:
    # I can run the atomic HF and use that density to evaluate the solvent effect
    # for both MF and CAS calculations.
    mol = get_mol()

    from pyscf.scf.hf import RHF
    dm = RHF(mol).get_init_guess(mol, 'atom')
    
    mf = scf.RHF(mol).SMD(dm=dm)
    mf.with_solvent.solvent = 'water'
    mf.with_solvent.frozen = True
    mf.kernel()

    mc = mcscf.CASCI(mf, 2, 4)
    mc = solvent.SMD(mc, dm=dm)
    mc.with_solvent.frozen = True
    e1 = mc.kernel()

    compare_mf_cas(mf, mc)
    del mol, mf, mc

# Test-3: Different solvent and density_fitting
if test3:
    mol = get_mol()
    from pyscf.scf.hf import RHF
    dm = RHF(mol).get_init_guess(mol, 'atom')

    mf = scf.RHF(mol).density_fit()
    mf = solvent.SMD(mf, dm=dm)
    mf.with_solvent.solvent = 'methanol'
    mf.with_solvent.frozen = True
    mf.kernel()

    mc = mcscf.CASCI(mf, 2, 4)
    mc = solvent.SMD(mc, dm=dm)
    mc.with_solvent.solvent = 'methanol'
    mc.with_solvent.frozen = True
    e1 = mc.kernel()

    compare_mf_cas(mf, mc)
    del mf, mc
