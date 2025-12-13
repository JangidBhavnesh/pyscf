from pyscf import gto, scf, solvent, mcscf, lib, mcpdft
from pyscf.scf.hf import RHF
from pyscf.solvent import SMD

# Authors: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

'''
Example of using implicit solvent models with MC-PDFT calculations.
In this example, we have shown MC-PDFT computations with CASCI wavefunction
however CASSCF can also be used in a similar manner.
'''

mol = gto.Mole(atom = '''
    C        0.000000    0.000000    -0.542500
    O        0.000000    0.000000     0.677500
    H        0.000000    0.9353074   -1.082500
    H        0.000000   -0.9353074   -1.082500
    ''',
    basis = '6-31G',
    charge = 0,
    spin = 0,
    max_memory = 10000,
    verbose = lib.logger.INFO)
mol.build()

# Define the solvent model
smdsol = SMD(mol)
smdsol.solvent = 'methanol'

# SCF with SMD
mf = scf.RHF(mol)
mf = solvent.SMD(mf, smdsol)
mf.kernel()

# CASCI with SMD (Note this is self consistent reation field calculation, that means
# CASCI CI vectors are iteratively optimized in the presence of solvent field.
mc = mcscf.CASCI(mf, 4, 4)
mc = solvent.SMD(mc, smdsol) # Create the SMD CASCI object
mc.kernel()

# Now to run the MC-PDFT calculation with SMD, we need to pass the CI vectors without
# re-optimizing them and then add the solvent effects.

# Solvent effects: for the optimized density matrix from CASCI compute the solvent effects
esolv = smdsol.kernel(dm=mc.make_rdm1())[0]
ecds = smdsol.get_cds() # Only SMD requires this.
esolv_total = esolv + ecds

# Now run the MC-PDFT calculation without re-optimizing the CI vectors, works with
# all the on-top functionals.
my_pdft = mcpdft.CASCI(mc, 'tPBE', 4, 4)
e_pdft = my_pdft.compute_pdft_energy_(mo_coeff=mc.mo_coeff, ci=mc.ci, dump_chk=False)[0]

# Even for the hybrid functionals add the complete solvent effect to the final energy.
e_pdft_solv = e_pdft + esolv_total

print("MC-PDFT Energy without SMD: ", f"{e_pdft:.10f}", "Eh")
print("Solvent-solute Electrostatic Energy: ", f"{esolv:.10f}", "Eh")
print("Solvent-solute Cavity-Dispersion-Solvent Structure Energy: ", f"{ecds:.10f}", "Eh")
print("MC-PDFT Energy with SMD   : ", f"{e_pdft_solv:.10f}", "Eh")




