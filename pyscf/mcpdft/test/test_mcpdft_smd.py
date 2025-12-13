from pyscf import gto, scf, solvent, mcscf, lib, mcpdft
from pyscf.solvent import SMD


# Some tests for the SMD solvent model in MC-PDFT calculations.
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
    mol.output = '/dev/null'
    return mol

def compare_dft_pdft(e1, e2):
    assert abs(e1 - e2) < 1e-8, f"DFT and PDFT energies differ by {abs(e1 - e2)}"
    # print("DFT Energy: ", f"{e1:.10f}", "Eh")
    # print("PDFT Energy: ", f"{e2:.10f}", "Eh")
    # print("Difference: ", f"{(e1 - e2):.7e}")

mol = get_mol()
mol.build()

# To use the SMD solvent model, one should create a solvent object and
# pass it to the respective SCF/CAS object.
smdsol = SMD(mol)
smdsol.solvent = 'methanol'

test1 = 1
if test1:
    mol = get_mol()
    mol.build()

    mf = scf.RHF(mol)
    mf = solvent.SMD(mf, smdsol)
    mf.kernel()

    dm = mf.make_rdm1()

    e_ref = {}
    for func in ['LDA', 'PBE', 'M06L', 'PBE0']:
        dftref = scf.RKS(mol)
        dftref.xc = func.lower()
        dftref.max_cycle = 0
        dftref = solvent.SMD(dftref, smdsol, dm=dm)
        dftref.kernel(dm0=dm)

        e_ref[func] = dftref.e_tot

    mc = mcscf.CASCI(mf, 2, 4)
    mc = solvent.SMD(mc, smdsol)
    mc.kernel()

    # To compute the PDFT energy with solvent, do not re reun the CASCI calculation, just use
    # the density matrix from the CASCI calculation with solvent effects included. Additionally
    # the solvent-solute interaction energy should be evaluated with the CAS density.
    esolv = smdsol.kernel(dm = mc.make_rdm1())[0]
    ecds = smdsol.get_cds()
    esmd = esolv + ecds

    e_pdft = {}
    for func in ['LDA', 'PBE', 'M06L']:
        otfunc = 't' + func
        mcnew = mcpdft.CASCI(mc, otfunc, 2, 4)
        e_pdft_ = mcnew.compute_pdft_energy_(mo_coeff=mc.mo_coeff, ci=mc.ci, dump_chk=False)[0]
        # Even for the hybrid functionals, the solvent effect is added in full, because the
        # in the MC-PDFT energy evaluation, the CAS contribution does not include any solvent effects.
        e_pdft_ += esmd

        e_pdft[func] = e_pdft_
        compare_dft_pdft(e_ref[func], e_pdft[func])
    del mf, mc

test2=1
# Hybrid functionals with lambda = 1, should match exactly CASCI energy
if test2:
    mol = get_mol()
    mol.build()

    mf = scf.RHF(mol)
    mf = solvent.SMD(mf, smdsol)
    mf.kernel()

    mc = mcscf.CASCI(mf, 2, 4)
    mc = solvent.SMD(mc, smdsol)
    mc.kernel()

    # To compute the PDFT energy with solvent, do not re reun the CASCI calculation, just use
    # the density matrix from the CASCI calculation with solvent effects included. Additionally
    # the solvent-solute interaction energy should be evaluated with the CAS density.
    esolv = smdsol.kernel(dm = mc.make_rdm1())[0]
    ecds = smdsol.get_cds()
    esmd = esolv + ecds

    # Construct the otfunc with lambda = 1.0: which should be equals to CASCI energy
    tCASCI = 't' + mcpdft.hyb('PBE',1.0, hyb_type='average')
    mcnew = mcpdft.CASCI(mc, tCASCI, 2, 4)
    e_pdft_ = mcnew.compute_pdft_energy_(mo_coeff=mc.mo_coeff, ci=mc.ci, dump_chk=False)[0]
    e_pdft_ += esmd

    compare_dft_pdft(e_pdft_, mc.e_tot)
    del mf, mc

# For sanity check, repeat this with CASSCF
test3 = 1
if test3:
    mol = get_mol()
    mol.build()

    mf = scf.RHF(mol)
    mf = solvent.SMD(mf, smdsol)
    mf.kernel()

    dm = mf.make_rdm1()

    e_ref = {}
    for func in ['LDA', 'PBE', 'M06L', 'PBE0']:
        dftref = scf.RKS(mol)
        dftref.xc = func.lower()
        dftref.max_cycle = 0
        dftref = solvent.SMD(dftref, smdsol, dm=dm)
        dftref.kernel(dm0=dm)

        e_ref[func] = dftref.e_tot

    mc = mcscf.CASSCF(mf, 2, 4)
    mc = solvent.SMD(mc, smdsol)
    mc.kernel()

    # To compute the PDFT energy with solvent, do not re reun the CASCI calculation, just use
    # the density matrix from the CASCI calculation with solvent effects included. Additionally
    # the solvent-solute interaction energy should be evaluated with the CAS density.
    esolv = smdsol.kernel(dm = mc.make_rdm1())[0]
    ecds = smdsol.get_cds()
    esmd = esolv + ecds

    e_pdft = {}
    for func in ['LDA', 'PBE', 'M06L']:
        otfunc = 't' + func
        mcnew = mcpdft.CASCI(mc, otfunc, 2, 4)
        e_pdft_ = mcnew.compute_pdft_energy_(mo_coeff=mc.mo_coeff, ci=mc.ci, dump_chk=False)[0]
        # Even for the hybrid functionals, the solvent effect is added in full, because the
        # in the MC-PDFT energy evaluation, the CAS contribution does not include any solvent effects.
        e_pdft_ += esmd

        e_pdft[func] = e_pdft_

        compare_dft_pdft(e_ref[func], e_pdft[func])
    del mf, mc
