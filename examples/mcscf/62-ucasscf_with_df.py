#!/usr/bin/env python

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.ed>

import time

from pyscf import gto, scf, mcscf

'''
Compare UCASSCF with and without density fitting (DF)
'''

mol = gto.Mole()
mol.atom = [
    ["O", (0., 0.,  0.7)],
    ["O", (0., 0., -0.7)],]
mol.basis = 'cc-pvdz'
mol.spin = 2
mol.build()

mf = scf.UHF(mol)
print('E(UHF) = %.15g' % mf.kernel())

t0 = time.perf_counter()
mc = mcscf.UCASSCF(mf, 4, (4,2))
emc = mc.kernel()[0]
t_ucasscf = time.perf_counter() - t0

t0 = time.perf_counter()
mc_df = mcscf.UCASSCF(mf, 4, (4,2)).density_fit()
emc_df = mc_df.kernel()[0]
t_dfucasscf = time.perf_counter() - t0

print('\nUCASSCF comparison')
print('Without DF: E = %.15g  time = %.2f sec' % (emc, t_ucasscf))
print('With DF:    E = %.15g  time = %.2f sec' % (emc_df, t_dfucasscf))
print('Energy difference = %.6g' % abs(emc_df - emc))
print('Speedup = %.2f' % (t_ucasscf / t_dfucasscf))
