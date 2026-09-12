#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from pyscf import lib
import numpy as np
from pyscf.grad import lpdft as lpdft_grad
from pyscf.grad import sacasscf as sacasscf_grad
from pyscf.df.grad import sacasscf as dfsacasscf_grad
from pyscf.df.grad import rhf as dfrhf_grad
from functools import partial


def _lpdft_df_response(mc, state, ci, lagrange_intermediates,
                       mf_grad, atmlst, auxbasis_response=True):
    '''Contract the L-PDFT Coulomb and SA Lagrange DF responses together.'''
    common, orbital_response, ci_response = lagrange_intermediates
    mo_cas = common['mo_cas']
    dm_core = common['dm_core']

    casdm1_state = mc.make_one_casdm1s(ci=ci, state=state)
    casdm1_state = casdm1_state[0] + casdm1_state[1]
    dm_cas_state = mo_cas @ casdm1_state @ mo_cas.T
    dm1_state = dm_core + dm_cas_state
    dm1_average = orbital_response['dm1']

    response_dms = (
        dm1_state,
        dm1_average,
        dm_core,
        orbital_response['dm_cas'],
        orbital_response['dmL_core'],
        orbital_response['dmL_cas'],
        ci_response['dm_cas'],
    )
    coulomb_pair_weights = np.zeros(
        (len(response_dms), len(response_dms)))
    exchange_pair_weights = np.zeros_like(coulomb_pair_weights)

    # J[D0,D] + J[D,D0] - J[D0,D0]
    coulomb_pair_weights[0,1] = 1
    coulomb_pair_weights[1,0] = 1
    coulomb_pair_weights[1,1] = -1

    lagrange_pairs = (
        (2,4), (4,2), (2,5), (4,3),
        (3,4), (5,2), (2,6), (6,2),
    )
    for i, j in lagrange_pairs:
        coulomb_pair_weights[i,j] += 1
        exchange_pair_weights[i,j] += 1

    casdm2_orb = (orbital_response['casdm2']
                  + orbital_response['casdm2'].transpose(1,0,3,2))
    df_ci, df_orb = dfsacasscf_grad.solve_df_rdm2(
        mc, mo_cas=mo_cas,
        casdm2=[ci_response['casdm2'], casdm2_orb])
    df_orb_internal_L = dfsacasscf_grad.solve_df_rdm2(
        mc, mo_cas=(mo_cas, orbital_response['moL_cas']),
        casdm2=casdm2_orb)[0]
    mo_df_pairs = (
        (mo_cas, mo_cas, df_ci + df_orb_internal_L),
        (mo_cas, orbital_response['moL_cas'], df_orb),
    )

    return dfsacasscf_grad._grad_elec_df_response_direct(
        mc, mf_grad, response_dms, coulomb_pair_weights, mo_df_pairs,
        atmlst, mc.max_memory, auxbasis_response=auxbasis_response,
        exchange_pair_weights=exchange_pair_weights)


def lpdft_nuc_response(mc_grad, Lvec, state=None, atmlst=None,
                       verbose=None, mo=None, ci=None, eris=None,
                       mf_grad=None, feff1=None, feff2=None, **kwargs):
    '''Combined DF-L-PDFT Hamiltonian, orbital, and CI nuclear response.'''
    if state is None:
        state = mc_grad.state
    if atmlst is None:
        atmlst = mc_grad.atmlst
    if atmlst is None:
        atmlst = list(range(mc_grad.mol.natm))
    if verbose is None:
        verbose = mc_grad.verbose
    if mo is None:
        mo = mc_grad.base.mo_coeff
    if ci is None:
        ci = mc_grad.base.ci
    if eris is None and mc_grad.eris is None:
        eris = mc_grad.eris = mc_grad.base.ao2mo(mo)
    elif eris is None:
        eris = mc_grad.eris
    if mf_grad is None:
        mf_grad = dfrhf_grad.Gradients(mc_grad.base._scf)
    if feff1 is None or feff2 is None:
        raise ValueError('feff1 and feff2 are required')

    Lorb, Lci = mc_grad.unpack_uniq_var(Lvec)
    lagrange_intermediates = sacasscf_grad.make_sa_lagrange_response_intermediates(
        Lorb, Lci, mc_grad.base, mo_coeff=mo, ci=ci, eris=eris)
    combined_df_response = _lpdft_df_response(
        mc_grad.base, state, ci, lagrange_intermediates, mf_grad, atmlst,
        auxbasis_response=mc_grad.auxbasis_response)
    return lpdft_grad.lpdft_HellmanFeynman_grad(
        mc_grad.base, mc_grad.base.otfnal, state, feff1, feff2,
        mo_coeff=mo, ci=ci, atmlst=atmlst, mf_grad=mf_grad,
        verbose=verbose, auxbasis_response=mc_grad.auxbasis_response,
        lagrange_intermediates=lagrange_intermediates,
        combined_df_response=combined_df_response)


# I need to resolve the __init__ and get_ham_response members. Otherwise everything should be fine!
class Gradients (dfsacasscf_grad.Gradients, lpdft_grad.Gradients):

    def __init__(self, pdft, state=None):
        self.auxbasis_response = True
        lpdft_grad.Gradients.__init__(self, pdft, state=state)

    def get_nuc_response(self, Lvec, **kwargs):
        return lpdft_nuc_response(self, Lvec, **kwargs)

    # TODO: rewrite the partialized fn to take the actual caller, use getattr,
    # and delete this
    def get_ham_response (self, **kwargs):
        pfn = partial (lpdft_grad.lpdft_HellmanFeynman_grad,
         auxbasis_response=self.auxbasis_response)
        with lib.temporary_env (lpdft_grad, lpdft_HellmanFeynman_grad=pfn):
            return lpdft_grad.Gradients.get_ham_response (self, **kwargs)

    kernel = lpdft_grad.Gradients.kernel
    get_wfn_response = lpdft_grad.Gradients.get_wfn_response
    get_init_guess = lpdft_grad.Gradients.get_init_guess
    get_otp_gradient_response = lpdft_grad.Gradients.get_otp_gradient_response
    get_Aop_Adiag = lpdft_grad.Gradients.get_Aop_Adiag
