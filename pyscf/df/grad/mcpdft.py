#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
from pyscf.grad import mcpdft as mcpdft_grad
from pyscf.grad import sacasscf as sacasscf_grad
from pyscf.df.grad import sacasscf as dfsacasscf_grad
from pyscf.df.grad import rhf as dfrhf_grad
from functools import partial


def _mcpdft_df_response(mc, fcasscf, ci_state, lagrange_intermediates,
                        mf_grad, atmlst, ot_hyb,
                        auxbasis_response=True):
    '''Contract the MC-PDFT Coulomb and SA Lagrange DF responses together.'''
    common, orbital_response, ci_response = lagrange_intermediates
    mo_cas = common['mo_cas']
    dm_core = common['dm_core']

    casdm1_state = fcasscf.fcisolver.make_rdm1(
        ci_state, common['ncas'], fcasscf.nelecas)
    dm_cas_state = mo_cas @ casdm1_state @ mo_cas.T
    dm1_state = dm_core + dm_cas_state

    response_dms = (
        dm1_state,
        dm_core,
        orbital_response['dm_cas'],
        orbital_response['dmL_core'],
        orbital_response['dmL_cas'],
        ci_response['dm_cas'],
    )
    coulomb_pair_weights = np.zeros(
        (len(response_dms), len(response_dms)))
    exchange_pair_weights = np.zeros_like(coulomb_pair_weights)

    # MC-PDFT retains only the classical Coulomb part of the wave-function
    # two-electron Hamiltonian, scaled by the non-hybrid fraction.
    coulomb_pair_weights[0,0] = ot_hyb

    # The SA-CASSCF constraints retain their usual J - K/2 response.
    lagrange_pairs = (
        (1,3), (3,1), (1,4), (3,2),
        (2,3), (4,1), (1,5), (5,1),
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


def mcpdft_nuc_response(mc_grad, Lvec, state=None, atmlst=None,
                        verbose=None, mo=None, ci=None, eris=None,
                        mf_grad=None, veff1=None, veff2=None, **kwargs):
    '''Combined DF-MC-PDFT Hamiltonian, orbital, and CI nuclear response.'''
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
    if veff1 is None or veff2 is None:
        raise ValueError('veff1 and veff2 are required')

    Lorb, Lci = mc_grad.unpack_uniq_var(Lvec)
    lagrange_intermediates = sacasscf_grad.make_sa_lagrange_response_intermediates(
        Lorb, Lci, mc_grad.base, mo_coeff=mo, ci=ci, eris=eris)

    fcasscf = mc_grad.make_fcasscf(state)
    fcasscf.mo_coeff = mo
    fcasscf.ci = ci[state]

    spin = abs(fcasscf.nelecas[0] - fcasscf.nelecas[1])
    omega, _, hyb = mc_grad.base.otfnal._numint.rsh_and_hybrid_coeff(
        mc_grad.base.otxc, spin=spin)
    if abs(omega) > 1e-11:
        raise NotImplementedError('range-separated on-top functionals')
    if abs(hyb[0] - hyb[1]) > 1e-11:
        raise NotImplementedError(
            'hybrid on-top functionals with different exchange,correlation components')
    ot_hyb = 1.0 - hyb[0]

    combined_df_response = _mcpdft_df_response(
        mc_grad.base, fcasscf, ci[state], lagrange_intermediates,
        mf_grad, atmlst, ot_hyb,
        auxbasis_response=mc_grad.auxbasis_response)
    return mcpdft_grad.mcpdft_HellmanFeynman_grad(
        fcasscf, mc_grad.base.otfnal, veff1, veff2,
        mo_coeff=mo, ci=ci[state], atmlst=atmlst, mf_grad=mf_grad,
        verbose=verbose, auxbasis_response=mc_grad.auxbasis_response,
        lagrange_intermediates=lagrange_intermediates,
        combined_df_response=combined_df_response)


# I need to resolve the __init__ and get_ham_response members. Otherwise everything should be fine!
class Gradients (dfsacasscf_grad.Gradients, mcpdft_grad.Gradients):

    def __init__(self, pdft, state=None):
        self.auxbasis_response = True
        mcpdft_grad.Gradients.__init__(self, pdft, state=state)

    def get_nuc_response(self, Lvec, **kwargs):
        return mcpdft_nuc_response(self, Lvec, **kwargs)

    # TODO: rewrite the partialized fn to take the actual caller, use getattr,
    # and delete this
    def get_ham_response (self, **kwargs):
        pfn = partial (mcpdft_grad.mcpdft_HellmanFeynman_grad,
         auxbasis_response=self.auxbasis_response)
        with lib.temporary_env (mcpdft_grad, mcpdft_HellmanFeynman_grad=pfn):
            return mcpdft_grad.Gradients.get_ham_response (self, **kwargs)

    kernel = mcpdft_grad.Gradients.kernel
    get_wfn_response = mcpdft_grad.Gradients.get_wfn_response
    get_init_guess = mcpdft_grad.Gradients.get_init_guess
    project_Aop = mcpdft_grad.Gradients.project_Aop
