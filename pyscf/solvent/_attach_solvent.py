#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Attach ddCOSMO to SCF, MCSCF, and post-SCF methods.
'''

import copy
import numpy
from pyscf import lib
from pyscf.lib import logger
from functools import reduce
from pyscf import scf

def _for_scf(mf, solvent_obj, dm=None):
    '''Add solvent model to SCF (HF and DFT) method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mf, _Solvation):
        mf.with_solvent = solvent_obj
        return mf

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_mf = SCFWithSolvent(mf, solvent_obj)
    name = solvent_obj.__class__.__name__ + mf.__class__.__name__
    return lib.set_class(sol_mf, (SCFWithSolvent, mf.__class__), name)

# 1. A tag to label the derived method class
class _Solvation:
    pass

class SCFWithSolvent(_Solvation):
    _keys = {'with_solvent'}

    def __init__(self, mf, solvent):
        self.__dict__.update(mf.__dict__)
        self.with_solvent = solvent

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, SCFWithSolvent, name_mixin))
        del obj.with_solvent
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    # Note v_solvent should not be added to get_hcore for scf methods.
    # get_hcore is overloaded by many post-HF methods. Modifying
    # SCF.get_hcore may lead error.

    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        # FIXME: super() here and after might be problematic and need to be
        # fixed in the future. Consider the combination of solvent and QM/MM.
        # Strictly, vhf = self.undo_solvent().get_veff()
        vhf = super().get_veff(mol, dm, *args, **kwargs)
        with_solvent = self.with_solvent
        if not with_solvent.frozen:
            with_solvent.e, with_solvent.v = with_solvent.kernel(dm)
        e_solvent, v_solvent = with_solvent.e, with_solvent.v

        # NOTE: v_solvent should not be added to vhf in this place. This is
        # because vhf is used as the reference for direct_scf in the next
        # iteration. If v_solvent is added here, it may break direct SCF.
        return lib.tag_array(vhf, e_solvent=e_solvent, v_solvent=v_solvent)

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                 diis=None, diis_start_cycle=None,
                 level_shift_factor=None, damp_factor=None, fock_last=None):
        if dm is None: dm = self.make_rdm1()

        # DIIS was called inside super().get_fock. v_solvent, as a function of
        # dm, should be extrapolated as well. To enable it, v_solvent has to be
        # added to the fock matrix before DIIS was called.
        if getattr(vhf, 'v_solvent', None) is None:
            vhf = self.get_veff(self.mol, dm)
        return super().get_fock(h1e, s1e, vhf+vhf.v_solvent, dm, cycle, diis,
                                diis_start_cycle, level_shift_factor, damp_factor,
                                fock_last)

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        if getattr(vhf, 'e_solvent', None) is None:
            vhf = self.get_veff(self.mol, dm)

        e_tot, e_coul = super().energy_elec(dm, h1e, vhf)
        e_solvent = vhf.e_solvent
        e_tot += e_solvent
        self.scf_summary['e_solvent'] = vhf.e_solvent.real

        if (hasattr(self.with_solvent, 'method') and
            self.with_solvent.method.upper() == 'SMD'):
            if self.with_solvent.e_cds is None:
                e_cds = self.with_solvent.get_cds()
                self.with_solvent.e_cds = e_cds
            else:
                e_cds = self.with_solvent.e_cds

            if isinstance(e_cds, numpy.ndarray):
                e_cds = e_cds[0]
            e_tot += e_cds
            self.scf_summary['e_cds'] = e_cds
            logger.info(self, f'CDS correction = {e_cds:.15f}')
        logger.info(self, 'Solvent Energy = %.15g', vhf.e_solvent)
        return e_tot, e_coul

    def nuc_grad_method(self):
        from pyscf.solvent.grad.pcm import make_grad_object
        # FIXME: when applying DF after solvent:
        #    mf = mol.RKS().PCM().density_fit().run()
        # The df.grad.rhf.Gradients.kernel is called. The
        # grad.pcm.WithSolventGrad.kernel is not executed.
        return make_grad_object(self)

    Gradients = nuc_grad_method

    def Hessian(self):
        from pyscf.solvent.hessian.pcm import make_hess_object
        return make_hess_object(self)

    def gen_response(self, *args, **kwargs):
        # The response function consists of two parts: the gas-phase and the
        # solvent response. The "vind" computes the gas-phase response.
        # The attribute .equilibrium_solvation controls whether to add the
        # solvents response.
        #
        # "equilibrium_solvation=True" corresponds to a slow process where the
        # solvents are fully relaxed wrt the first order electron density.
        # Vertical excitations in TDDFT are typically a fast process where the
        # solvation is non-equilibrium. The solvent does not fully relax against
        # the first order density, (corresponding to equilibrium_solvation=False).
        #
        # TDDFT are separately handled in the TDSCFWithSolvent class. This
        # response function handles all other response calculations (such as
        # stability analysis, SOSCF, polarizability and Hessian).
        vind = self.undo_solvent().gen_response(*args, **kwargs)
        is_uhf = isinstance(self, scf.uhf.UHF)
        def vind_with_solvent(dm1):
            v = vind(dm1)
            if self.with_solvent.equilibrium_solvation:
                if is_uhf:
                    v += self.with_solvent._B_dot_x(dm1[0]+dm1[1])
                else:
                    v += self.with_solvent._B_dot_x(dm1)
            return v
        return vind_with_solvent

    def stability(self, *args, **kwargs):
        # When computing orbital hessian, the second order derivatives of
        # solvent energy needs to be computed. It is enabled by
        # the attribute equilibrium_solvation in gen_response method.
        # If solvent was frozen, its contribution is treated as the
        # external potential. The response of solvent does not need to
        # be considered in stability analysis.
        with lib.temporary_env(self.with_solvent,
                               equilibrium_solvation=not self.with_solvent.frozen):
            return super().stability(*args, **kwargs)

    def to_gpu(self):
        from gpu4pyscf.solvent import _attach_solvent # type: ignore
        solvent_obj = self.with_solvent.to_gpu()
        obj = _attach_solvent._for_scf(self.undo_solvent().to_gpu(), solvent_obj)
        return obj

    def TDA(self, equilibrium_solvation=False):
        return _for_tdscf(super().TDA(), equilibrium_solvation=equilibrium_solvation)

    def TDHF(self, equilibrium_solvation=False):
        return _for_tdscf(super().TDHF(), equilibrium_solvation=equilibrium_solvation)

    CasidaTDDFT = NotImplemented

    def TDDFT(self, equilibrium_solvation=False):
        return _for_tdscf(super().TDDFT(), equilibrium_solvation=equilibrium_solvation)

    def MP2(self):
        solvent_model = _dispatch_solvent_model(self.with_solvent)
        # Note the super().MP2 might actually point to the DFMP2
        return solvent_model(super().MP2())

    def CISD(self):
        solvent_model = _dispatch_solvent_model(self.with_solvent)
        return solvent_model(super().CISD())

    def CCSD(self):
        solvent_model = _dispatch_solvent_model(self.with_solvent)
        # Note the super().CCSD might actually point to the DFCCSD
        return solvent_model(super().CCSD())

    def CASCI(self, ncas, nelecas, **kwargs):
        solvent_model = _dispatch_solvent_model(self.with_solvent)
        return solvent_model(super().CASCI(ncas, nelecas, **kwargs))

    def CASSCF(self, ncas, nelecas, **kwargs):
        solvent_model = _dispatch_solvent_model(self.with_solvent)
        return solvent_model(super().CASSCF(ncas, nelecas, **kwargs))

def _dispatch_solvent_model(solvent_obj):
    from pyscf import solvent
    solvent_name = solvent_obj.__class__.__name__
    if solvent_name in ('PCM', 'ddCOSMO', 'ddPCM', 'SMD'):
        return getattr(solvent, solvent_name)
    if solvent_name == 'PolEmbed':
        return solvent.PE
    raise RuntimeError(f'Unknown solvent model {solvent}')

def _for_casscf(mc, solvent_obj, dm=None):
    '''Add solvent model to CASSCF method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mc, _Solvation):
        mc.with_solvent = solvent_obj
        return mc

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_cas = CASSCFWithSolvent(mc, solvent_obj)
    name = solvent_obj.__class__.__name__ + mc.__class__.__name__
    return lib.set_class(sol_cas, (CASSCFWithSolvent, mc.__class__), name)

class CASSCFWithSolvent(_Solvation):
    _keys = {'with_solvent'}

    def __init__(self, mc, solvent):
        self.__dict__.update(mc.__dict__)
        self.with_solvent = solvent
        self._e_tot_without_solvent = 0

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, CASSCFWithSolvent, name_mixin))
        del obj.with_solvent
        del obj._e_tot_without_solvent
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        if self.conv_tol < 1e-7:
            logger.warn(self, 'CASSCF+ddCOSMO may not be able to '
                        'converge to conv_tol=%g', self.conv_tol)

        if (getattr(self._scf, 'with_solvent', None) and
            not getattr(self, 'with_solvent', None)):
            logger.warn(self, '''Solvent model %s was found in SCF object.
COSMO is not applied to the CASCI object. The CASSCF result is not affected by the SCF solvent model.
To enable the solvent model for CASSCF, a decoration to CASSCF object as below needs to be called
    from pyscf import solvent
    mc = mcscf.CASSCF(...)
    mc = solvent.ddCOSMO(mc)
''',
                        self._scf.with_solvent.__class__)
        return self

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    def update_casdm(self, mo, u, fcivec, e_ci, eris, envs={}):
        casdm1, casdm2, gci, fcivec = \
                super().update_casdm(mo, u, fcivec, e_ci, eris, envs)

# The potential is generated based on the density of current micro iteration.
# It will be added to hcore in casci function. Strictly speaking, this density
# is not the same to the CASSCF density (which was used to measure
# convergence) in the macro iterations.  When CASSCF is converged, it
# should be almost the same to the CASSCF density of the last macro iteration.
        with_solvent = self.with_solvent
        if not with_solvent.frozen:
            # Code to mimic dm = self.make_rdm1(ci=fcivec)
            mocore = mo[:,:self.ncore]
            mocas = mo[:,self.ncore:self.ncore+self.ncas]
            dm = reduce(numpy.dot, (mocas, casdm1, mocas.T))
            dm += numpy.dot(mocore, mocore.T) * 2
            with_solvent.e, with_solvent.v = with_solvent.kernel(dm)

        return casdm1, casdm2, gci, fcivec

# ddCOSMO Potential should be added to the effective potential. However, there
# is no hook to modify the effective potential in CASSCF. The workaround
# here is to modify hcore. It can affect the 1-electron operator in many CASSCF
# functions: gen_h_op, update_casdm, casci.  Note hcore is used to compute the
# energy for core density (Ecore).  The resultant total energy from casci
# function will include the contribution from ddCOSMO potential. The
# duplicated energy contribution from solvent needs to be removed.
    def get_hcore(self, mol=None):
        hcore = self._scf.get_hcore(mol)
        if self.with_solvent.v is not None:
            hcore += self.with_solvent.v
        return hcore

    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        log = logger.new_logger(self, verbose)
        log.debug('Running CASCI with solvent. Note the total energy '
                  'has duplicated contributions from solvent.')

        # In super().casci function, dE was computed based on the total
        # energy without removing the duplicated solvent contributions.
        # However, envs['elast'] is the last total energy with correct
        # solvent effects. Hack envs['elast'] to make super().casci print
        # the correct energy difference.
        envs['elast'] = self._e_tot_without_solvent
        e_tot, e_cas, fcivec = super().casci(mo_coeff, ci0, eris,
                                             verbose, envs)
        self._e_tot_without_solvent = e_tot

        log.debug('Computing corrections to the total energy.')
        dm = self.make_rdm1(ci=fcivec, ao_repr=True)

        with_solvent = self.with_solvent
        if with_solvent.e is not None:
            edup = numpy.einsum('ij,ji->', with_solvent.v, dm)
            e_tot = e_tot - edup + with_solvent.e
            log.info('Removing duplication %.15g, '
                     'adding E(solvent) = %.15g to total energy:\n'
                     '    E(CASSCF+solvent) = %.15g', edup, with_solvent.e, e_tot)

        # Update solvent effects for next iteration if needed
        if not with_solvent.frozen:
            with_solvent.e, with_solvent.v = with_solvent.kernel(dm)

        return e_tot, e_cas, fcivec

    def nuc_grad_method(self):
        logger.warn(self, '''
The code for CASSCF gradients was based on variational CASSCF wavefunction.
However, the ddCOSMO-CASSCF energy was not computed variationally.
Approximate gradients are evaluated here. A small error may be expected in the
gradients which corresponds to the contribution of
MCSCF_DM * V_solvent[d/dX MCSCF_DM] + V_solvent[MCSCF_DM] * d/dX MCSCF_DM
''')
        from pyscf.solvent.grad.pcm import make_grad_object
        return make_grad_object(self)

    Gradients = nuc_grad_method

    def to_gpu(self):
        obj = self.undo_solvent().to_gpu()
        obj = _for_casscf(obj, self.with_solvent)
        return lib.to_gpu(self, obj)


def _for_casci(mc, solvent_obj, dm=None):
    '''Add solvent model to CASCI method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mc, _Solvation):
        mc.with_solvent = solvent_obj
        return mc

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_mc = CASCIWithSolvent(mc, solvent_obj)
    name = solvent_obj.__class__.__name__ + mc.__class__.__name__
    return lib.set_class(sol_mc, (CASCIWithSolvent, mc.__class__), name)

class CASCIWithSolvent(_Solvation):
    _keys = {'with_solvent'}

    def __init__(self, mc, solvent):
        self.__dict__.update(mc.__dict__)
        self.with_solvent = solvent

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, CASCIWithSolvent, name_mixin))
        del obj.with_solvent
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    def get_hcore(self, mol=None):
        hcore = self._scf.get_hcore(mol)
        if self.with_solvent.v is not None:
            # NOTE: get_hcore was called by CASCI to generate core
            # potential.  v_solvent is added in this place to take accounts the
            # effects of solvent. Its contribution is duplicated and it
            # should be removed from the total energy.
            hcore += self.with_solvent.v
        return hcore

    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        with_solvent = self.with_solvent

        log = logger.new_logger(self)
        log.info('\n** Self-consistently update the solvent effects for %s **',
                 self.__class__.__name__)
        log1 = copy.copy(log)
        log1.verbose -= 1  # Suppress a few output messages

        mc_base_kernel = super().kernel
        def casci_iter_(ci0, log):
            # self.e_tot, self.e_cas, and self.ci are updated in the call
            # to super().kernel
            e_tot, e_cas, ci0 = mc_base_kernel(mo_coeff, ci0, log)[:3]

            if isinstance(self.e_cas, (float, numpy.number)):
                dm = self.make_rdm1(ci=ci0)
            else:
                log.debug('Computing solvent responses to DM of state %d',
                          with_solvent.state_id)
                dm = self.make_rdm1(ci=ci0[with_solvent.state_id])

            if with_solvent.e is not None:
                edup = numpy.einsum('ij,ji->', with_solvent.v, dm)
                self.e_tot += with_solvent.e - edup

            if not with_solvent.frozen:
                with_solvent.e, with_solvent.v = with_solvent.kernel(dm)
            return self.e_tot, e_cas, ci0

        if with_solvent.frozen:
            with lib.temporary_env(self, _finalize=lambda:None):
                casci_iter_(ci0, log)
            log.note('Total energy with solvent effects')
            self._finalize()
            return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

        self.converged = False
        with lib.temporary_env(self, canonicalization=False):
            e_tot = e_last = 0
            for cycle in range(self.with_solvent.max_cycle):
                log.info('\n** Solvent self-consistent cycle %d:', cycle)
                e_tot, e_cas, ci0 = casci_iter_(ci0, log1)

                de = e_tot - e_last
                if isinstance(e_cas, (float, numpy.number)):
                    log.info('Solvent cycle %d  E(CASCI+solvent) = %.15g  '
                             'dE = %g', cycle, e_tot, de)
                else:
                    for i, e in enumerate(e_tot):
                        log.info('Solvent cycle %d  CASCI root %d  '
                                 'E(CASCI+solvent) = %.15g  dE = %g',
                                 cycle, i, e, de[i])

                if abs(e_tot-e_last).max() < with_solvent.conv_tol:
                    self.converged = True
                    break
                e_last = e_tot

        # An extra cycle to canonicalize CASCI orbitals
        with lib.temporary_env(self, _finalize=lambda:None):
            casci_iter_(ci0, log)
        if self.converged:
            log.info('self-consistent CASCI+solvent converged')
        else:
            log.info('self-consistent CASCI+solvent not converged')
        log.note('Total energy with solvent effects')
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def nuc_grad_method(self):
        logger.warn(self, '''
The code for CASCI gradients was based on variational CASCI wavefunction.
However, the ddCOSMO-CASCI energy was not computed variationally.
Approximate gradients are evaluated here. A small error may be expected in the
gradients which corresponds to the contribution of
MCSCF_DM * V_solvent[d/dX MCSCF_DM] + V_solvent[MCSCF_DM] * d/dX MCSCF_DM
''')
        from pyscf.solvent.grad.pcm import make_grad_object
        return make_grad_object(self)

    Gradients = nuc_grad_method

    def to_gpu(self):
        obj = self.undo_solvent().to_gpu()
        obj = _for_casci(obj, self.with_solvent)
        return lib.to_gpu(self, obj)


def _for_post_scf(method, solvent_obj, dm=None):
    '''A wrapper of solvent model for post-SCF methods (CC, CI, MP etc.)

    NOTE: this implementation often causes (macro iteration) convergence issue

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(method, _Solvation):
        method.with_solvent = solvent_obj
        method._scf.with_solvent = solvent_obj
        return method

    # Ensure that the underlying _scf object has solvent model enabled
    if getattr(method._scf, 'with_solvent', None):
        scf_with_solvent = method._scf
    else:
        scf_with_solvent = _for_scf(method._scf, solvent_obj, dm)
        if dm is None:
            solvent_obj = scf_with_solvent.with_solvent
            solvent_obj.e, solvent_obj.v = \
                    solvent_obj.kernel(scf_with_solvent.make_rdm1())

    if dm is not None:
        solvent_obj = scf_with_solvent.with_solvent
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    postmf = PostSCFWithSolvent(method, scf_with_solvent)
    name = solvent_obj.__class__.__name__ + method.__class__.__name__
    return lib.set_class(postmf, (PostSCFWithSolvent, method.__class__), name)

class PostSCFWithSolvent(_Solvation):
    def __init__(self, method, scf_with_solvent):
        self.__dict__.update(method.__dict__)
        self._scf = scf_with_solvent
        # Post-HF objects access the solvent effects indirectly through the
        # underlying ._scf object.
        self._basic_scanner = method.as_scanner()
        self._basic_scanner._scf = scf_with_solvent.as_scanner()

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self._scf.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, PostSCFWithSolvent, name_mixin))
        obj._scf = self._scf.undo_solvent()
        del obj._basic_scanner
        return obj

    @property
    def with_solvent(self):
        return self._scf.with_solvent

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    def kernel(self, *args, **kwargs):
        with_solvent = self.with_solvent
        # The underlying ._scf object is decorated with solvent effects.
        # The resultant Fock matrix and orbital energies both include the
        # effects from solvent. It means that solvent effects for post-HF
        # methods are automatically counted if solvent is enabled at scf
        # level.
        if with_solvent.frozen:
            return super().kernel(*args, **kwargs)

        log = logger.new_logger(self)
        log.info('\n** Self-consistently update the solvent effects for %s **',
                 self.__class__.__name__)
        ##TODO: Suppress a few output messages
        #log1 = copy.copy(log)
        #log1.note, log1.info = log1.info, log1.debug

        basic_scanner = self._basic_scanner
        e_last = 0
        #diis = lib.diis.DIIS()
        for cycle in range(self.with_solvent.max_cycle):
            log.info('\n** Solvent self-consistent cycle %d:', cycle)
            # Solvent effects are applied when accessing the
            # underlying ._scf objects. The flag frozen=True ensures that
            # the generated potential with_solvent.v is passed to the
            # the post-HF object, without being updated in the implicit
            # call to the _scf iterations.
            with lib.temporary_env(with_solvent, frozen=True):
                e_tot = basic_scanner(self.mol)
                dm = basic_scanner.make_rdm1(ao_repr=True)
                #dm = diis.update(dm)

            # To generate the solvent potential for ._scf object. Since
            # frozen is set when calling basic_scanner, the solvent
            # effects are frozen during the scf iterations.
            with_solvent.e, with_solvent.v = with_solvent.kernel(dm)

            de = e_tot - e_last
            log.info('Solvent cycle %d  E_tot = %.15g  dE = %g',
                     cycle, e_tot, de)

            if abs(e_tot-e_last).max() < with_solvent.conv_tol:
                break
            e_last = e_tot

        # An extra cycle to compute the total energy
        log.info('\n** Extra cycle for solvent effects')
        with lib.temporary_env(with_solvent, frozen=True):
            #Update everything except the _scf object and _keys
            basic_scanner(self.mol)
            self.__dict__.update(basic_scanner.__dict__)
            self._scf.__dict__.update(basic_scanner._scf.__dict__)
        self._finalize()
        return self.e_corr, None

    def nuc_grad_method(self):
        logger.warn(self, '''
Approximate gradients are evaluated here. A small error may be expected in the
gradients which corresponds to the contribution of
DM * V_solvent[d/dX DM] + V_solvent[DM] * d/dX DM
''')
        from pyscf.solvent.grad.pcm import make_grad_object
        return make_grad_object(self)

    Gradients = nuc_grad_method

    def to_gpu(self):
        obj = self.undo_solvent().to_gpu()
        obj = _for_post_scf(obj, self.with_solvent)
        return lib.to_gpu(self, obj)


def _for_tdscf(method, solvent_obj=None, dm=None, equilibrium_solvation=False):
    '''Add solvent model in TDDFT calculations.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    assert hasattr(method._scf, 'with_solvent')
    if method._scf.with_solvent.__class__.__name__ == 'PolEmbed':
        # PolEmbed is currently not compatible with the implicit solvent implementation
        from pyscf.solvent.pol_embed import pe_for_tdscf
        return pe_for_tdscf(method, solvent_obj, dm, equilibrium_solvation)

    if solvent_obj is None:
        if isinstance(method, _Solvation):
            return method

        solvent_obj = method._scf.with_solvent.copy()
        solvent_obj.equilibrium_solvation = equilibrium_solvation
        if not equilibrium_solvation:
            # The vertical excitation is a fast process, applying non-equilibrium
            # solvation with optical dielectric constant eps=1.78
            # TODO: reset() can be skipped. Most intermeidates can be reused.
            solvent_obj.reset()
            solvent_obj.eps = 1.78
            solvent_obj.build()

    if isinstance(method, _Solvation):
        method = method.copy()
        method.with_solvent = solvent_obj
        return method

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True
        if solvent_obj.equilibrium_solvation:
            raise RuntimeError(
                '"frozen" solvent model conflicts to the assumption of equilibrium solvation.')

    sol_td = TDSCFWithSolvent(method, solvent_obj)
    name = solvent_obj.__class__.__name__ + method.__class__.__name__
    return lib.set_class(sol_td, (TDSCFWithSolvent, method.__class__), name)

class TDSCFWithSolvent(_Solvation):
    '''LR Solvent for TDDFT.

    Note: This class does not support the state-specific excited state solvent.
    '''

    _keys = {'with_solvent'}

    def __init__(self, method, solvent_obj):
        self.__dict__.update(method.__dict__)
        self.with_solvent = solvent_obj

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, TDSCFWithSolvent, name_mixin))
        obj._scf = self._scf.undo_solvent()
        return obj

    @property
    def equilibrium_solvation(self):
        '''Whether to allow the solvent rapidly responds to the changes of
        electronic structure or geometry of solute.
        '''
        return self.with_solvent.equilibrium_solvation
    @equilibrium_solvation.setter
    def equilibrium_solvation(self, val):
        if val and self.with_solvent.frozen:
            raise RuntimeError(
                '"frozen" Solvent model was set in the '
                'ground state SCF calculation. It conflicts to '
                'the assumption of equilibrium solvation.\n'
                'You can set _scf.with_solvent.frozen = False and '
                'rerun the ground state calculation _scf.run().')
        self.with_solvent.equilibrium_solvation = val

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('Solvent model for TDDFT:')
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    def gen_response(self, *args, **kwargs):
        # vind computes the response in gas-phase
        vind = self._scf.undo_solvent().gen_response(
            *args, with_nlc=not self.exclude_nlc, **kwargs)

        # The contribution of the solvent to an excited state include the fast
        # and the slow response parts. In the process of fast vertical excitation,
        # only the fast part is able to respond to changes of the solute
        # wavefunction. This process is described by the non-equilibrium
        # solvation. In the excited Hamiltonian, the potential from the slow part is
        # omitted. Changes of the solute electron density would lead to a
        # redistribution of the surface charge (due to the fast part).
        # The redistributed surface charge is computed by solving
        #     K^{-1} R (dm_response)
        # using a different dielectric constant. The optical dielectric constant
        # (eps=1.78, see QChem manual) is a suitable choice for the excited state.
        if not self.with_solvent.equilibrium_solvation:
            # Solvent with optical dielectric constant, for evaluating the
            # response of the fast solvent part
            with_solvent = self.with_solvent
            logger.info(self, 'TDDFT non-equilibrium solvation with eps=%g', with_solvent.eps)
        else:
            # Solvent with zero-frequency dielectric constant. The ground state
            # solvent is utilized to ensure the same eps are used in the
            # gradients of excited state.
            with_solvent = self._scf.with_solvent
            logger.info(self, 'TDDFT equilibrium solvation with eps=%g', with_solvent.eps)

        is_uhf = isinstance(self._scf, scf.uhf.UHF)
        singlet = kwargs.get('singlet', True)
        singlet = singlet or singlet is None
        def vind_with_solvent(dm1):
            v = vind(dm1)
            if is_uhf:
                v += with_solvent._B_dot_x(dm1[0]+dm1[1])
            elif singlet:
                v += with_solvent._B_dot_x(dm1)
            else:
                # The triplet excitation does not change the total electron
                # density, thus does not lead to solvent response.
                pass
            return v
        return vind_with_solvent

    def get_ab(self, mf=None):
        raise NotImplementedError

    def nuc_grad_method(self):
        from pyscf.solvent.pcm import PCM
        from pyscf.solvent.grad.ddcosmo_tdscf_grad import make_grad_object
        if isinstance(self.with_solvent, PCM):
            raise NotImplementedError('PCM-TDDFT Gradients')
        return make_grad_object(self)

    Gradients = nuc_grad_method

    def to_gpu(self):
        obj = self.undo_solvent().to_gpu()
        obj = _for_tdscf(obj, self.with_solvent)
        return lib.to_gpu(self, obj)
