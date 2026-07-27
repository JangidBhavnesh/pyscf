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


import ctypes
from functools import reduce
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf.mcscf.casci import CASCI
from pyscf.mcscf.ucasci import UCASCI
from pyscf.mcscf.umc1step import UCASSCF
from pyscf import df


def density_fit(casscf, auxbasis=None, with_df=None):
    '''Generate DF-CASSCF for given CASSCF object.  It is done by overwriting
    three CASSCF member functions:
        * casscf.ao2mo which generates MO integrals
        * casscf.get_veff which generate JK from core density matrix
        * casscf.get_jk which

    Args:
        casscf : an CASSCF object

    Kwargs:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.

    Returns:
        An CASSCF object with a modified J, K matrix constructor which uses density
        fitting integrals to compute J and K

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = DFCASSCF(mf, 4, 4)
    -100.05994191570504
    '''
    from pyscf.df.addons import predefined_auxbasis
    if with_df is None:
        if (getattr(casscf._scf, 'with_df', None) and
            (auxbasis is None or auxbasis == casscf._scf.with_df.auxbasis)):
            with_df = casscf._scf.with_df
        else:
            mol = casscf.mol
            if auxbasis is None and isinstance(mol.basis, str):
                auxbasis = predefined_auxbasis(mol, mol.basis, xc='HF')
            with_df = df.DF(mol, auxbasis)
            with_df.max_memory = casscf.max_memory
            with_df.stdout = casscf.stdout
            with_df.verbose = casscf.verbose

    if isinstance(casscf, _DFCAS):
        if casscf.with_df is None:
            casscf.with_df = with_df
        elif getattr(casscf.with_df, 'auxbasis', None) != auxbasis:
            #logger.warn(casscf, 'DF might have been initialized twice.')
            casscf = casscf.copy()
            casscf.with_df = with_df
        return casscf

    if isinstance(casscf, UCASCI):
        cls = _DFUCASCI
    elif isinstance(casscf, UCASSCF):
        cls = _DFUCASSCF
    elif isinstance(casscf, CASCI):
        cls = _DFCASCI
    else:
        cls = _DFCASSCF
    return lib.set_class(cls(casscf, with_df), (cls, casscf.__class__))

# A tag to label the derived MCSCF class
class _DFCAS:
    __name_mixin__ = 'DF'

    _keys = {'with_df'}

    def __init__(self, mc, with_df):
        self.__dict__.update(mc.__dict__)
        #self.grad_update_dep = 0
        self.with_df = with_df

    def undo_df(self):
        '''Remove the density fitting mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, _DFCAS))
        del obj.with_df
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'DFCASCI/DFCASSCF: density fitting for JK matrix '
                    'and 2e integral transformation')
        return self

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return super().reset(mol)

    # Modify get_veff for JK matrix of core density because get_h1eff calls
    # self.get_veff to generate core JK
    def get_veff(self, mol=None, dm=None, hermi=1):
        if dm is None:
            mocore = self.mo_coeff[:,:self.ncore]
            dm = np.dot(mocore, mocore.T) * 2
        vj, vk = self.get_jk(mol, dm, hermi)
        return vj - vk * .5

    # only approximate jk for self.update_jk_in_ah
    def get_jk(self, mol, dm, hermi=1, with_j=True, with_k=True, omega=None):
        if self.with_df:
            return self.with_df.get_jk(dm, hermi,
                                       with_j=with_j, with_k=with_k, omega=omega)
        else:
            return super().get_jk(mol, dm, hermi,
                                  with_j=with_j, with_k=with_k, omega=omega)

    def _exact_paaa(self, mo, u, out=None):
        if self.with_df:
            nmo = mo.shape[1]
            ncore = self.ncore
            ncas = self.ncas
            nocc = ncore + ncas
            mo1 = np.dot(mo, u)
            mo1_cas = mo1[:,ncore:nocc]
            paaa = self.with_df.ao2mo([mo1, mo1_cas, mo1_cas, mo1_cas], compact=False)
            return paaa.reshape(nmo,ncas,ncas,ncas)
        else:
            return super()._exact_paaa(mol, u, out)

    def nuc_grad_method(self):
        raise NotImplementedError

    def _state_average_nuc_grad_method (self, state=None):
        raise NotImplementedError

class _DFUCAS(_DFCAS):
    def dump_flags(self, verbose=None):
        super(_DFCAS, self).dump_flags(verbose)
        logger.info(self, 'DFUCASCI/DFUCASSCF: density fitting for JK matrix '
                    'and 2e integral transformation')
        return self

    def get_veff(self, mol=None, dm=None, hermi=1):
        if dm is None:
            mocore = (self.mo_coeff[0][:,:self.ncore[0]],
                      self.mo_coeff[1][:,:self.ncore[1]])
            dm = (np.dot(mocore[0], mocore[0].T),
                  np.dot(mocore[1], mocore[1].T))
        vj, vk = self.get_jk(mol, dm, hermi)
        return vj[0] + vj[1] - vk

    # _exact_paaa is only used by restricted CASSCF and is not called by
    # UCASCI or UCASSCF.
    def _exact_paaa(self, mo, u, out=None):
        raise NotImplementedError

class _DFCASCI(_DFCAS):
    def get_h2eff(self, mo_coeff=None):
        if self.with_df:
            ncore = self.ncore
            nocc = ncore + self.ncas
            if mo_coeff is None:
                mo_coeff = self.mo_coeff[:,ncore:nocc]
            elif mo_coeff.shape[1] != self.ncas:
                mo_coeff = mo_coeff[:,ncore:nocc]
            return self.with_df.ao2mo(mo_coeff)
        else:
            return super(_DFCAS, self).get_h2eff(mo_coeff)

class _DFUCASCI(_DFUCAS):
    def get_h2eff(self, mo_coeff=None):
        if self.with_df:
            ncore = self.ncore
            nocc = (ncore[0] + self.ncas, ncore[1] + self.ncas)
            if mo_coeff is None:
                mo_coeff = (self.mo_coeff[0][:,ncore[0]:nocc[0]],
                            self.mo_coeff[1][:,ncore[1]:nocc[1]])
            elif (mo_coeff[0].shape[1] != self.ncas or
                  mo_coeff[1].shape[1] != self.ncas):
                mo_coeff = (mo_coeff[0][:,ncore[0]:nocc[0]],
                            mo_coeff[1][:,ncore[1]:nocc[1]])

            ncas = self.ncas
            eri_aa = self.with_df.ao2mo(mo_coeff[0], compact=False)
            eri_ab = self.with_df.ao2mo((mo_coeff[0], mo_coeff[0],
                                        mo_coeff[1], mo_coeff[1]),
                                       compact=False)
            eri_bb = self.with_df.ao2mo(mo_coeff[1], compact=False)
            eri_aa = eri_aa.reshape(ncas,ncas,ncas,ncas)
            eri_ab = eri_ab.reshape(ncas,ncas,ncas,ncas)
            eri_bb = eri_bb.reshape(ncas,ncas,ncas,ncas)
            return (eri_aa, eri_ab, eri_bb)
        else:
            return super(_DFCAS, self).get_h2eff(mo_coeff)

class _DFCASSCF(_DFCAS):
    get_h2eff = _DFCASCI.get_h2eff

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if self.with_df:
            return _ERIS(self, mo_coeff, self.with_df)
        else:
            return super(_DFCAS, self).ao2mo(mo_coeff)

    def nuc_grad_method(self):
        from pyscf.df.grad import casscf
        return casscf.Gradients (self)

    def _state_average_nuc_grad_method (self, state=None):
        from pyscf.df.grad import sacasscf
        return sacasscf.Gradients (self, state=state)

class _DFUCASSCF(_DFUCAS):
    get_h2eff = _DFUCASCI.get_h2eff

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if self.with_df:
            return _DFUERIS(self, mo_coeff, self.with_df)
        else:
            return super(_DFCAS, self).ao2mo(mo_coeff)


def approx_hessian(casscf, auxbasis=None, with_df=None):
    '''Approximate the orbital hessian with density fitting integrals

    Note this function has no effects if the input casscf object is DF-CASSCF.
    It only modifies the orbital hessian of normal CASSCF object.

    Args:
        casscf : an CASSCF object

    Kwargs:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.
            The default basis 'weigend+etb' means weigend-coulomb-fit basis
            for light elements and even-tempered basis for heavy elements.

    Returns:
        A CASSCF object with approximated JK contraction for orbital hessian

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> mc = mcscf.approx_hessian(mcscf.CASSCF(mf, 4, 4))
    -100.06458716530391
    '''

    if isinstance(casscf, CASCI):
        return casscf  # because CASCI does not need orbital optimization

    if getattr(casscf, 'with_df', None):
        return casscf

    if with_df is None:
        if (getattr(casscf._scf, 'with_df', None) and
            (auxbasis is None or auxbasis == casscf._scf.with_df.auxbasis)):
            with_df = casscf._scf.with_df
        else:
            with_df = df.DF(casscf.mol)
            with_df.max_memory = casscf.max_memory
            with_df.stdout = casscf.stdout
            with_df.verbose = casscf.verbose
            if auxbasis is not None:
                with_df.auxbasis = auxbasis

    dfcas = _DFHessianCASSCF(casscf, with_df)
    return lib.set_class(dfcas, (_DFHessianCASSCF, casscf.__class__))

class _DFHessianCASSCF:

    __name_mixin__ = 'DFHessian'

    _keys = {'with_df'}

    def __init__(self, mc, with_df):
        self.__dict__.update(mc.__dict__)
        #self.grad_update_dep = 0
        self.with_df = with_df

    def undo_df(self):
        '''Remove the DFHessianCASSCF Mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, _DFHessianCASSCF))
        del obj.with_df
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.info(self, 'CASSCF: density fitting for orbital hessian')

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return super().reset(mol)

    def ao2mo(self, mo_coeff):
        # the exact integral transformation
        eris = super().ao2mo(mo_coeff)

        log = logger.Logger(self.stdout, self.verbose)
        # Add the approximate diagonal term for orbital hessian
        t1 = t0 = (logger.process_clock(), logger.perf_counter())
        mo = np.asarray(mo_coeff, order='F')
        nao, nmo = mo.shape
        ncore = self.ncore
        eris.j_pc = np.zeros((nmo,ncore))
        k_cp = np.zeros((ncore,nmo))
        fmmm = _ao2mo.libao2mo.AO2MOmmm_nr_s2_iltj
        fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
        ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2

        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = max(4, int(min(self.with_df.blockdim, max_memory*.3e6/8/nmo**2)))
        bufs1 = np.empty((blksize,nmo,nmo))
        for eri1 in self.with_df.loop(blksize):
            naux = eri1.shape[0]
            buf = bufs1[:naux]
            fdrv(ftrans, fmmm,
                 buf.ctypes.data_as(ctypes.c_void_p),
                 eri1.ctypes.data_as(ctypes.c_void_p),
                 mo.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), ctypes.c_int(nao),
                 (ctypes.c_int*4)(0, nmo, 0, nmo),
                 ctypes.c_void_p(0), ctypes.c_int(0))
            bufd = np.einsum('kii->ki', buf)
            eris.j_pc += np.einsum('ki,kj->ij', bufd, bufd[:,:ncore])
            k_cp += np.einsum('kij,kij->ij', buf[:,:ncore], buf[:,:ncore])
            t1 = log.timer_debug1('j_pc and k_pc', *t1)
        eris.k_pc = k_cp.T.copy()
        log.timer('ao2mo density fit part', *t0)
        return eris

    def get_jk(self, mol, dm, hermi=1, with_j=True, with_k=True, omega=None):
        if self.with_df:
            return self.with_df.get_jk(dm, hermi,
                                       with_j=with_j, with_k=with_k, omega=omega)
        else:
            return super().get_jk(mol, dm, hermi,
                                  with_j=with_j, with_k=with_k, omega=omega)

'''
Note: Currently, I am generating all the cvcv type of integrals required for
the orbital hessian. This is not memory efficient and can be improved in the future.
Like it is done in RCASSCF, the cvcv type integrals can be generated from respective
jk calls.
'''
class _DFUERIS:
    def __init__(self, casscf, mo, with_df):
        log = logger.Logger(casscf.stdout, casscf.verbose)

        nao, nmo = mo[0].shape
        ncore = self.ncore = casscf.ncore
        ncas = self.ncas = casscf.ncas
        nocc = (ncore[0] + ncas, ncore[1] + ncas)
        nvir = (nmo - ncore[0], nmo - ncore[1])

        # Memory estimation for DF-UCASSCF integral transformation
        mem_eris, mem_work = _mem_usage_uhf(ncore, ncas, nmo)
        mem_now = lib.current_memory()[0]
        max_memory = max(0, casscf.max_memory*.9-mem_now)
        nao_pair = nao * (nao + 1) // 2
        block_words = 2*nao_pair + 2*nmo**2 + 2*nmo
        mem_block = block_words * 8/1e6
        mem_required = mem_eris + mem_work + mem_block
        if mem_required > max_memory:
            raise RuntimeError(
                'Not enough memory for DF-UCASSCF integral transformation. '
                'Required %.3f MB, available %.3f MB'
                % (mem_required, max_memory))

        blksize = max(1, min(with_df.blockdim,
                            int((max_memory-mem_eris-mem_work)*1e6
                                / 8/block_words)))
        log.debug('DF-UCASSCF integral transformation needs %.3f MB '
                  'memory, block size %d', mem_required, blksize)

        self.jkcpp = np.zeros((ncore[0],nmo,nmo))
        self.jkcPP = np.zeros((ncore[1],nmo,nmo))
        self.jC_pp = np.zeros((nmo,nmo))
        self.jc_PP = np.zeros((nmo,nmo))

        self.aapp = np.zeros((ncas,ncas,nmo,nmo))
        self.aaPP = np.zeros((ncas,ncas,nmo,nmo))
        self.AApp = np.zeros((ncas,ncas,nmo,nmo))
        self.AAPP = np.zeros((ncas,ncas,nmo,nmo))
        self.appa = np.zeros((ncas,nmo,nmo,ncas))
        self.apPA = np.zeros((ncas,nmo,nmo,ncas))
        self.APPA = np.zeros((ncas,nmo,nmo,ncas))

        self.Iapcv = np.zeros((ncas,nmo,ncore[0],nvir[0]))
        self.IAPCV = np.zeros((ncas,nmo,ncore[1],nvir[1]))
        self.apCV = np.zeros((ncas,nmo,ncore[1],nvir[1]))
        self.APcv = np.zeros((ncas,nmo,ncore[0],nvir[0]))
        self.Icvcv = np.zeros((ncore[0],nvir[0],ncore[0],nvir[0]))
        self.ICVCV = np.zeros((ncore[1],nvir[1],ncore[1],nvir[1]))
        self.cvCV = np.zeros((ncore[0],nvir[0],ncore[1],nvir[1]))

        mo = (np.asarray(mo[0], order='F'),
              np.asarray(mo[1], order='F'))
        fmmm = _ao2mo.libao2mo.AO2MOmmm_nr_s2_iltj
        fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
        ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2

        bufs1 = np.empty((blksize,nmo,nmo))
        bufs2 = np.empty((blksize,nmo,nmo))

        t1 = t0 = (logger.process_clock(), logger.perf_counter())
        for eri1 in with_df.loop(blksize):
            naux = eri1.shape[0]
            bufpp = bufs1[:naux]
            bufPP = bufs2[:naux]
            fdrv(ftrans, fmmm,
                 bufpp.ctypes.data_as(ctypes.c_void_p),
                 eri1.ctypes.data_as(ctypes.c_void_p),
                 mo[0].ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), ctypes.c_int(nao),
                 (ctypes.c_int*4)(0, nmo, 0, nmo),
                 ctypes.c_void_p(0), ctypes.c_int(0))
            fdrv(ftrans, fmmm,
                 bufPP.ctypes.data_as(ctypes.c_void_p),
                 eri1.ctypes.data_as(ctypes.c_void_p),
                 mo[1].ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), ctypes.c_int(nao),
                 (ctypes.c_int*4)(0, nmo, 0, nmo),
                 ctypes.c_void_p(0), ctypes.c_int(0))

            bufd = np.einsum('kii->ki', bufpp)
            bufD = np.einsum('kii->ki', bufPP)
            self.jkcpp += np.einsum('ki,kpq->ipq',
                                       bufd[:,:ncore[0]], bufpp)
            self.jkcpp -= np.einsum('kip,kiq->ipq',
                                       bufpp[:,:ncore[0]],
                                       bufpp[:,:ncore[0]])
            self.jkcPP += np.einsum('ki,kpq->ipq',
                                       bufD[:,:ncore[1]], bufPP)
            self.jkcPP -= np.einsum('kip,kiq->ipq',
                                       bufPP[:,:ncore[1]],
                                       bufPP[:,:ncore[1]])
            self.jC_pp += np.einsum('ki,kpq->pq',
                                       bufD[:,:ncore[1]], bufpp)
            self.jc_PP += np.einsum('ki,kpq->pq',
                                       bufd[:,:ncore[0]], bufPP)

            bufaa = bufpp[:,ncore[0]:nocc[0],ncore[0]:nocc[0]]
            bufAA = bufPP[:,ncore[1]:nocc[1],ncore[1]:nocc[1]]
            self.aapp += np.einsum('kuv,kpq->uvpq', bufaa, bufpp)
            self.aaPP += np.einsum('kuv,kpq->uvpq', bufaa, bufPP)
            self.AApp += np.einsum('kuv,kpq->uvpq', bufAA, bufpp)
            self.AAPP += np.einsum('kuv,kpq->uvpq', bufAA, bufPP)

            bufap = bufpp[:,ncore[0]:nocc[0],:]
            bufpa = bufpp[:,:,ncore[0]:nocc[0]]
            bufAP = bufPP[:,ncore[1]:nocc[1],:]
            bufPA = bufPP[:,:,ncore[1]:nocc[1]]
            self.appa += np.einsum('kup,kqv->upqv', bufap, bufpa)
            self.apPA += np.einsum('kup,kqv->upqv', bufap, bufPA)
            self.APPA += np.einsum('kup,kqv->upqv', bufAP, bufPA)

            bufcv = bufpp[:,:ncore[0],ncore[0]:]
            bufCV = bufPP[:,:ncore[1],ncore[1]:]
            self.cvCV += np.einsum('kiv,kjw->ivjw', bufcv, bufCV)

            bufcc = bufpp[:,:ncore[0],:ncore[0]]
            bufvv = bufpp[:,ncore[0]:,ncore[0]:]
            self.Icvcv += np.einsum('kiv,kjw->ivjw',
                                        bufcv, bufcv) * 2
            self.Icvcv -= np.einsum('kij,kvw->ivjw',
                                        bufcc, bufvv)
            self.Icvcv -= np.einsum('kiw,kjv->ivjw',
                                        bufcv, bufcv)

            bufCC = bufPP[:,:ncore[1],:ncore[1]]
            bufVV = bufPP[:,ncore[1]:,ncore[1]:]
            self.ICVCV += np.einsum('kiv,kjw->ivjw',
                                        bufCV, bufCV) * 2
            self.ICVCV -= np.einsum('kij,kvw->ivjw',
                                        bufCC, bufVV)
            self.ICVCV -= np.einsum('kiw,kjv->ivjw',
                                        bufCV, bufCV)

            bufpv = bufpp[:,:,ncore[0]:]
            bufcu = bufpp[:,:ncore[0],ncore[0]:nocc[0]]
            bufcp = bufpp[:,:ncore[0],:]
            bufvu = bufpp[:,ncore[0]:,ncore[0]:nocc[0]]
            self.Iapcv += np.einsum('kup,kiv->upiv',
                                        bufap, bufcv) * 2
            self.Iapcv -= np.einsum('kpv,kiu->upiv',
                                        bufpv, bufcu)
            self.Iapcv -= np.einsum('kip,kvu->upiv',
                                        bufcp, bufvu)

            bufPV = bufPP[:,:,ncore[1]:]
            bufCU = bufPP[:,:ncore[1],ncore[1]:nocc[1]]
            bufCP = bufPP[:,:ncore[1],:]
            bufVU = bufPP[:,ncore[1]:,ncore[1]:nocc[1]]
            self.IAPCV += np.einsum('kup,kiv->upiv',
                                        bufAP, bufCV) * 2
            self.IAPCV -= np.einsum('kpv,kiu->upiv',
                                        bufPV, bufCU)
            self.IAPCV -= np.einsum('kip,kvu->upiv',
                                        bufCP, bufVU)

            self.apCV += np.einsum('kup,kiv->upiv', bufap, bufCV)
            self.APcv += np.einsum('kup,kiv->upiv', bufAP, bufcv)
            t1 = log.timer_debug1('DF-UCASSCF integral transformation', *t1)

        self.vhf_c = (np.einsum('ipq->pq', self.jkcpp) + self.jC_pp,
                      np.einsum('ipq->pq', self.jkcPP) + self.jc_PP)
        log.timer('DF-UCASSCF integral transformation', *t0)


class _ERIS:
    def __init__(self, casscf, mo, with_df):
        log = logger.Logger(casscf.stdout, casscf.verbose)

        mol = casscf.mol
        nao, nmo = mo.shape
        ncore = casscf.ncore
        ncas = casscf.ncas
        nocc = ncore + ncas
        naoaux = with_df.get_naoaux()

        mem_incore, mem_outcore, mem_basic = _mem_usage(ncore, ncas, nmo)
        mem_now = lib.current_memory()[0]
        max_memory = max(3000, casscf.max_memory*.9-mem_now)
        if max_memory < mem_basic:
            log.warn('Calculation needs %d MB memory, over CASSCF.max_memory (%d MB) limit',
                     (mem_basic+mem_now)/.9, casscf.max_memory)

        t1 = t0 = (logger.process_clock(), logger.perf_counter())
        self.feri = lib.H5TmpFile()
        self.ppaa = self.feri.create_dataset('ppaa', (nmo,nmo,ncas,ncas), 'f8')
        self.papa = self.feri.create_dataset('papa', (nmo,ncas,nmo,ncas), 'f8')
        self.j_pc = np.zeros((nmo,ncore))
        k_cp = np.zeros((ncore,nmo))

        mo = np.asarray(mo, order='F')
        fxpp = lib.H5TmpFile()

        blksize = max(4, int(min(with_df.blockdim, (max_memory*.95e6/8-naoaux*nmo*ncas)/3/nmo**2)))
        bufpa = np.empty((naoaux,nmo,ncas))
        bufs1 = np.empty((blksize,nmo,nmo))
        fmmm = _ao2mo.libao2mo.AO2MOmmm_nr_s2_iltj
        fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
        ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
        fxpp_keys = []
        b0 = 0
        for k, eri1 in enumerate(with_df.loop(blksize)):
            naux = eri1.shape[0]
            bufpp = bufs1[:naux]
            fdrv(ftrans, fmmm,
                 bufpp.ctypes.data_as(ctypes.c_void_p),
                 eri1.ctypes.data_as(ctypes.c_void_p),
                 mo.ctypes.data_as(ctypes.c_void_p),
                 ctypes.c_int(naux), ctypes.c_int(nao),
                 (ctypes.c_int*4)(0, nmo, 0, nmo),
                 ctypes.c_void_p(0), ctypes.c_int(0))
            fxpp_keys.append([str(k), b0, b0+naux])
            fxpp[str(k)] = bufpp.transpose(1,2,0)
            bufpa[b0:b0+naux] = bufpp[:,:,ncore:nocc]
            bufd = np.einsum('kii->ki', bufpp)
            self.j_pc += np.einsum('ki,kj->ij', bufd, bufd[:,:ncore])
            k_cp += np.einsum('kij,kij->ij', bufpp[:,:ncore], bufpp[:,:ncore])
            b0 += naux
            t1 = log.timer_debug1('j_pc and k_pc', *t1)
        self.k_pc = k_cp.T.copy()
        bufs1 = bufpp = None
        t1 = log.timer('density fitting ao2mo pass1', *t0)

        mem_now = lib.current_memory()[0]
        nblk = int(max(8, min(nmo, ((max_memory-mem_now)*1e6/8-bufpa.size)/(ncas**2*nmo))))
        bufs1 = np.empty((nblk,ncas,nmo,ncas))
        dgemm = lib.np_helper._dgemm
        for p0, p1 in prange(0, nmo, nblk):
            #tmp = np.dot(bufpa[:,p0:p1].reshape(naoaux,-1).T,
            #                bufpa.reshape(naoaux,-1))
            tmp = bufs1[:p1-p0]
            dgemm('T', 'N', (p1-p0)*ncas, nmo*ncas, naoaux,
                  bufpa.reshape(naoaux,-1), bufpa.reshape(naoaux,-1),
                  tmp.reshape(-1,nmo*ncas), 1, 0, p0*ncas, 0, 0)
            self.papa[p0:p1] = tmp.reshape(p1-p0,ncas,nmo,ncas)
        bufaa = bufpa[:,ncore:nocc,:].copy().reshape(-1,ncas**2)
        bufs1 = bufpa = None
        t1 = log.timer('density fitting papa pass2', *t1)

        mem_now = lib.current_memory()[0]
        nblk = int(max(8, min(nmo, (max_memory-mem_now)*1e6/8/(nmo*naoaux+ncas**2*nmo))))
        bufs1 = np.empty((nblk,nmo,naoaux))
        bufs2 = np.empty((nblk,nmo,ncas,ncas))
        for p0, p1 in prange(0, nmo, nblk):
            nrow = p1 - p0
            buf = bufs1[:nrow]
            tmp = bufs2[:nrow].reshape(-1,ncas**2)
            for key, col0, col1 in fxpp_keys:
                buf[:nrow,:,col0:col1] = fxpp[key][p0:p1]
            lib.dot(buf.reshape(-1,naoaux), bufaa, 1, tmp)
            self.ppaa[p0:p1] = tmp.reshape(p1-p0,nmo,ncas,ncas)
        bufs1 = bufs2 = buf = None
        t1 = log.timer('density fitting ppaa pass2', *t1)

        self.feri.flush()

        dm_core = np.dot(mo[:,:ncore], mo[:,:ncore].T)
        vj, vk = casscf.get_jk(mol, dm_core)
        self.vhf_c = reduce(np.dot, (mo.T, vj*2-vk, mo))
        t0 = log.timer('density fitting ao2mo', *t0)

def _mem_usage_uhf(ncore, ncas, nmo):
    ncore_a, ncore_b = ncore
    nvir_a = nmo - ncore_a
    nvir_b = nmo - ncore_b
    nmo2 = nmo**2

    mem_core = (ncore_a + ncore_b + 4) * nmo2
    mem_active = 7 * ncas**2 * nmo2
    mem_response = (2*ncas*nmo*ncore_a*nvir_a +
                    2*ncas*nmo*ncore_b*nvir_b +
                    ncore_a**2*nvir_a**2 +
                    ncore_b**2*nvir_b**2 +
                    ncore_a*nvir_a*ncore_b*nvir_b)
    mem_eris = (mem_core + mem_active + mem_response) * 8/1e6

    max_tensor = max(ncore_a*nmo2, ncore_b*nmo2, nmo2,
                     ncas**2*nmo2,
                     ncas*nmo*ncore_a*nvir_a,
                     ncas*nmo*ncore_b*nvir_b,
                     ncore_a**2*nvir_a**2,
                     ncore_b**2*nvir_b**2,
                     ncore_a*nvir_a*ncore_b*nvir_b)
    mem_work = max_tensor * 2 * 8/1e6
    return mem_eris, mem_work


def _mem_usage(ncore, ncas, nmo):
    outcore = basic = ncas**2*nmo**2*2 * 8/1e6
    incore = outcore + (ncore+ncas)*nmo**3*4/1e6
    return incore, outcore, basic

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import mcscf
    from pyscf.mcscf import addons

    mol = gto.Mole()
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()

    m = scf.RHF(mol)
    ehf = m.scf()
    mc = approx_hessian(mcscf.CASSCF(m, 6, 4))
    mc.verbose = 4
    mo = addons.sort_mo(mc, m.mo_coeff, (3,4,6,7,8,9), 1)
    emc = mc.kernel(mo)[0]
    print(ehf, emc, emc-ehf)
    #-76.0267656731 -76.0873922924 -0.0606266193028
    print(emc - -76.0873923174, emc - -76.0926176464)

    mc = approx_hessian(mcscf.CASSCF(m, 6, (3,1)))
    mc.verbose = 4
    emc = mc.mc2step(mo)[0]
    print(emc - -75.7155632535814)

    mf = scf.density_fit(m)
    mf.kernel()
    #mc = density_fit(mcscf.CASSCF(mf, 6, 4))
    #mc = mcscf.CASSCF(mf, 6, 4)
    mc = mcscf.DFCASSCF(mf, 6, 4)
    mc.verbose = 4
    mo = addons.sort_mo(mc, mc.mo_coeff, (3,4,6,7,8,9), 1)
    emc = mc.kernel(mo)[0]
    print(emc, 'ref = -76.0917567904955', emc - -76.0917567904955)
    mc.with_dep4 = True
    mc.max_cycle_micro = 10
    emc = mc.kernel(mo)[0]
    print(emc, 'ref = -76.0917567904955', emc - -76.0917567904955)

    #mc = density_fit(mcscf.CASCI(mf, 6, 4))
    #mc = mcscf.CASCI(mf, 6, 4)
    mc = mcscf.DFCASCI(mf, 6, 4)
    mo = addons.sort_mo(mc, mc.mo_coeff, (3,4,6,7,8,9), 1)
    emc = mc.kernel(mo)[0]
    print(emc, 'ref = -76.0476686258461', emc - -76.0476686258461)
