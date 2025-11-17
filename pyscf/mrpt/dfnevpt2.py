#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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
# Authors: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

import ctypes
import tempfile
import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def _cvcveris(mc, mo_coeff, with_df):
    '''
    Construction of cvcv ERIs required by NEVPT2 in MO basis using DF intermediates.

    Steps:
    1. Transform DF integrals to MO basis in blocks of auxiliary functions to get (L|pq)
    2. Using (L|pq), construct the cvcv intermediates required for NEVPT2.
    '''

    log = logger.Logger(mc.stdout, mc.verbose)

    nao, nmo = mo_coeff.shape
    ncore = mc.ncore
    ncas = mc.ncas
    nvir = nmo - ncore - ncas
    naoaux = with_df.get_naoaux()

    mem_now = lib.current_memory()[0]
    max_memory = max(4000, 0.9*mc.max_memory-mem_now)

    # Step-1: transform DF integrals to MO basis to get (L|pq)
    t1 = t0 = (logger.process_clock(), logger.perf_counter())

    mo = np.asarray(mo_coeff, order='F')

    fxpp = lib.H5TmpFile()

    blksize = max(4, int(min(with_df.blockdim, (max_memory*.95e6/8-naoaux*nmo*ncas)/3/nmo**2)))
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
        fxpp[str(k)] = bufpp.transpose(1,2,0)[:ncore, ncore+ncas:, ]
        b0 += naux

    bufs1 = bufpp = None
    t1 = log.timer('density fitting ao2mo step-1', *t0)

    # Step-2: from the transfomed (L|pq), build pacv and cvcv
    tmpdir = lib.param.TMPDIR
    cvcvfile = tempfile.NamedTemporaryFile(dir=tmpdir)
    # Edge cases
    if ncore * nvir == 0 or ncore * nvir == 0:
        f5 = lib.H5TmpFile(cvcvfile.name, 'w')
        cvcv = f5.create_dataset('eri_mo', (ncore*nvir,ncore*nvir), 'f8')
        cvcv[:,:] = 0
    else:
        mem_now = lib.current_memory()[0]
        nblk = int(max(8, min(nmo, (max_memory-mem_now)*1e6/8/(ncore*naoaux+nvir*ncore*nvir))))
        buf = np.empty((ncore, nvir, naoaux))
        bufs1 = np.empty((nblk,nvir,ncore*nvir))
        bufs2 = np.empty((nblk, nvir, naoaux))
        f5 = lib.H5TmpFile(cvcvfile.name, 'w')
        cvcv = f5.create_dataset('eri_mo', (ncore*nvir,ncore*nvir), 'f8')
        for p0, p1 in prange(0, ncore, nblk):
            nrow = p1 - p0
            tmp = bufs1[:nrow].reshape(nrow*nvir, -1)
            for key, col0, col1 in fxpp_keys:
                buf[:,:,col0:col1] = fxpp[key][p0:p1]

            bufs3 = buf.reshape(-1, naoaux).T
            bufs2 = buf[:nrow].reshape(nrow*nvir, naoaux)
            lib.dot(bufs2, bufs3, 1.0, tmp)

            r0 = p0*nvir
            r1 = min(p1, ncore)*nvir
            cvcv[r0:r1,:] = tmp

        bufs1 = bufs2 = bufs3 = buf = None
    t1 = log.timer('density fitting step-2', *t1)
    t0 = log.timer('density fitting cvcv', *t0)
    return cvcvfile

def _mem_usage(ncore, ncas, nmo):
    '''Estimate memory usage (in MB) for DF-NEVPT2 ERIs
        1. outcore memory for storing cvcv on disk
        2. incore memory for storing all ERIs in memory
    '''
    nvir = nmo - ncore - ncas
    papa = ppaa = nmo**2*ncas**2
    pacv = nmo*ncas*ncore*nvir
    cvcv = ncore*nvir*ncore*nvir
    outcore = (papa + ppaa + pacv) * 8/1e6
    incore = outcore + cvcv*8/1e6
    return incore, outcore

def _ERIS(mc, mo, with_df, method='incore'):
    ncore = mc.ncore
    ncas = mc.ncas
    nmo = mo.shape[1]
    nvir = nmo - ncore - ncas
    moa = mo[:, ncore:ncore+ncas]
    moc = mo[:, :ncore]
    mov = mo[:, ncore+ncas:]

    max_memory = max(4000, 0.9*mc.max_memory-lib.current_memory()[0])
    mem_incore, mem_outcore = _mem_usage(ncore, ncas, nmo)
    mem_now = lib.current_memory()[0]

    if with_df is not None and ((mem_incore+mem_now < mc.max_memory*.9) or mem_outcore < max_memory):
        papa = with_df.ao2mo([mo, moa, mo, moa], compact=False)
        ppaa = with_df.ao2mo([mo, mo, moa, moa], compact=False)
        pacv = with_df.ao2mo([mo, moa, moc, mov], compact=False)
        papa = papa.reshape(nmo, ncas, nmo, ncas)
        ppaa = ppaa.reshape(nmo, nmo, ncas, ncas)
        pacv = pacv.reshape(nmo, ncas, ncore, nvir)
    else:
        raise RuntimeError('DF-NEVPT2 ERIs cannot be constructed with the available memory %d MB'
                    % (max_memory))

    if (method == 'incore'):
        cvcv = with_df.ao2mo([moc, mov, moc, mov], compact=False)
    else:
        cvcv = _cvcveris(mc, mo, with_df)

    dmcore = np.dot(mo[:,:ncore], mo[:,:ncore].conj().T)
    vj, vk = mc._scf.get_jk(mc.mol, dmcore)
    vhfcore = reduce(np.dot, (mo.T, vj*2-vk, mo))
    h1eff = reduce(np.dot, (mo.conj().T, mc.get_hcore(), mo)) + vhfcore

    # Assemble a dictionary of ERIs
    eris = {}
    eris['vhf_c'] = vhfcore
    eris['ppaa'] = ppaa
    eris['papa'] = papa
    eris['pacv'] = pacv
    eris['cvcv'] = cvcv
    eris['h1eff'] = h1eff
    return eris
