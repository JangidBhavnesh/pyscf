#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#

'''
parse CP2K PP format, following parse_nwchem.py
'''

import sys
import re
from pyscf.lib.exceptions import BasisNotFoundError
import numpy as np

def parse(string, symb=None):
    '''Parse the pseudo text *string* which is in CP2K format, return an internal
    basis format which can be assigned to :attr:`Cell.pseudo`
    Lines started with # are ignored.

    Args:
        string : Blank linke and the lines of "PSEUDOPOTENTIAL" and "END" will be ignored

    Examples:

    >>> cell = gto.Cell()
    >>> cell.pseudo = {'C': pyscf.gto.basis.pseudo_cp2k.parse("""
    ... #PSEUDOPOTENTIAL
    ... C GTH-BLYP-q4
    ...     2    2
    ...      0.33806609    2    -9.13626871     1.42925956
    ...     2
    ...      0.30232223    1     9.66551228
    ...      0.28637912    0
    ... """)}
    '''
    if symb is not None:
        raw_data = list(filter(None, re.split('#PSEUDOPOTENTIAL', string)))
        pseudotxt = _search_gthpp_block(raw_data, symb)
        if not pseudotxt:
            raise BasisNotFoundError(f'Pseudopotential not found for {symb}.')
    else:
        pseudotxt = [x.strip() for x in string.splitlines()
                     if x.strip() and 'END' not in x and '#PSEUDOPOTENTIAL' not in x]
    return _parse(pseudotxt)

def load(pseudofile, symb, suffix=None):
    '''Parse the *pseudofile's entry* for atom 'symb', return an internal
    pseudo format which can be assigned to :attr:`Cell.pseudo`
    '''
    return _parse(search_seg(pseudofile, symb, suffix))

def _parse(plines):
    header_ln = plines.pop(0)  # noqa: F841
    nelecs = [ int(nelec) for nelec in plines.pop(0).split() ]
    rnc_ppl = plines.pop(0).split()
    rloc = float(rnc_ppl[0])
    nexp = int(rnc_ppl[1])
    cexp = [ float(c) for c in rnc_ppl[2:] ]
    aline = plines.pop(0)
    SOCPP = "SOC" in aline
    nproj_types = int(re.search(r'\d+', aline).group() if SOCPP else int(aline))
    del aline
    r = []
    nproj = []
    hproj = []
    if SOCPP: kproj = []
    for p in range(nproj_types):
        l = len(r)
        rnh_ppnl = plines.pop(0).split()
        r.append(float(rnh_ppnl[0]))
        nproj.append(int(rnh_ppnl[1]))
        hproj_p_ij = []
        for h in rnh_ppnl[2:]:
            hproj_p_ij.append(float(h))
        for i in range(1,nproj[-1]):
            for h in plines.pop(0).split():
                hproj_p_ij.append(float(h))
        if l > 0:
            kproj_p_ij = []
            for i in range(nproj[-1]):
                for h in plines.pop(0).split():
                    kproj_p_ij.append(float(h))

            kproj_p = np.zeros((nproj[-1],nproj[-1]))
            kproj_p[np.triu_indices(nproj[-1])] = list(kproj_p_ij)
            kproj_p_symm = kproj_p + kproj_p.T - np.diag(kproj_p.diagonal())
            kproj.append(kproj_p_symm.tolist())

        hproj_p = np.zeros((nproj[-1],nproj[-1]))
        hproj_p[np.triu_indices(nproj[-1])] = list(hproj_p_ij)
        hproj_p_symm = hproj_p + hproj_p.T - np.diag(hproj_p.diagonal())
        hproj.append(hproj_p_symm.tolist())

    pseudo_params = [nelecs,
                     rloc, nexp, cexp,
                     nproj_types]
    for ri,ni,hi in zip(r,nproj,hproj):
        pseudo_params.append([ri, ni, hi])
    if SOCPP:
        pseudo_params.append(nproj_types-1)  # Number of SOC projectors
        for ri,ni,ki in zip(r[1:],nproj[1:],kproj):
            pseudo_params.append([ri, ni, ki])
    return pseudo_params

def search_seg(pseudofile, symb, suffix=None):
    '''
    Find the pseudopotential entry for atom 'symb' in file 'pseudofile'
    '''
    fin = open(pseudofile, 'r')
    fdata = fin.read().split('#PSEUDOPOTENTIAL')
    fin.close()
    dat = _search_gthpp_block(fdata[1:], symb, suffix)
    if not dat:
        raise BasisNotFoundError(f'Pseudopotential for {symb} in {pseudofile}')
    return dat

def _search_gthpp_block(raw_data, symb, suffix=None):
    for dat in raw_data:
        dat0 = dat.split(None, 1)
        if dat0 and dat0[0] == symb:
            dat = [x.strip() for x in dat.splitlines()
                   if x.strip() and 'END' not in x]
            if suffix is None:  # use default PP
                qsuffix = dat[0].split()[-1].split('-')[-1]
                if not (qsuffix.startswith('q') and qsuffix[1:].isdigit()):
                    return dat
            else:
                if any(suffix == x.split('-')[-1] for x in dat[0].split()):
                    return dat
    return None

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 2:
        ppfile = args[0]
        atom = args[1]
    else:
        print('usage: ppfile atomlabel ')
        sys.exit(1)

    print("Testing search_seg():")
    print(search_seg(ppfile,atom))

    print("Testing load() [[from file]]:")
    load(ppfile,atom)

    print("Testing parse():")
    print(parse("""
    #PSEUDOPOTENTIAL
    C GTH-BLYP-q4
        2    2
         0.33806609    2    -9.13626871     1.42925956
        2
         0.30232223    1     9.66551228
         0.28637912    0
    """))

    # Along with the scalar relativistic PP terms, the SOC terms are also parsed.
    print(parse("""
    C GTH-BLYP-q4 GTH-BLYP
        2    2
        0.33806609    2    -9.13626871     1.42925956
        2  SOC
        0.30232223    1     9.66551228
        0.28637912    1     0.00000000
                            0.00218693
    """))
