#!/usr/bin/env python

'''
PBC-SOC integrals with GTH-SOC-PP
'''

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.gto.pseudo.pp_int import get_gth_pp_so as get_gth_pp_so_pbc

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-soc-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.build()


# Obtain the SOC integrals from GTH-SOC-PP
hso = -0.5*get_gth_pp_so_pbc(cell)

# With open bounday conditions,
mol = cell.to_mol()
mol.build(False, False)

from pyscf.gto.pp_int import get_gth_pp_so
hso_1c = -0.5*get_gth_pp_so(mol)
