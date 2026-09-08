#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

import unittest

import numpy as np

from pyscf import dft, fci, gto, mcpdft, scf
from pyscf.mcpdft.lpdft_scf import LPDFSCF


class KnownValues(unittest.TestCase):
    def test_maximum_overlap_assignment(self):
        rotation = np.array([[0.1, 0.995], [0.995, -0.1]])
        order = LPDFSCF._maximum_overlap_order(rotation)
        np.testing.assert_array_equal(order, [1, 0])

    def test_unequal_weight_fixed_point(self):
        atom_specific_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False
        self.addCleanup(
            setattr,
            dft.radi,
            "ATOM_SPECIFIC_TREUTLER_GRIDS",
            atom_specific_grids,
        )

        mol = gto.M(
            atom="Li 0 0 0; H 1.5 0 0",
            basis="sto-3g",
            output="/dev/null",
            verbose=0,
        )
        mf = scf.RHF(mol).run()
        mc = mcpdft.CASSCF(mf, "ftLDA,VWN3", 2, 2, grids_level=1)
        mc.fix_spin_(ss=0)
        mc = mc.multi_state([0.8, 0.2], method="lin")
        mc.optimize_mcscf_()

        ls = LPDFSCF(
            mc,
            max_cycle=50,
            conv_tol=1e-8,
            conv_tol_rotation=1e-5,
            conv_tol_rdm=1e-6,
            damping=0.5,
        ).run()

        self.assertTrue(ls.converged)
        self.assertLess(ls.residual, ls.conv_tol_residual)
        self.assertAlmostEqual(ls.e_tot, ls.e_ensemble, 8)

        final_ham = mc.make_lpdft_ham_(ci=ls.ci)
        weight_difference = ls.weights[:, None] - ls.weights[None, :]
        final_residual = np.max(
            np.abs(np.triu(weight_difference * final_ham, k=1))
        )
        self.assertLess(final_residual, ls.conv_tol_residual)

    def test_unequal_weight_state_average_mix(self):
        atom_specific_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False
        self.addCleanup(
            setattr,
            dft.radi,
            "ATOM_SPECIFIC_TREUTLER_GRIDS",
            atom_specific_grids,
        )

        mol = gto.M(
            atom="Li 0 0 0; H 1.5 0 0",
            basis="sto-3g",
            output="/dev/null",
            verbose=0,
        )
        mf = scf.RHF(mol).run()

        triplet_solver = fci.direct_spin1.FCI(mol)
        triplet_solver.spin = 2
        triplet_solver = fci.addons.fix_spin(
            triplet_solver, shift=0.2, ss=2
        )
        triplet_solver.nroots = 1

        singlet_solver = fci.direct_spin0.FCI(mol)
        singlet_solver.spin = 0
        singlet_solver.nroots = 2

        mc = mcpdft.CASSCF(mf, "ftLDA,VWN3", 2, 2, grids_level=1)
        mc = mc.multi_state_mix(
            [triplet_solver, singlet_solver],
            [0.1, 0.6, 0.3],
            method="lin",
        )
        mc.optimize_mcscf_()

        ls = LPDFSCF(mc, damping=0.5).run()

        self.assertTrue(ls.converged)
        self.assertEqual([c.shape for c in ls.ci], [(1, 1), (2, 2), (2, 2)])
        self.assertEqual([s.stop - s.start for s in ls._state_slices], [1, 2])
        self.assertLess(ls.residual, ls.conv_tol_residual)
        self.assertAlmostEqual(ls.e_tot, ls.e_ensemble, 7)


if __name__ == "__main__":
    unittest.main()
