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

"""Self-consistent state interaction for L-PDFT.

This module optimizes rotations among a fixed set of L-PDFT model-space
states.  Molecular orbitals are not optimized.  The implementation is
initially restricted to pure on-top functionals. Conventional state-average
and ``state_average_mix`` references are supported.

For an ensemble with weights ``w_I``, stationarity with respect to a rotation
between model-space states I and J requires

    (w_I - w_J) H_L-PDFT[I,J] = 0.

This condition is automatic for equal weights.  With unequal weights, the
weighted density is rebuilt after each state-interaction rotation until the
condition is satisfied self-consistently.
"""

import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment

from pyscf.lib import logger
from pyscf.mcpdft import lpdft


class LPDFSCF:
    """Fixed-orbital, self-consistent L-PDFT state-interaction driver.

    Args:
        mc : instance of :class:`pyscf.mcpdft.lpdft._LPDFT`
            An L-PDFT object whose underlying SA-CASSCF calculation has
            already been run with ``mc.optimize_mcscf_()``.

    Kwargs:
        max_cycle : int
            Maximum number of state-interaction iterations.
        conv_tol : float
            Convergence threshold for the ensemble-energy change.
        conv_tol_rotation : float
            Convergence threshold for the state-rotation step.
        conv_tol_rdm : float
            Convergence threshold for changes in the weighted 1- and 2-RDMs.
        conv_tol_residual : float or None
            Convergence threshold for
            ``(w_I-w_J) * H_L-PDFT[I,J]``. When None, use
            ``conv_tol_rotation``.
        damping : float
            Fraction of the diagonalizing state rotation applied per cycle.
            Must be in the interval ``(0, 1]``.
        root_tracking : {"overlap", "energy"}
            With ``"overlap"``, use maximum-overlap matching so that weights
            follow state character between iterations. With ``"energy"``,
            assign roots by ascending energy within each solver block.
        verbose : int
            PySCF verbosity. Defaults to ``mc.verbose``.

    Notes:
        This is an experimental fixed-orbital prototype.  It is not the
        standard one-shot L-PDFT method.  Hybrid and range-separated on-top
        functionals are intentionally rejected.
    """

    def __init__(
        self,
        mc,
        max_cycle=50,
        conv_tol=1e-8,
        conv_tol_rotation=1e-5,
        conv_tol_rdm=1e-6,
        conv_tol_residual=None,
        damping=0.5,
        root_tracking="overlap",
        verbose=None,
    ):
        self.base = mc
        self.max_cycle = max_cycle
        self.conv_tol = conv_tol
        self.conv_tol_rotation = conv_tol_rotation
        self.conv_tol_rdm = conv_tol_rdm
        if conv_tol_residual is None:
            conv_tol_residual = conv_tol_rotation
        self.conv_tol_residual = conv_tol_residual
        self.damping = damping
        self.root_tracking = root_tracking
        self.verbose = mc.verbose if verbose is None else verbose

        self.converged = False
        self.niter = 0
        self.e_states = None
        self.e_tot = None
        self.e_ensemble = None
        self.ci = None
        self.lpdft_ham = None
        self.residual = None
        self.root_assignment = None
        self.history = []

        self._validate_input()
        self._state_slices = self._get_state_slices()

    def _validate_input(self):
        mc = self.base
        if not isinstance(mc, lpdft._LPDFT):
            raise TypeError("LPDFSCF requires an L-PDFT object")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must satisfy 0 < damping <= 1")
        if self.root_tracking not in ("overlap", "energy"):
            raise ValueError("root_tracking must be 'overlap' or 'energy'")
        if self.max_cycle < 1:
            raise ValueError("max_cycle must be a positive integer")

        weights = np.asarray(mc.weights, dtype=float)
        if weights.ndim != 1 or len(weights) < 2:
            raise ValueError("LPDFSCF requires at least two state weights")
        if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("state weights must be nonnegative and sum to one")

        spin = abs(mc.nelecas[0] - mc.nelecas[1])
        omega, _, hyb = mc.otfnal._numint.rsh_and_hybrid_coeff(
            mc.otfnal.otxc, spin=spin
        )
        if abs(omega) > 1e-11 or np.max(np.abs(hyb)) > 1e-11:
            raise NotImplementedError(
                "The initial LPDFSCF prototype supports only pure, "
                "non-range-separated on-top functionals"
            )

    def _get_state_slices(self):
        if not isinstance(self.base, lpdft._LPDFTMix):
            return [slice(0, len(self.weights))]

        state_slices = []
        start = 0
        for solver in self.base.fcisolver.fcisolvers:
            stop = start + solver.nroots
            state_slices.append(slice(start, stop))
            start = stop
        if start != len(self.weights):
            raise ValueError(
                "The state-average-mix solver roots do not match the weights"
            )
        return state_slices

    @property
    def weights(self):
        return np.asarray(self.base.weights, dtype=float)

    @staticmethod
    def _copy_ci(ci):
        return [np.array(c, copy=True) for c in ci]

    def _ensemble_rdms(self, ci):
        casdm1s, casdm2 = self.base.get_casdm12_0(
            ci=ci, weights=self.weights
        )
        return np.asarray(casdm1s), np.asarray(casdm2)

    def _ensemble_energy(self, casdm1s, casdm2):
        mc = self.base
        e_mcwfn = mc.energy_mcwfn(
            mo_coeff=mc.mo_coeff,
            casdm1s=casdm1s,
            casdm2=casdm2,
            verbose=self.verbose,
        )
        e_ot = mc.energy_dft(
            mo_coeff=mc.mo_coeff,
            casdm1s=casdm1s,
            casdm2=casdm2,
        )
        return float(np.real(e_mcwfn + e_ot))

    def _make_hamiltonian(self, ci):
        ham = self.base.make_lpdft_ham_(ci=ci)
        if isinstance(ham, (list, tuple)):
            ham = linalg.block_diag(*ham)
        ham = np.asarray(ham)
        return (ham + ham.conj().T) * 0.5

    @staticmethod
    def _maximum_overlap_order(rotation):
        """Assign eigenvector columns to current roots by maximum overlap."""
        rows, columns = linear_sum_assignment(-np.abs(rotation))
        order = np.empty(rotation.shape[0], dtype=int)
        order[rows] = columns
        return order

    def _diagonalize(self, ham):
        """Diagonalize only within state-interaction-compatible blocks."""
        energies = np.empty(len(self.weights))
        rotation = np.eye(len(self.weights), dtype=ham.dtype)
        root_assignment = np.arange(len(self.weights))
        for state_slice in self._state_slices:
            block = ham[state_slice, state_slice]
            block_energies, block_rotation = linalg.eigh(block)
            if self.root_tracking == "overlap":
                order = self._maximum_overlap_order(block_rotation)
                block_energies = block_energies[order]
                block_rotation = block_rotation[:, order]
                root_assignment[state_slice] = state_slice.start + order
            energies[state_slice] = block_energies
            rotation[state_slice, state_slice] = block_rotation
        return energies, rotation, root_assignment

    def _state_rotation_residual(self, ham):
        weight_difference = self.weights[:, None] - self.weights[None, :]
        residual = weight_difference * ham
        return float(np.max(np.abs(np.triu(residual, k=1))))

    @staticmethod
    def _align_eigenvector_phases(rotation):
        """Choose deterministic column phases before damping a rotation."""
        rotation = np.array(rotation, copy=True)
        for i in range(rotation.shape[1]):
            if abs(rotation[i, i]) > 1e-12:
                phase = rotation[i, i] / abs(rotation[i, i])
            else:
                j = np.argmax(np.abs(rotation[:, i]))
                phase = rotation[j, i] / abs(rotation[j, i])
            rotation[:, i] /= phase
        return rotation

    def _damped_rotation(self, rotation):
        rotation = self._align_eigenvector_phases(rotation)
        if self.damping == 1.0:
            return rotation

        # Project the linearly damped matrix onto the nearest unitary matrix.
        trial = (
            (1.0 - self.damping) * np.eye(rotation.shape[0], dtype=rotation.dtype)
            + self.damping * rotation
        )
        left, _, right_h = linalg.svd(trial)
        return left @ right_h

    def _rotate_ci(self, ci, rotation):
        rotated_ci = []
        for state_slice in self._state_slices:
            ci_block = np.asarray(ci[state_slice])
            rotation_block = rotation[state_slice, state_slice]
            rotated_ci.extend(
                np.tensordot(rotation_block.conj().T, ci_block, axes=1)
            )
        return rotated_ci

    def _canonicalize_equal_weight_blocks(self, ci, ham):
        """Diagonalize H within blocks whose rotations preserve the ensemble."""
        rotation = np.eye(len(self.weights), dtype=ham.dtype)
        for state_slice in self._state_slices:
            unused = set(range(state_slice.start, state_slice.stop))
            while unused:
                i = unused.pop()
                block = [i]
                for j in list(unused):
                    if np.isclose(
                        self.weights[i], self.weights[j], rtol=0.0, atol=1e-12
                    ):
                        block.append(j)
                        unused.remove(j)
                if len(block) > 1:
                    block = np.asarray(sorted(block))
                    _, block_rotation = linalg.eigh(ham[np.ix_(block, block)])
                    if self.root_tracking == "overlap":
                        order = self._maximum_overlap_order(block_rotation)
                        block_rotation = block_rotation[:, order]
                    rotation[np.ix_(block, block)] = block_rotation

        ci = self._rotate_ci(ci, rotation)
        ham = rotation.conj().T @ ham @ rotation
        return ci, ham

    def kernel_self_consistent(self, ci=None):
        """Iterate the unequal-weight L-PDFT model-space states.

        Args:
            ci : list of ndarrays, optional
                Initial orthonormal model-space CI vectors. Defaults to
                ``mc.ci``.

        Returns:
            e_states : ndarray
                Diagonal L-PDFT energies in the converged state basis.
            ci : list of ndarrays
                Self-consistent model-space CI vectors.
        """
        mc = self.base
        log = logger.new_logger(self, self.verbose)
        if ci is None:
            ci = mc.ci
        if ci is None or len(ci) != len(self.weights):
            raise ValueError(
                "Run mc.optimize_mcscf_() first or supply one CI vector per weight"
            )

        current_ci = self._copy_ci(ci)
        previous_energy = None
        previous_dm1s = None
        previous_dm2 = None

        self.converged = False
        self.history = []

        for cycle in range(self.max_cycle):
            casdm1s, casdm2 = self._ensemble_rdms(current_ci)
            ensemble_energy = self._ensemble_energy(casdm1s, casdm2)
            ham = self._make_hamiltonian(current_ci)
            residual = self._state_rotation_residual(ham)
            _, rotation, root_assignment = self._diagonalize(ham)
            rotation = self._damped_rotation(rotation)
            delta_rotation = np.linalg.norm(
                rotation - np.eye(rotation.shape[0], dtype=rotation.dtype)
            )

            if previous_energy is None:
                delta_energy = np.inf
                norm_ddm = np.inf
            else:
                delta_energy = abs(ensemble_energy - previous_energy)
                norm_ddm = max(
                    np.linalg.norm(casdm1s - previous_dm1s),
                    np.linalg.norm(casdm2 - previous_dm2),
                )

            self.history.append(
                {
                    "cycle": cycle + 1,
                    "e_ensemble": ensemble_energy,
                    "delta_e": delta_energy,
                    "delta_rotation": delta_rotation,
                    "norm_ddm": norm_ddm,
                    "residual": residual,
                    "root_assignment": root_assignment.copy(),
                }
            )
            log.info(
                "cycle= %d E(L-PDFT)= %.15g  dE= %4.3g  dR= %4.3g  "
                "|ddm|= %4.3g  |residual|= %4.3g",
                cycle + 1,
                ensemble_energy,
                delta_energy,
                delta_rotation,
                norm_ddm,
                residual,
            )

            if (
                previous_energy is not None
                and delta_energy < self.conv_tol
                and delta_rotation < self.conv_tol_rotation
                and norm_ddm < self.conv_tol_rdm
                and residual < self.conv_tol_residual
            ):
                self.converged = True
                break

            if cycle == self.max_cycle - 1:
                break

            previous_energy = ensemble_energy
            previous_dm1s = casdm1s
            previous_dm2 = casdm2
            current_ci = self._rotate_ci(current_ci, rotation)

        self.niter = cycle + 1
        current_ci, ham = self._canonicalize_equal_weight_blocks(current_ci, ham)
        self.ci = self._copy_ci(current_ci)
        self.lpdft_ham = ham
        self.residual = residual
        self.root_assignment = root_assignment.copy()
        # At self-consistency the Hamiltonian is diagonal between states with
        # different weights.  Its diagonal therefore gives the state energies
        # associated with the converged, weight-labelled states.
        self.e_states = np.real(np.diag(ham)).copy()
        self.e_tot = float(np.dot(self.weights, self.e_states))
        self.e_ensemble = ensemble_energy

        if self.converged:
            log.note("LPDFSCF converged in %d cycles", self.niter)
        else:
            log.warn("LPDFSCF did not converge in %d cycles", self.niter)

        return self.e_states, self.ci

    kernel = kernel_self_consistent

    def run(self, ci=None):
        """Run :meth:`kernel_self_consistent` and return this object."""
        self.kernel_self_consistent(ci=ci)
        return self


__all__ = ["LPDFSCF"]
