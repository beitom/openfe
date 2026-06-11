# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe
"""Tests for shared charge correction utilities."""

import numpy as np
import pytest
from openff.units import unit
from openmm import NonbondedForce, app

from openfe.protocols.openmm_utils.charge_correction import (
    get_alchemical_waters,
    get_water_atom_indices,
)


class TestGetAlchemicalWaters:
    """Tests for get_alchemical_waters."""

    def test_zero_charge_difference_returns_empty(self):
        """No waters needed when charge difference is 0."""
        # Doesn't matter what topology/positions are since we return early
        result = get_alchemical_waters(
            topology=None,  # won't be used
            positions=None,
            charge_difference=0,
        )
        assert result == []

    def test_raises_if_no_qualifying_waters(self):
        """Should raise if no waters outside the cutoff."""
        # Create a minimal topology with a single water too close to solute
        from openmm import app as omm_app

        topology = omm_app.Topology()
        chain = topology.addChain()
        # Solute residue
        res_sol = topology.addResidue("LIG", chain)
        topology.addAtom("C", omm_app.Element.getBySymbol("C"), res_sol)
        # Water very close to solute
        res_wat = topology.addResidue("HOH", chain)
        topology.addAtom("O", omm_app.Element.getBySymbol("O"), res_wat)
        topology.addAtom("H1", omm_app.Element.getBySymbol("H"), res_wat)
        topology.addAtom("H2", omm_app.Element.getBySymbol("H"), res_wat)
        # Add bonds for water
        atoms = list(topology.atoms())
        topology.addBond(atoms[1], atoms[2])
        topology.addBond(atoms[1], atoms[3])

        # Box vectors needed for periodic
        topology.setPeriodicBoxVectors(
            [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]
        )

        # Positions: solute at origin, water at 0.01 nm away
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.01, 0.01, 0.0],
            [0.01, -0.01, 0.0],
        ])

        with pytest.raises(ValueError, match="no waters outside"):
            get_alchemical_waters(
                topology=topology,
                positions=positions,
                charge_difference=1,
                distance_cutoff=0.8 * unit.nanometer,
            )
